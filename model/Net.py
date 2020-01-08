import torch
import torch.nn as nn
from torchvision import transforms
from model.Transform import *
from model.SANet import *
from model.Decoder import *
from model.VGG import *
import numpy as np

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        vgg = encoder
        # self.enc_0 = nn.Sequential(*list(vgg.children())[:1])
        # enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1
        # transform
        self.transform = Transform(in_planes=512)
        self.decoder = decoder
        if (start_iter > 0):
            self.transform.load_state_dict(torch.load('weight/transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('weight/decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        self.variation_loss = nn.L1Loss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.dx_bias = np.zeros([256, 256])
        self.dy_bias = np.zeros([256, 256])
        for i in range(256):
            self.dx_bias[:, i] = i
            self.dx_bias[i, :] = i

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target, norm=False):
        if (norm == False):
            return self.mse_loss(input, target)
        else:
            return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_temporal_loss(self,  x1, x2):
        h = x1.shape[2]
        w = x1.shape[3]
        D = h*w
        return self.mse_loss(x1, x2)

    def compute_total_variation_loss_l1(self, inputs):
        h = inputs.shape[2]
        w = inputs.shape[3]
        h1 = inputs[:, :, 0:h-1, :]
        h2 = inputs[:, :, 1:h, :]
        w1 = inputs[:, :, :, 0:w-1]
        w2 = inputs[:, :, :, 1:w]
        return self.variation_loss(h1, h2)+self.variation_loss(w1, w2)

    def forward(self, content, style, content2=None, video=False):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        stylized = self.transform(content_feats[3], style_feats[3], content_feats[4], style_feats[4])
        g_t = self.decoder(stylized)
        loss_v = self.compute_total_variation_loss_l1(g_t)
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3], norm=True) + self.calc_content_loss(
            g_t_feats[4], content_feats[4], norm=True)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        """IDENTITY LOSSES"""
        Icc = self.decoder(self.transform(content_feats[3], content_feats[3], content_feats[4], content_feats[4]))
        Iss = self.decoder(self.transform(style_feats[3], style_feats[3], style_feats[4], style_feats[4]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i],
                                                                                                     style_feats[i])
        if video==False:
            return loss_c, loss_s, l_identity1, l_identity2, loss_v
        else:
            content_feats2 = self.encode_with_intermediate(content2)
            stylized2 = self.transform(content_feats2[3], style_feats[3], content_feats2[4], style_feats[4])
            g_t2 = self.decoder(stylized2)
            g_t2_feats = self.encode_with_intermediate(g_t2)

            temporal_loss = self.calc_temporal_loss(g_t_feats[0], g_t2_feats[0])


            return loss_c, loss_s, l_identity1, l_identity2, temporal_loss, loss_v



