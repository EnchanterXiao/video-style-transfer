import numpy as np
import argparse
import os
import torch
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from model.Decoder import *
from model.VGG import *
from model.SANet import *
from model.Net import *
from dataset.dataset import *

import numpy as np
from torch.utils import data

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/media/wwh/XIaoxin/Datasets/coco_2014/data/coco_2014/images/train2014/',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/home/wwh/Desktop/wikiArt/',
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='weight/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1600000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--style_weight', type=float, default=3.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--start_iter', type=float, default=500000)
args = parser.parse_args('')

device = torch.device('cuda')

decoder = Decoder('Decoder')
vgg = VGG('VGG19')

vgg.features.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.features.children())[:44])
network = Net(vgg, decoder, args.start_iter)

network.train()
network.to(device)

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

optimizer = torch.optim.Adam([
                              {'params': network.decoder.parameters()},
                              {'params': network.transform.parameters()}], lr=args.lr)

# if(args.start_iter > 0):
#     optimizer.load_state_dict(torch.load('optimizer_iter_' + str(args.start_iter) + '.pth'))

writer = SummaryWriter('runs/loss4')

for i in tqdm(range(args.start_iter, args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    print(content_images.shape)
    print(style_images.shape)
    loss_c, loss_s, l_identity1, l_identity2, T_loss = network(content_images, style_images, content_images, True)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + l_identity1 * 50 + l_identity2 * 1
    writer.add_scalar('total loss', loss, global_step=i)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.transform.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = optimizer.state_dict()
        torch.save(state_dict,
                   '{:s}/optimizer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

writer.close()