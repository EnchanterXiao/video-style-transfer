import argparse
from model.VGG import *
from model.Decoder import *
from model.Transform import *
import os
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import cv2 as cv
import numpy as np
import time

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = 'input/test3.avi',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = 'style/style11.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'weight/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'experiments4/decoder_iter_600000.pth')
parser.add_argument('--transform', type=str, default = 'experiments4/transformer_iter_600000.pth')
# Additional options
parser.add_argument('--save_ext', default = 'output+',
                    help='The extension name of the output viedo')
parser.add_argument('--output', type=str, default = 'output',
                    help='Directory to save the output image(s)')

# Advanced options
args = parser.parse_args('')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(args.output):
    os.mkdir(args.output)
decoder = Decoder('Decoder')
transform = Transform(in_planes=512)
vgg = VGG('VGG19')

decoder.eval()
transform.eval()
vgg.eval()

# decoder.features.load_state_dict(torch.load(args.decoder))
decoder.load_state_dict(torch.load(args.decoder))
transform.load_state_dict(torch.load(args.transform))
vgg.features.load_state_dict(torch.load(args.vgg))

enc_1 = nn.Sequential(*list(vgg.features.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.features.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.features.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.features.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.features.children())[31:44])  # relu4_1 -> relu5_1


enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

cap = cv.VideoCapture(args.content)

style = style_tf(Image.open(args.style))

style = style.to(device).unsqueeze(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv.CAP_PROP_FPS)

out = cv.VideoWriter(args.save_ext, fourcc, fps, (512, 512))

fps = 0
while(1):
    fps += 1
    ret, frame = cap.read()
    if ret!=True:
        break
    frame = cv.resize(frame, (512, 512))
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = frame.astype(np.float) / 255
    content = torch.from_numpy(frame)
    content = content.transpose(0, 2)
    content = content.transpose(1, 2)
    content = content.type(torch.FloatTensor)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        start = time.time()

        Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
        Style5_1 = enc_5(Style4_1)
        Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
        Content5_1 = enc_5(Content4_1)

        content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))

        end = time.time()
        content.clamp(0, 255)
        content = content.cpu()
        content = content[0]
        content = content.transpose(1, 2)
        content = content.transpose(0, 2)
        content = content.numpy()*255

        output_value = np.clip(content, 0, 255).astype(np.uint8)
        output_value = cv.cvtColor(output_value, cv.COLOR_RGB2BGR)

        out.write(output_value)
        print('frame(%s) transfer! fps:%.2f' % (fps,1/(end-start)))