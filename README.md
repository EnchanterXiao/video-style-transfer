# video-style-transfer
This is video style transfer PyTorch implementation based on  "Arbitrary Style Transfer with Style-Attentional Networks".

Official paper: https://arxiv.org/abs/1812.02342v5.

Source code: https://github.com/GlebBrykin/SANET

## Dataset:
COCO  
WikiArt  
Video sequence（60 videos， from https://www.videvo.net/）  

## Modify:
Add temporal loss and Spatial smoothing loss to fine-tune.  
Use image pair from video to fine-tune.   

## Train:
Image_train: COCO+WikiArt  
video_train: Video sequence+WikiArt  

## Test:
Image_transfer: single image transfer  
video_transfer: video transfer  

## Result:
demo1:https://pan.baidu.com/s/1o40EPY7_6FnMKsnjGaD24Q  
demo2:https://pan.baidu.com/s/1ZMPegXQCBB35NimzmEg2fQ  

