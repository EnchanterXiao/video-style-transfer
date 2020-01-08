import numpy as np
import numpy as np
from torch.utils import data
from PIL import Image
from PIL import ImageFile
import torch.backends.cudnn as cudnn
from torchvision import transforms
import os



class Video_dataset(data.Dataset):
    def __init__(self, root, transform):
        super(Video_dataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        if index<1:
            path2 = self.paths[index]
        else:
            path2 = self.paths[index-1]
        img2 = Image.open(os.path.join(self.root, path2)).convert('RGB')
        img2 = self.transform(img2)
        return img, img2

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'MPISeq'


if __name__ == '__main__':
    seqs_str = '''alley_1
                    alley_2
                    ambush_2
                    ambush_4
                   ambush_5'''
    seqs = [seq.strip() for seq in seqs_str.split()]


