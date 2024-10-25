from glob import glob
import os
import torchvision
import torch
from tqdm import tqdm
from PIL import Image
import random

class CelebDataLoader:

    def __init__(self, batch_size, im_size=256, im_path='data/celeba', split='train'):
        self.im_size = im_size
        self.split = split
        self.batch_size = batch_size
        
        self.current_position = 0

        assert os.path.exists(im_path), f"images path {im_path} not exist"

        im_path = os.path.join(im_path, 'CelebA-HQ-img', '*.jpg')
        self.fnames = glob(im_path)
        print(f"Found {len(self.fnames)} images")
        random.shuffle(self.fnames)

    def load_batch(self, fnames):
        ims = []

        for fname in fnames:
            im = Image.open(fname)
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor()
            ])(im)

            im.close()
            # convert values to [-1,1]
            im_tensor = (2 * im_tensor) - 1
            ims.append(im_tensor)

        ims = torch.stack(ims, dim=0)

        return ims
    
    def next_batch(self):
        B = self.batch_size

        batch_fnames = self.fnames[self.current_position: self.current_position + B]

        images = self.load_batch(batch_fnames)

        self.current_position += B

        if self.current_position + B > len(self.fnames):
            self.current_position = 0

        return images, None
    
    def __len__(self):
        return len(self.fnames) // self.batch_size

