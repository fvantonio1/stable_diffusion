from glob import glob
import os
import torchvision
import torch
from tqdm import tqdm
from PIL import Image
import random

class MnistDataLoader:

    def __init__(self, batch_size, split, im_path='data/MNIST'):
        assert split in {'train', 'test'}

        self.split = split
        self.batch_size = batch_size

        self.current_position = 0
        
        im_path = os.path.join(im_path, split)
        self.images, self.labels = self.load_images(im_path)

    def reset(self):
        self.current_position = 0

    def load_images(self, im_path):
        assert os.path.exists(im_path), f"images path {im_path} does not exists"

        ims = []
        labels = []

        fnames = glob(os.path.join(im_path,'*/*.png'))
        random.shuffle(fnames)

        for fname in tqdm(fnames, desc='Reading Images'):

            im = Image.open(fname)
            im_tensor = torchvision.transforms.ToTensor()(im)

            im.close()
            # convert input to (-1, 1) range
            im_tensor = (2 * im_tensor) - 1

            ims.append(im_tensor)

            label = fname.split('/')[-2]
            labels.append(int(label))

        print(f"Found {len(ims)} images for split {self.split}")

        ims = torch.stack(ims, dim=0)
        labels = torch.tensor(labels, dtype=torch.int16)

        return ims, labels
    
    def next_batch(self):
        B = self.batch_size

        images = self.images[self.current_position: self.current_position + B]
        labels = self.labels[self.current_position: self.current_position + B]

        self.current_position += B

        if self.current_position + B > len(self.images):
            self.current_position = 0

        return images, labels
    
    def __len__(self):
        return len(self.images) // self.batch_size

