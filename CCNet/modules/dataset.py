import numpy as np
import os

import time
import glob
import math

from PIL import Image, ImageFilter
import random 

import torch
from torch.utils.data import Dataset


def load_image(file):
    return Image.open(file).convert('L')

def image_path(root, basename, extension):
    flnm = "{bs}{ext}".format(bs = basename, ext = extension)
    return os.path.join(root, flnm)
    #return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class IrisSegmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filename, input_transform=None, target_transform=None, cvParam = 0.9, Train = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.input_transform = input_transform
        self.target_transform = target_transform

        with open(filename, 'rb') as f:
            lines = f.readlines()
            meta = list(map(lambda l: [i.rstrip('\n') for i in l.decode("utf-8").split(',')], lines))
            meta = meta[1:]

        self.image_paths = []
        self.mask_paths = []
        for m in meta:
            image_path = image_dir + m[0][2:]
            mask_path = mask_dir + m[0][2:]
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)

        # Split train and test
        if Train:
            self.image_paths = self.image_paths[:int(len(self.image_paths)*cvParam)]
            self.mask_paths = self.mask_paths[:int(len(self.mask_paths)*cvParam)]
        else:
            self.image_paths = self.image_paths[int(len(self.image_paths)*cvParam):]
            self.mask_paths = self.mask_paths[int(len(self.mask_paths)*cvParam):]
        
        self.length = len(self.image_paths)

    def __getitem__(self, index):
        # Get the image and the mask
        image = load_image(self.image_paths[index])
        mask = load_image(self.mask_paths[index])

        # Resize image and mask
        image = image.resize((320, 240), Image.BILINEAR)
        mask = mask.resize((320, 240), Image.NEAREST)

        # Data augmentation
        # horizontal flip
        if random.random() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT) 
            mask.transpose(Image.FLIP_LEFT_RIGHT)

        # gaussian blurring (degree 2,3,4) and edge enhancing
        if random.random() < 0.83:
            random_degree = np.random.choice([1,2,3,4,5])
            if random_degree >= 2 and random_degree <= 4:
                # gaussian blurring
                image = image.filter(ImageFilter.GaussianBlur(random_degree))
            elif random_degree == 1:
                image = image.filter(ImageFilter.EDGE_ENHANCE)
            else:
                image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # random cropping of the image
        if random.random() < 0.7:
            crop_left = np.random.randint(20)
            crop_top = np.random.randint(15)
            crop_right = 320 - np.random.randint(20)
            crop_bottom = 240 - np.random.randint(15)
            
            image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))
            image = image.resize((320, 240), Image.BILINEAR)
            mask = mask.resize((320, 240), Image.NEAREST)

        if self.input_transform is not None:
            image = self.input_transform(image)
        
        if self.target_transform is not None:
            mask = np.array(mask)
            mask[mask<128] = 0
            mask[mask>=128] = 1
            mask = self.target_transform(mask)

        return {"image": image, \
                "mask": mask}

    def __len__(self):
        return self.length
