import numpy as np
import os

import time
import glob
import math

from PIL import Image
import random 

import colorsys

from torch.utils.data import Dataset

EXTENSIONS = ['.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    flnm = "{bs}{ext}".format(bs = basename, ext = extension)
    return os.path.join(root, flnm)
    #return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


def HSVColor(img):
    if isinstance(img,Image.Image):
        r,g,b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(),g.getdata(),b.getdata()) :
            h,s,v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB',(r,g,b))
    else:
        return None

class NeuronEM2012(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None, cvParam = 0.75, Train = True):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = []

        all_file = os.path.join(self.images_root, '*.jpg')

        filename = glob.glob(all_file)
        numel = len(filename)

        print(math.floor(numel*cvParam))

        if Train:
           for i in range(0, math.floor(numel*cvParam)):
               self.filenames.append(image_basename(filename[i]))
        else:
          for i in range(math.floor(numel*cvParam)+1, numel):
               self.filenames.append(image_basename(filename[i]))
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
            image_hsv = HSVColor(image)
        with open(image_path(self.labels_root, filename, '_segmentation.png'), 'rb') as f:
            label = load_image(f).convert('L')

        # apply extra augmentation (same random operation on both image and label)
        
        # horizontal flip
        flip_prob = random.random()
        if flip_prob<0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
            image_hsv.transpose(Image.FLIP_LEFT_RIGHT) 
            label.transpose(Image.FLIP_LEFT_RIGHT)
        # rotation:
        deg = random.randint(-90, 90)
        image = image.rotate(deg, resample=Image.BICUBIC)
        image_hsv = image_hsv.rotate(deg, resample=Image.BICUBIC)
        label = label.rotate(deg, resample=Image.NEAREST)
        

        if self.input_transform is not None:
            image = self.input_transform(image)
            image_hsv = self.input_transform(image_hsv)
        if self.target_transform is not None:
            label = self.target_transform(label)

        image_rgb = np.asarray(image)
        image_hsv = np.asarray(image_hsv)
        final_image = np.concatenate((image_rgb,image_hsv),axis=0)

        return final_image, label
    def __len__(self):
        return len(self.filenames)


class Glands(Dataset):
    def __init__(self, root, input_transform=None, target_transform=None, cvParam = 0.8, Train = True):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')
        self.filenames = []

        all_file = os.path.join(self.images_root, '*.png')

        filename = sorted(glob.glob(all_file))
        numel = len(filename)
        #print('hello!')
        #print(numel)
        #print(cvParam) 
        #print(math.floor(numel*cvParam))
        #print('bye!')
        if Train:
           for i in range(0,math.floor(numel*cvParam)):
               self.filenames.append(image_basename(filename[i]))
        else:
          for i in range(math.floor(numel*cvParam)+1, numel):
               self.filenames.append(image_basename(filename[i]))

        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '_anno.png'), 'rb') as f:
            label = load_image(f).convert('L')

        # apply extra augmentation (same random operation on both image and label)
        
        # horizontal flip
        flip_prob = random.random()
        if flip_prob<0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT) 
            label.transpose(Image.FLIP_LEFT_RIGHT)
        # rotation:
        deg = random.randint(-90, 90)
        image = image.rotate(deg, resample=Image.BICUBIC)
        label = label.rotate(deg, resample=Image.NEAREST)
        

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
        
    def __len__(self):
        return len(self.filenames)

'''
class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)
'''
