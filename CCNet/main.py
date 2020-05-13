import numpy as np
import torch
import os
import csv
import math
import random
import glob

from PIL import Image
from argparse import ArgumentParser
from skimage import transform
from scipy.stats.mstats import mquantiles
import torch.nn as nn

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, ToPILImage

from modules.dataset import IrisSegmDataset
from modules.network import UNet
from modules.criterion import CrossEntropyLoss2d
from modules.transform import Relabel, ToLabel

NUM_CHANNELS = 1
NUM_CLASSES = 2
cvParam = 0.9
ins = 256

output_feature = []


image_transform = ToPILImage()
input_transform = Compose([
    ToTensor(),
])

target_transform = Compose([
    ToLabel(),
    Relabel(255, 1),
])

def train(args, model):

    directory = os.path.dirname("checkpoint/")
    if not os.path.exists(directory):
        os.makedirs(directory)
   
    weight = torch.ones(NUM_CLASSES)

    loader = DataLoader(IrisSegmDataset(args.image_dir, args.mask_dir, args.dataset_metadata, input_transform, target_transform, cvParam, True), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(IrisSegmDataset(args.image_dir, args.mask_dir, args.dataset_metadata, input_transform, target_transform, cvParam, False), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    softmax = nn.LogSoftmax(dim=1)
    print('training length:', len(loader))
    print('validation length:', len(val_loader))

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)
    optimizer = Adam(model.parameters(),lr = 0.00002)

    for epoch in range(1, args.num_epochs+1):

        model.train()

        epoch_loss = []
        val_epoch_loss = []
        train_IoU = 0
        val_IoU = 0
        for batch, data in enumerate(loader):
            # setup input
            images = data["image"]
            labels = torch.LongTensor(data["mask"].long())
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            # push through network and optimize
            inputs = Variable(images)
            targets = Variable(labels)     
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:,0])
            loss.backward()
            optimizer.step()

            # compute IoU
            logprob = softmax(outputs).data.cpu().numpy()
            pred = np.argmax(logprob, axis=1)
            gt = targets[:,0].data.cpu().numpy()
            IoU = np.sum(np.logical_and(pred,gt)) / np.sum(np.logical_or(pred,gt))
            train_IoU += IoU
 
            epoch_loss.append(loss.item())
            
            if batch % args.log_batch== 0:
                train_loss_average = sum(epoch_loss) / len(epoch_loss)
                print("Train loss: {aver} (epoch: {epoch}, batch: {batch})".format(aver = train_loss_average, epoch = epoch, batch = batch))

        train_IoU = train_IoU / len(loader)

        # Save checkpoint
        if epoch > 0 and epoch % args.save_epoch == 0:
            filename ="checkpoint/" + "{model}-{epoch:03}.pth".format(model = args.model, epoch = epoch)
            torch.save(model.state_dict(), filename)

        # Validation set
        if len(val_loader) > 0:
            for batch, data in enumerate(val_loader):
                # setup input
                images = data["image"]
                labels = torch.LongTensor(data["mask"].long())
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                
                # push through network and compute loss
                inputs = Variable(images)
                targets = Variable(labels)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets[:,0])
                val_epoch_loss.append(val_loss.item())

                # compute IoU
                logprob = softmax(outputs).data.cpu().numpy()
                pred = np.argmax(logprob, axis=1)
                gt = targets[:,0].data.cpu().numpy()
                IoU = np.sum(np.logical_and(pred,gt)) / np.sum(np.logical_or(pred,gt))
                val_IoU += IoU
                
            val_loss_average = sum(val_epoch_loss) / len(val_epoch_loss)
            val_IoU /= len(val_loader)
            print("Val loss: {aver} (epoch: {epoch})".format(aver = val_loss_average, epoch = epoch))
        
        with open("loss.csv", "a") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow([epoch, train_loss_average, val_loss_average, train_IoU, val_IoU])

def evaluate(args, model):
    image_root = "/afs/crc.nd.edu/user/z/zfang/irisDataCollection/images/"
    save_root = "/afs/crc.nd.edu/user/z/zfang/irisDataCollection/ccnet1500_res/"
    
    with open('LG4000_meta.csv', 'rb') as f:
        lines = f.readlines()
    meta = list(map(lambda l: [i.rstrip('\n') for i in l.decode("utf-8").split(',')], lines))
    meta = meta[1:]

    model.eval()

    softmax = nn.LogSoftmax(dim=1)
    for m in meta:
        # Fetch input image
        image_path = image_root + m[0][2:]
        image = Image.open(image_path).convert('L')
        image = image.resize((320, 240), Image.BILINEAR)
        image = input_transform(image).unsqueeze(0)
        if args.cuda:
            image = image.cuda()
        image = Variable(image)

        # Run through the network
        print(image.shape)
        outputs = model(image)
        logprob = softmax(outputs).data.cpu().numpy()
        pred = np.argmax(logprob, axis=1)*255
        im = Image.fromarray(pred[0].astype(np.uint8))
        out_path = save_root + m[0][2:].split('/')[-1].split('.')[0] + '_mask.bmp'
        im.save(out_path)

    print("Test set images saved!")

    return None

def main(args):
    model = None
    if args.model == 'unet':
        model = UNet(NUM_CLASSES, NUM_CHANNELS)
        print('unet')

    if args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    if args.state:
        try:
            if args.cuda:
                model.load_state_dict(torch.load(args.state))
            else:
                model.load_state_dict(torch.load(args.state, map_location=torch.device('cpu')))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))

    print(model)

    if args.mode == 'eval':
        evaluate(args, model)
    if args.mode == 'train':
        train(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state')
    parser.add_argument('--mode', default = "train")
    parser.add_argument('--image_dir')
    parser.add_argument('--mask_dir')
    parser.add_argument('--dataset_metadata')
    parser.add_argument('--lr', type=float, default = 0.0001)
    parser.add_argument('--num_epochs', type=int, default=2001)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--log_batch', type=int, default=50)
    parser.add_argument('--save_epoch', type=int, default=50)
    parser.add_argument('--eval_epoch', type=int, default=50)
    parser.add_argument('--test_path')

    main(parser.parse_args())

