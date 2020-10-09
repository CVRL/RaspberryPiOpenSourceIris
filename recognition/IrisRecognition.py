import numpy as np
import cv2
import scipy
import scipy.io
import scipy.signal

from PIL import Image
from utils import utils
import argparse
from segmentation.SegNet import SegNet
import multiprocessing

class IrisRecognition(object):
    def __init__(self, cfg):
        self.height = cfg["rubbersheet_height"]
        self.width = cfg["rubbersheet_width"]
        self.angles = angles = np.arange(0, 2 * np.pi, 2 * np.pi / self.width)
        self.cos_angles = np.zeros((self.width))
        self.sin_angles = np.zeros((self.width))
        for i in range(self.width):
            self.cos_angles[i] = np.cos(self.angles[i])
            self.sin_angles[i] = np.sin(self.angles[i])

        self.filter_size = cfg["recog_filter_size"]
        self.num_filters = cfg["recog_num_filters"]
        self.max_shift = cfg["recog_max_shift"]
        self.filter = scipy.io.loadmat(cfg["recog_bsif_dir"]+'ICAtextureFilters_{0}x{1}_{2}bit.mat'.format(self.filter_size, self.filter_size, self.num_filters))['ICAtextureFilters']
        print("Initialized IrisRecognition")

    # rubbersheet model for iris recognition
    def get_rubbersheet(self, image, cx, cy, pupil_r, iris_r):
        # Angle value
        rs = np.zeros((self.height, self.width), np.uint8)

        for j in range(self.height):
            rad = j /self.height

            x_lowers = cx + pupil_r * self.cos_angles
            y_lowers = cy + pupil_r * self.sin_angles
            x_uppers = cx + iris_r * self.cos_angles
            y_uppers = cy + iris_r * self.sin_angles

            # Fill in the rubbersheet
            Xc = (1 - rad) * x_lowers + rad * x_uppers
            Yc = (1 - rad) * y_lowers + rad * y_uppers

            rs[j, :] = image[Xc.astype(int), Yc.astype(int)]

        return rs

    # extract iris code
    def extract_code(self, rs):
        # wrap image
        r = int(np.floor(self.filter_size / 2));
        imgWrap = np.zeros((r*2+self.height, r*2+self.width))
        imgWrap[:r, :r] = rs[-r:, -r:]
        imgWrap[:r, r:-r] = rs[-r:, :]
        imgWrap[:r, -r:] = rs[-r:, :r]

        imgWrap[r:-r, :r] = rs[:, -r:]
        imgWrap[r:-r, r:-r] = rs
        imgWrap[r:-r, -r:] = rs[:, :r]

        imgWrap[-r:, :r] = rs[:r, -r:]
        imgWrap[-r:, r:-r] = rs[:r, :]
        imgWrap[-r:, -r:] = rs[:r, :r]

        # Loop over all kernels in the filter set
        codeBinary = np.zeros((self.height, self.width, self.num_filters))
        for i in range(1,self.num_filters+1):
            ci = scipy.signal.convolve2d(imgWrap, np.rot90(self.filter[:,:,self.num_filters-i],2), mode='valid')
            codeBinary[:,:,i-1] = ci>0

        return codeBinary

    def matchCodes(self, code1, code2, mask1, mask2):
        margin = int(np.ceil(self.filter_size/2))
        self.code1 = code1[margin:-margin, :, :]
        self.code2 = code2[margin:-margin, :, :]
        self.mask1 = mask1[margin:-margin, :]
        self.mask2 = mask2[margin:-margin, :]

        scoreC = np.zeros((self.num_filters, 2*self.max_shift+1))
        for shift in range(-self.max_shift, self.max_shift+1):
            andMasks = np.logical_and(self.mask1, np.roll(self.mask2, shift, axis=1))
            xorCodes = np.logical_xor(self.code1, np.roll(self.code2, shift, axis=1))
            xorCodesMasked = np.logical_and(xorCodes, np.tile(np.expand_dims(andMasks,axis=2),self.num_filters))
            scoreC[:,shift] = np.sum(xorCodesMasked, axis=(0,1)) / np.sum(andMasks)

        scoreC = np.min(np.mean(scoreC, axis=0))

        return scoreC
