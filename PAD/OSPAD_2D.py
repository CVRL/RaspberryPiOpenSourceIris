import os
import argparse
import cv2
import bsif
import numpy as np
import multiprocessing

class OSPAD_2D(object):
    def __init__(self, cfg):
        self.modelDir = cfg["modelDir"]
        self.bitSizes = cfg["bitSizes"]
        self.modelSizes = cfg["modelSizes"]
        self.modelTypes = cfg["modelTypes"]
        self.segmentation = cfg["segmentation"]
        
        self.models = []
        self.filters = []
        for i in range(len(self.bitSizes)):
            # Load filters
            size = self.modelSizes[i]
            bits = self.bitSizes[i]
            
            if size % 2 == 0:
                size = size//2
            filter = bsif.load(size, bits)
            
            # Initialize current filter
            filterset = []
            currentFilter = np.empty((size, size), np.float64)
            filterNum = bits - 1
            
            while (filterNum >= 0):
                # Load current filter (need to do it this way due to the storage of the filter - matlab file)
                for row in range(size):
                    for col in range(size):
                        currentFilter[row, col] = filter[self.s2i(size, bits, row, col, filterNum)]
                # Move to next filter
                filterNum -= 1
                filterset.append(np.copy(currentFilter))
            
            self.filters.append(filterset)
            
            # Load models
            model_filename = "BSIF-" + str(self.bitSizes[i]) + "-" + str(self.modelSizes[i]) + "-" + self.modelTypes[i] + "-" + self.segmentation + ".xml"
            exists = os.path.isfile(self.modelDir + model_filename) 
            if exists:
                if self.modelTypes[i] == "svm":
                    self.models.append(cv2.ml.SVM_load(self.modelDir + model_filename)) 
                elif self.modelTypes[i] == "rf":
                    self.models.append(cv2.ml.RTrees_load(self.modelDir + model_filename)) 
                elif self.modelTypes[i] == "mlp":
                    self.models.append(cv2.ml.ANN_MLP_load(self.modelDir + model_filename))
        
        print("Initialized OSPAD 2D")
        
    def s2i(self, size, bits, row, col, bit):
        # C + + and python use row - major order, so the last dimension is contiguous
        # in doubt, refer to https: // en.wikipedia.org / wiki / Row - _and_column - major_order  # Column-major_order
        return (bit + bits * (col + size * row))
    
    def generateHistogram(self, src, i, segmentationType):
        size = self.modelSizes[i]
        bits = self.bitSizes[i]
        # load image
        if (segmentationType == "bg"):
            src = src[125:375, 195:445]

        downsample = False
        if ((size%2) == 0):
            downsample = True
            size = size//2

        if downsample:
            # Downsample (size is defined as x,y for this function not row,col)
            src = cv2.pyrDown(src, dstsize=(src.shape[1]//2, src.shape[0]//2))

        # initialize matrix of ones (to hold BSIF results)
        codeImg = np.ones((src.shape[0], src.shape[1]), np.int64)

        # Create wrapping border around the image
        # e.g. if size is 3x3, want a border of 1 to account for edges
        border = int(size) // 2
        
        imgWrap = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_WRAP)

        # Loop through filters, starting with the last one
        filterNum = bits - 1
        while (filterNum >= 0):
            # Filter with filter2d - need to specify no image wrapping since we have done this previously
            # using default kernel anchor (centerpoint is anchor)
            #print(self.filters[i][bits-1-filterNum])
            filteredImg = cv2.filter2D(imgWrap, ddepth=cv2.CV_64F,kernel=self.filters[i][bits-1-filterNum],delta=0,borderType=cv2.BORDER_CONSTANT)
            
            # Convert any positive values in the matrix to 2^(i-1) as in the matlab software
            binary_add = 2**(bits - 1 - filterNum)
            #indices = np.where(filteredImg[border:-border, border:-border] > 0.001)
            codeImg[filteredImg[border:-border, border:-border] > 0.001] += binary_add
            
            # Move to next filter
            filterNum -= 1

        # Create the histogram
        bins = 2**bits # for example, 256 bins for 8 bit filters (1-256)
        # This occurs because the image is initialized to ones, so zero position will need to be ignored

        hist = np.histogram(codeImg, bins=bins)
        hist = np.asarray(hist[0])

        # normalize
        mean = np.mean(hist)
        std = np.std(hist)
        hist = (hist - mean) / std

        return hist

          
    def predict(self, im_left, im_right):
        # print("Running OSPAD_2D...")
        # prediction vectors
        predictions = []

        # extract features
        #im_left = cv2.imread((left_im_name),flags=0)
        #im_right = cv2.imread((right_im_name),flags=0)

        for i in range(len(self.bitSizes)):
            with multiprocessing.Pool(4) as p:
                [hist_left, hist_right] = p.starmap(self.generateHistogram, [(im_left, i, self.segmentation), (im_right, i, self.segmentation)])
            #hist_left = self.generateHistogram(im_left, i, self.segmentation)
            #hist_right = self.generateHistogram(im_right, i, self.segmentation)
            features = np.stack([hist_left, hist_right], axis = 0).astype(np.float32)
            
            if self.modelTypes[i] == "mlp":
                # predict with MLP
                cur_predictions = self.models[i].predict(features)[1]
                preds = []
                for k in range(len(cur_predictions)):
                    if cur_predictions[k,0] > 0.8:
                        preds.append(1)
                    else:
                        preds.append(0)
                predictions += preds
            else:
                # predict with svm or rf
                cur_predictions = self.models[i].predict(features)[1]
                # need to flatten this list
                preds = [predict for sublist in cur_predictions for predict in sublist]
                predictions += preds
            
            # Early stopping if supporters for a decision has exceeded half of the number of voters
            if sum(predictions) > len(self.bitSizes) or len(predictions) - sum(predictions) > len(self.bitSizes):
                break
        
        score = sum(predictions) - len(self.bitSizes)

        return score
    
