"""
This file implements BSIF (http://www.ee.oulu.fi/~jkannala/bsif/bsif.html) histogram extraction.
"""
import bsif
import numpy as np
import cv2
import time

def s2i(size, bits, row, col, bit):
    # C + + and python use row - major order, so the last dimension is contiguous
    # in doubt, refer to https: // en.wikipedia.org / wiki / Row - _and_column - major_order  # Column-major_order
    return (bit + bits * (col + size * row))

def generateHistogram(src, size, bits, segmentationType):
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
    # Load hard coded filters
    filter = bsif.load(size, bits)
    print(filter)
 
    # Initialize current filter
    currentFilter = np.empty((size,size), np.float64)

    # Loop through filters, starting with the last one
    filterNum = bits - 1
    while (filterNum >= 0):
        # Load current filter (need to do it this way due to the storage of the filter - matlab file)
        for row in range(size):
            for col in range(size):
                currentFilter[row, col] = filter[s2i(size, bits, row, col, filterNum)]
                
        print(currentFilter)
        # Filter with filter2d - need to specify no image wrapping since we have done this previously
        # using default kernel anchor (centerpoint is anchor)
        filteredImg = cv2.filter2D(imgWrap, ddepth=cv2.CV_64F, kernel=currentFilter,delta=0,borderType=cv2.BORDER_CONSTANT)

        # Convert any positive values in the matrix to 2^(i-1) as in the matlab software
        for row in range(src.shape[0]):
            for col in range(src.shape[1]):
                # need to ignore the added border
                # for statements loop over the ORIGINAL image size so the end border (bottom and left) will be ignored
                if (filteredImg[(row + border), (col + border)] > 0.001):
                    # add the binary amount corresponding to the filter number
                    codeImg[row,col] += 2**(bits - 1 - filterNum)

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

if __name__ == "__main__":
    generateHistogram(np.random.rand(256,256), 22, 12, 'bg')
