import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import argparse
import cv2
import numpy as np
import scipy
import scipy.io
from PIL import Image
from gpiozero import LED
from time import sleep
from utils.utils import get_cfg, pi_camera, xyr_from_txt
from segmentation.SegNet import SegNet
from segmentation.UNet import UNet
from PAD.OSPAD_2D import OSPAD_2D
from PAD.OSPAD_3D import OSPAD_3D
from recognition.IrisRecognition import IrisRecognition
from OSIRIS_SEGM.OSIRIS_SEGM import OsIris
import multiprocessing
from multiprocessing import Pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for the Open Source Iris System")
    parser.add_argument("--cfg_path",
                        type=str,
                        default="/home/pi/Desktop/iris/cfg/cfg.yaml",
                        help="path of the configuration file")
    args = parser.parse_args()
    cfg = get_cfg(args.cfg_path)
    
    '''
    cam = pi_camera(cfg)
    cam.capture_images('left')
    segnet = SegNet(cfg)
    
    img1_name = "{0}/{1}".format(cfg["image_save_dir"], "left.jpg")
    img2_name = "{0}/{1}".format(cfg["image_save_dir"], "right.jpg")
    
    img1 = Image.open(img1_name).convert('L')
    mask1, pupil_xyr1, iris_xyr1 = segnet.get_seg(img1)
    img1 = np.array(img1)
    img1 = img1[:,:,0] if len(img1.shape)==3 else img1
    
    img2 = Image.open(img2_name).convert('L')
    mask2, pupil_xyr2, iris_xyr2 = segnet.get_seg(img2)
    img2 = np.array(img2)
    img2 = img2[:,:,0] if len(img2.shape)==3 else img2
    
    imgs = np.stack([img1, img2], axis=2)
    masks = np.stack([mask1, mask2], axis=2)
    pupil_xyr = [pupil_xyr1, pupil_xyr2]
    iris_xyr = [iris_xyr1, iris_xyr2]
    pad = OSPAD_3D(cfg)
    pad_score, normal = pad.predict(imgs, masks, pupil_xyr, iris_xyr, left = True)
    
    print("score =", pad_score)
    '''
    
    ### Test images for running time
    dataset = "WACV"
    if dataset == "WACV":
        # Load data for WACV
        with open('/home/pi/Desktop/iris/WACV_Data/metadata.csv', 'rb') as f:
            lines = f.readlines()
        meta = list(map(lambda l: [i.rstrip('\n') for i in l.decode("utf-8").split(',')], lines))
        meta = meta[1:]
    else:
        # Load data for NDIris3D
        with open('/media/pi/PKBACK/LG4000_meta.csv', 'rb') as f:
            lines = f.readlines()
        orig_meta = list(map(lambda l: [i.rstrip('\n') for i in l.decode("utf-8").split(',')], lines))
        x = scipy.io.loadmat('/media/pi/PKBACK/index_label.mat')
        index_cross = x['index_cross'].flatten()
        index_direct = x['index_direct'].flatten()
        meta = []
        for i in range(len(index_cross)):
            orig_cross = orig_meta[index_cross[i]]
            tmp_cross = [orig_cross[0], str(i+1), orig_cross[1]]
            tmp_cross.append('live' if orig_cross[5]=='no' else 'fake')
            tmp_cross.append(orig_cross[2])
            tmp_cross.append(orig_cross[3])
            tmp_cross.append(orig_cross[6])
            tmp_cross.append(orig_cross[-2])
            
            orig_direct = orig_meta[index_direct[i]]
            tmp_direct = [orig_direct[0], str(i+1), orig_direct[1]]
            tmp_direct.append('live' if orig_direct[5]=='no' else 'fake')
            tmp_direct.append(orig_direct[2])
            tmp_direct.append(orig_direct[3])
            tmp_direct.append(orig_direct[6])
            tmp_direct.append(orig_direct[-2])
            
            meta.append(tmp_cross)
            meta.append(tmp_direct)

    # Load segmentation method
    if cfg["use_segnet"]:
        segnet = SegNet(cfg)
    elif cfg["use_unet"]:
        unet = UNet(cfg)
    elif cfg["use_osiris"]:
        osiris = OsIris(cfg)
        pool = Pool()
    else:
        print("Please specify a segmentation method")
        sys.exit()

    # Load PAD methods
    ospad_3d = OSPAD_3D(cfg)
    ospad_2d = OSPAD_2D(cfg)
    ir = IrisRecognition(cfg)
    
    # time for segmentation (SegNet / UNet / OSIRIS)
    time_segnet = []
    time_unet = []
    time_osiris = []
    
    # time for PAD (OSPAD_3D / OSPAD_2D)
    time_ospad_3d = []
    time_ospad_2d = []
    
    # time for iris recognition
    time_ir = []
    
    # scores for 3D and errors for 2d
    real_scores_3D = []
    fake_scores_3D = []
    error_2d = 0
    
    print('Total pairs: {}'.format(len(meta)//2))

    # Test all images
    for idx in range(len(meta)//2):
        if dataset == "WACV":
            # Image names (WACV)
            img1_name = '/home/pi/Desktop/iris/WACV_Data/' + meta[2*idx][0] + '.tiff'
            img2_name = '/home/pi/Desktop/iris/WACV_Data/' + meta[2*idx+1][0] + '.tiff'
        else:
            # Image names (NDIris3D)
            img1_name = '/media/pi/PKBACK/' + meta[2*idx][0] 
            img2_name = '/media/pi/PKBACK/' + meta[2*idx+1][0] 
        
        # Load images and convert to grayscale
        img1 = Image.open(img1_name).convert('L')
        img2 = Image.open(img2_name).convert('L')
        
        # Perform segmentation
        if cfg["use_segnet"]:
            # Segmentation by SegNet
            segnet_start = time.time()
            mask1, mask2 = segnet.get_mask([img1, img2])
            mask1, pupil_xyr1, iris_xyr1 = segnet.get_circle(mask1)
            mask2, pupil_xyr2, iris_xyr2 = segnet.get_circle(mask2)
            time_segnet.append(time.time() - segnet_start)
        elif cfg["use_unet"]:
            # Segmentation by CC-Net
            unet_start = time.time()
            mask1, mask2 = unet.get_mask([img1, img2])
            mask1, pupil_xyr1, iris_xyr1 = unet.get_circle(mask1)
            mask2, pupil_xyr2, iris_xyr2 = unet.get_circle(mask2)
            time_unet.append(time.time() - unet_start)
        else:
            #Segmentation by OSIRIS
            osiris_start = time.time()
            res = pool.map(osiris.get_mask, [[img1_name], [img2_name]]) #for img in [img1_name, img2_name]]

            mask1 = res[0][0]
            mask2 = res[1][0]
            pupil_xyr1, iris_xyr1 = osiris.get_circle(img1_name)
            pupil_xyr2, iris_xyr2 = osiris.get_circle(img2_name)
            
            time_osiris.append(time.time() - osiris_start)
            os.system("rm "+cfg["output_path"]+'*')
            os.system("rm "+cfg["imagelist_path"]+"*.txt")
        
        # OSPAD 3D
        img1 = np.array(img1)
        img1 = img1[:,:,0] if len(img1.shape)==3 else img1
        img2 = np.array(img2)
        img2 = img2[:,:,0] if len(img2.shape)==3 else img2
        
        imgs = np.stack([img1, img2], axis=2)
        masks = np.stack([mask1, mask2], axis=2)
        pupil_xyr = [pupil_xyr1, pupil_xyr2]
        iris_xyr = [iris_xyr1, iris_xyr2]
        
        ospad3d_start = time.time()
        score_3d, normal = ospad_3d.predict(imgs, masks, pupil_xyr, iris_xyr, left = 'left' in img1_name.lower())
        time_ospad_3d.append(time.time() - ospad3d_start)
        if meta[2*idx][3] == 'live':
            real_scores_3D.append(score_3d)
        else:
            fake_scores_3D.append(score_3d)
        
        # OSPAD 2D
        ospad2d_start = time.time()
        score_2d = ospad_2d.predict(img1, img2)
        time_ospad_2d.append(time.time() - ospad2d_start)
        if score_2d < 0:
            if meta[2*idx][3] == 'fake':
                error_2d += 1
        else:
            if meta[2*idx][3] == 'live':
                error_2d += 1
        
        # Iris recognition (measure time for matching against a template)
        img1_norm = ir.get_rubbersheet(img1, pupil_xyr1[1], pupil_xyr1[0], pupil_xyr1[2], iris_xyr1[2])
        mask1_norm = ir.get_rubbersheet(mask1, pupil_xyr1[1], pupil_xyr1[0], pupil_xyr1[2], iris_xyr1[2])
        code1 = ir.extract_code(img1_norm) 
        
        ir_start = time.time()
        img2_norm = ir.get_rubbersheet(img2, pupil_xyr2[1], pupil_xyr2[0], pupil_xyr2[2], iris_xyr2[2])
        mask2_norm = ir.get_rubbersheet(mask2, pupil_xyr2[1], pupil_xyr2[0], pupil_xyr2[2], iris_xyr2[2])
        code2 = ir.extract_code(img2_norm)
        ir_score = ir.matchCodes(code1, code2, mask1_norm, mask2_norm)
        time_ir.append(time.time() - ir_start)
        
        # Print
        print("Sample {0}, {1}, 3D score = {2}, 2D score = {3}, ir score = {4}".format(idx, meta[2*idx][3], score_3d, score_2d, ir_score))
    
    # Running time summary
    if cfg["use_segnet"]:
        print('#######################################################')
        print('SegNet Summary')
        print('Avg time per pair =', sum(time_segnet)/(len(meta)//2))
        print('#######################################################')
    elif cfg["use_unet"]:
        print('#######################################################')
        print('UNet Summary')
        print('Avg time per pair =', sum(time_unet)/(len(meta)//2))
        print('#######################################################')
    else:
        print('#######################################################')
        print('OSIRIS Summary')
        print('Avg time per pair =', sum(time_osiris)/(len(meta)//2))
        print('#######################################################')
    
    scipy.io.savemat('times_all.mat',
                     mdict={'time_segnet': time_segnet,\
                            'time_unet': time_unet,\
                            'time_osiris': time_osiris,\
                            'time_ospad_3d': time_ospad_3d,\
                            'time_ospad_2d': time_ospad_2d,\
                            'time_ir': time_ir})
    
    
