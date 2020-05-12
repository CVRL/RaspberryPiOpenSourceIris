import cv2
import numpy as np
from numpy import linalg
import scipy
import scipy.io as scio
from PIL import Image

class OSPAD_3D(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
        # setup light
        self.tilt_left = self.cfg["tilt_left"]
        self.tilt_right = self.cfg["tilt_right"]
        self.slant = self.cfg["slant"]
        self.left_light_pinv = self.get_light_pinv(left_eye=True)
        self.right_light_pinv = self.get_light_pinv(left_eye=False)
        
        # set up morph
        self.kernel_erode = np.ones((9,9), np.uint8)
        self.kernel_dilate = np.ones((2,2), np.uint8)
        
        print("Initialized OSPAD 3D") 
        
    def get_light_pinv(self, left_eye = True):
        # Get light
        if left_eye:
            tilt = self.tilt_left
        else:
            tilt = self.tilt_right
        light_num = 2
        light = np.zeros([2,3])
        for ctr in range(0, light_num):
            light[ctr,0] = np.sin(np.deg2rad(self.slant[ctr]))*np.cos(np.deg2rad(tilt[ctr]))
            light[ctr,1] = np.sin(np.deg2rad(self.slant[ctr]))*np.sin(np.deg2rad(tilt[ctr]))
            light[ctr,2] = np.cos(np.deg2rad(self.slant[ctr]))
        
        light_pinv = linalg.pinv(light)
        
        return light_pinv

    def predict(self, imgs, masks, pupil_xyr, iris_xyr, left = True, iris_pct = 0.5):
        if left:
            light_pinv = self.left_light_pinv
        else:
            light_pinv = self.right_light_pinv
        
        images = np.zeros([256, 256, imgs.shape[2]])
        mask = np.ones([256, 256], np.uint8)
        
        # image preprocessing
        for ctr in range(imgs.shape[2]):
            # Read the mask
            mask_ctr = masks[:,:,ctr]
            
            # Get iris center
            iris_x = iris_xyr[ctr][0]
            iris_y = iris_xyr[ctr][1]
            
            # Calculate pupil center
            pupil_x = pupil_xyr[ctr][0]
            pupil_y = pupil_xyr[ctr][1]
            
            # Iris radius
            iris_radius = iris_xyr[ctr][2]
            iris_radius_internal = int(round(iris_radius * iris_pct))
            
            # Image cropping
            img_ctr = imgs[iris_y-iris_radius : iris_y+iris_radius, iris_x-iris_radius : iris_x+iris_radius, ctr].astype(np.double)
            img_ctr = cv2.resize(img_ctr, (256, 256), interpolation = cv2.INTER_AREA)
            images[:,:,ctr] = img_ctr
            
            # Larger pupil removal of mask
            mask_larger_pupil = np.copy(mask_ctr)
            for pixh in range(max(0, pupil_y - iris_radius_internal), min(480, pupil_y + iris_radius_internal)):
                for pixw in range(max(0, pupil_x - iris_radius_internal), min(640, pupil_x + iris_radius_internal)):
                    x = np.abs(pixw - pupil_x)
                    y = np.abs(pixh - pupil_y)
                    
                    # Mask the pixels within the internal iris
                    if np.sqrt(x**2 + y**2) < iris_radius_internal:
                        mask_larger_pupil[pixh, pixw] = 0
                    else:
                        break
            
            mask_larger_pupil = mask_larger_pupil[iris_y-iris_radius : iris_y+iris_radius, iris_x-iris_radius : iris_x+iris_radius]
            mask_larger_pupil = cv2.resize(mask_larger_pupil, (256, 256), interpolation = cv2.INTER_AREA)
            
            # Mask cropping
            mask_ctr = mask_ctr[iris_y-iris_radius : iris_y+iris_radius, iris_x-iris_radius : iris_x+iris_radius]
            mask_ctr = cv2.resize(mask_ctr, (256, 256), interpolation = cv2.INTER_AREA)
            
            # Morphological opening of the mask
            mask_morph_ctr = np.copy(mask_ctr)
            mask_morph_ctr = cv2.erode(mask_morph_ctr, self.kernel_erode, iterations = 1)
            mask_morph_ctr = cv2.dilate(mask_morph_ctr, self.kernel_dilate, iterations = 1)
            
            # Specular relfection removal
            img_spec = img_ctr > 240
            [r, c] = img_spec.nonzero()
            mask_ctr[r,c] = 0
            
            # Aggregate current mask on the overall mask
            mask *= mask_ctr*mask_morph_ctr*mask_larger_pupil
        
        [r, c] = mask.nonzero()
        
        # Calculate normal vectors
        normal = np.dot(images[r, c, :], np.transpose(light_pinv))
        normal = normal / linalg.norm(normal, axis=1)[:, np.newaxis]
        normal[np.isnan(normal)] = 0
        
        # Convert back to surface normal maps
        map_normal = np.zeros([images.shape[0], images.shape[1], 3])
        map_normal[r,c,:] = normal
        
        # Calculate the PAD score
        diff = normal - np.mean(normal, axis = 0)
        diff_norm = linalg.norm(diff, axis = 1)
        pad_score = np.var(diff_norm)
        return pad_score, map_normal
    
    
