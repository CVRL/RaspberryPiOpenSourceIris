import numpy as np 
import os
from PIL import Image
import time

class OsIris(object):
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get_mask(self, img_paths):
        # Call OSIRIS from command line
        txt_path = self.cfg["imagelist_path"] + str(int(time.time()*1000000%1000))+ '.txt'
        while os.path.exists(txt_path):
            txt_path = self.cfg["imagelist_path"] + str(int(time.time()*1000000%1000))+ '.txt'
        with open(txt_path, 'w') as f:
            for img_path in img_paths:
                f.write(img_path + "\n")
        
        os.chdir("/home/pi/Desktop/iris/OSIRIS_SEGM/src")
        os.system("./osiris "+ txt_path)
        
        masks = []
        for img_path in img_paths:
            img_name = self.cfg["output_path"] + img_path.split('/')[-1].split('.')[0]
            mask = np.array(Image.open(img_name+self.cfg["mask_suffix"]).convert('L'))
            mask = mask[:,:,0] if len(mask.shape) == 3 else mask
            masks.append(mask)
            
        return masks
    
    def get_circle(self, img_path):
        # Get circles from the circle coordinates given by OSIRIS
        img_name = self.cfg["output_path"] + img_path.split('/')[-1].split('.')[0]
        
        with open(img_name+self.cfg['param_suffix'], 'rb') as f:
            lines = f.readlines()
            lines = list(map(lambda l: [l.decode("utf-8").rstrip('\n')], lines))
            
        pupil_num = int(lines[0][0])
        iris_num = int(lines[1][0])
        pupil_coords = np.array(list(int(float(a)) for a in lines[2][0].split( ))).reshape(-1, 3)
        iris_coords = np.array(list(int(float(a)) for a in lines[3][0].split( ))).reshape(-1, 3)
        
        pupil_xyr = np.mean(pupil_coords, axis = 0).astype(int)
        pupil_xyr[2] = np.sqrt((pupil_coords[0,0]-pupil_xyr[0])**2 + (pupil_coords[0,1]-pupil_xyr[1])**2)
        pupil_xyr = [int(a) for a in pupil_xyr]
        
        iris_xyr = np.mean(iris_coords, axis = 0).astype(int)
        iris_xyr[2] = np.sqrt((iris_coords[0,0]-iris_xyr[0])**2 + (iris_coords[0,1]-iris_xyr[1])**2)
        iris_xyr = [int(a) for a in iris_xyr]
        
        return np.array(pupil_xyr), np.array(iris_xyr)
            
        