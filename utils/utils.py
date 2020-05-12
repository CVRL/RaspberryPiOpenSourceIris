# created by zfang 2020/02/18 10:59pm
import yaml
from picamera import PiCamera
from gpiozero import LED
import numpy as np
from PIL import Image
import cv2

def get_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'))
    return cfg

def xyr_from_txt(param_path):
    with open(param_path, 'rb') as f:
        lines = f.readlines()
        lines = list(map(lambda l: [l.decode("utf-8").rstrip('\n').rstrip(' ')], lines))
    pupil_num = int(lines[0][0])
    iris_num = int(lines[1][0])
    pupil_coords = np.array(list(int(float(a)) for a in lines[2][0].split(' '))).reshape(-1,3)
    iris_coords = np.array(list(int(float(a)) for a in lines[3][0].split(' '))).reshape(-1,3)
    
    pupil_xyr = np.mean(pupil_coords, axis=0).astype(int)
    pupil_xyr[2] = np.sqrt((pupil_coords[0,0]-pupil_xyr[0])**2 + (pupil_coords[0,1]-pupil_xyr[1])**2)
    pupil_xyr = [int(a) for a in pupil_xyr]
    
    iris_xyr = np.mean(iris_coords, axis=0).astype(int)
    iris_xyr[2] = np.sqrt((iris_coords[0,0]-iris_xyr[0])**2 + (iris_coords[0,1]-iris_xyr[1])**2)
    iris_xyr = [int(a) for a in iris_xyr]
    
    return pupil_xyr, iris_xyr

def show(x):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image',640,480)
    cv2.imshow('image',x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save(x, name):
    x = Image.fromarray(x)
    x.save(name)

class pi_camera(object):
    def __init__(self, cfg):
        self.camera = PiCamera()
        self.camera.sensor_mode = 2
        self.camera.resolution = (2592, 1944)
        self.left_illum = LED(cfg["left_illum_gpio_pin"])
        self.right_illum = LED(cfg["right_illum_gpio_pin"])
        self.cfg = cfg
        
    def capture_images(self, image_prefix):
        
        # capture w/ left illum
        self.left_illum.on()
        self.right_illum.off()
        #self.camera.start_preview()
        self.camera.start_preview(fullscreen=False, window=(800,100,640,480))
        input("Press Enter to capture left illumination...")
        left_i_name = "{0}/{1}_l".format(self.cfg["image_save_dir"], image_prefix)
        self.camera.capture(left_i_name+'.jpg')
        
        # capture w/ right illum
        self.left_illum.off()
        self.right_illum.on()
        input("Press Enter to capture right illumination...")
        right_i_name = "{0}/{1}_r".format(self.cfg["image_save_dir"], image_prefix)
        self.camera.capture(right_i_name+'.jpg')
        self.camera.stop_preview()
        self.right_illum.off()
        
        # Post-process pictures
        left_i_name = "{0}/{1}_l".format(self.cfg["image_save_dir"], image_prefix)
        right_i_name = "{0}/{1}_r".format(self.cfg["image_save_dir"], image_prefix)
        self.post_process(left_i_name)
        self.post_process(right_i_name)
        
        print("Capture Complete!")
        
    def post_process(self, img_name):
        # Locate iris and save images
        o_img = Image.open(img_name+'.jpg').convert('L')
        o_img = np.array(o_img)
        img = (o_img < 50)*255
        img = img.astype('uint8')
        nb_components, output, stats, centoids = cv2.connectedComponentsWithStats(img, connectivity=8)
        sizes = stats[1:,-1]
        nb_components -=1
        max_size = 90000
        for i in range(nb_components):
            if sizes[i] > max_size:
                img[output==i+1] = 0

        kernel = np.ones((9,9),np.uint8)
        img = cv2.erode(img, kernel, iterations=2)

        center = np.mean(np.where(img == 255), axis=1).astype('int')
        img = Image.fromarray(o_img[center[0]-480:center[0]+480, center[1]-640:center[1]+640])
        img = img.resize((640, 480), Image.BILINEAR)
        img.save(img_name+'_processed.jpg')
