from utils import get_cfg, pi_camera
from PIL import Image
import cv2
import numpy as np

def show(x):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image',640,480)
    cv2.imshow('image',x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = get_cfg("/home/pi/Desktop/iris/cfg/cfg.yaml")
    cam = pi_camera(cfg)
    cam.capture_images('left')