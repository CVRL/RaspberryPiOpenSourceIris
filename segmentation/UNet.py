import tensorflow as tf
import numpy as np
from PIL import Image
import time
import cv2

# CCNet
class UNet(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.load_graph(self.cfg["unet_model_path"])
        
    def load_graph(self, model_filepath):
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)
        
        with tf.gfile.GFile(model_filepath, 'rb') as f:
            gf = tf.GraphDef()
            gf.ParseFromString(f.read())
        
        self.input = tf.placeholder(np.float32, shape = [None, 1, 240, 320], name = 'test_input')
        
        tf.import_graph_def(gf, {'0': self.input})
        
        self.output_tensor = self.graph.get_tensor_by_name("import/231:0")
        
    def get_mask(self, imgs):
        masks = []
        for idx, img in enumerate(imgs):
            img = np.array(img.resize((320, 240), Image.BILINEAR))
            img = img / 255
            out = self.sess.run(self.output_tensor, feed_dict={self.input: img.reshape(1,1,240,320)})
            out = np.argmax(out, axis = 1).astype('uint8') * 255
            out = out.reshape(240,320)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(out, connectivity=8)
            sizes = stats[1:,-1]
            nb_components -= 1
            min_size = 500

            for i in range(0, nb_components):
                if sizes[i] <= min_size:
                    out[output==i+1] = 0
            masks.append(out)
            
        return masks
    
    def get_circle(self, mask):
        # find iris circle
        mask_for_iris = 255 - mask
        iris_indices = np.where(mask_for_iris[40:-40, 40:-40] == 0)
        if len(iris_indices[0]) == 0:
            return None, None, None
        y_span = max(iris_indices[0]) - min(iris_indices[0])
        x_span = max(iris_indices[1]) - min(iris_indices[1])

        if x_span > y_span + 40:
            iris_radius_estimate = y_span // 2 + 20
        else:
            iris_radius_estimate = x_span // 2
        
        iris_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=self.cfg["iris_hough_param1"],
                                       param2=self.cfg["iris_hough_param2"],
                                       minRadius=iris_radius_estimate-self.cfg["iris_hough_lowermargin"],
                                       maxRadius=iris_radius_estimate+self.cfg["iris_hough_uppermargin"])
        if iris_circle is None:
            return None, None, None
        iris_x, iris_y, iris_r = np.rint(np.array(iris_circle[0][0])*2).astype(int)
        
        # find pupil circle
        pupil_circle = cv2.HoughCircles(mask_for_iris, cv2.HOUGH_GRADIENT, 1, 50,
                                        param1=self.cfg["pupil_hough_param1"],
                                        param2=self.cfg["pupil_hough_param2"],
                                        minRadius=self.cfg["pupil_hough_minimum"],
                                        maxRadius=iris_r//3+self.cfg["pupil_hough_margin"])
        if pupil_circle is None:
            return None, None, None
        pupil_x, pupil_y, pupil_r = np.rint(np.array(pupil_circle[0][0])*2).astype(int)
        
        if np.sqrt((pupil_x-iris_x)**2+(pupil_y-iris_y)**2) > self.cfg["max_separation"]:
            pupil_x = iris_x
            pupil_y = iris_y
            pupil_r = iris_r // 3
        mask = cv2.resize(mask, (640, 480), interpolation = cv2.INTER_AREA)
        
        return mask, np.array([pupil_x, pupil_y,pupil_r]), np.array([iris_x, iris_y,iris_r])