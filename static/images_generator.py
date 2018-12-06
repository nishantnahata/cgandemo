import os
import numpy as np
import cv2

def load_image(image):

    w = image.shape[1]
    w = w // 2
    real_image = image[:, :w, :]
    return real_image

#os.mkdir('input_images')
images = []
for imagename in os.listdir('demo/input_images'):
	image = cv2.imread(os.path.join('demo/input_images', imagename))
	image = load_image(image)
	cv2.imwrite(os.path.join('input_images', imagename), image)

