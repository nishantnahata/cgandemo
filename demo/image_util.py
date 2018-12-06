import tensorflow as tf
import os
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_image(image):

    w = tf.shape(image)[1]

    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    input_image = tf.image.resize_images(input_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                         align_corners=True,
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize_images(real_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                        align_corners=True,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # normalizing the images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

