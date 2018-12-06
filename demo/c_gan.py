import tensorflow as tf

tf.enable_eager_execution()

import os
import numpy as np
import matplotlib.pyplot as plt
from cgandemo.settings import STATIC_ROOT
from .image_util import load_image

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 100
OUTPUT_CHANNELS = 3
checkpoint_dir = os.path.join(STATIC_ROOT, 'demo/checkpoints')
output_dir = os.path.join(STATIC_ROOT, 'static/demo/output_images')


class Downsample(tf.keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):

    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        if x2 is not None:
            x = tf.concat([x, x2], axis=-1)
        return x


class ResidualBlock(tf.keras.Model):

    def __init__(self, filters, size):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        y = self.conv1(x)
        y = self.batchnorm1(y, training=training)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = self.batchnorm2(y, training=training)
        y = tf.math.add(x, y)
        return y

class ResNet9Generator(tf.keras.Model):

    def __init__(self, noise=False):
        super(ResNet9Generator, self).__init__()
        self.noise_inputs = noise
        initializer = tf.random_normal_initializer(0., 0.02)

        self.init = tf.keras.layers.Conv2D(64,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()

        self.down1 = Downsample(128, 4)
        self.down2 = Downsample(256, 4)

        self.res1 = ResidualBlock(256, 3)
        self.res2 = ResidualBlock(256, 3)
        self.res3 = ResidualBlock(256, 3)
        self.res4 = ResidualBlock(256, 3)
        self.res5 = ResidualBlock(256, 3)
        self.res6 = ResidualBlock(256, 3)
        self.res7 = ResidualBlock(256, 3)
        self.res8 = ResidualBlock(256, 3)
        self.res9 = ResidualBlock(256, 3)

        self.up1 = Upsample(128, 4)
        self.up2 = Upsample(64, 4)

        self.last = tf.keras.layers.Conv2D(OUTPUT_CHANNELS,
                                           (7, 7),
                                           padding='same',
                                           kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        if self.noise_inputs:
            z = tf.random_normal(shape=[BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 1])
            x = tf.concat([x, z], axis=-1)  # (bs, 256, 256, 4)

        x1 = self.init(x)  # (bs, 256, 256, 64)
        x1 = self.batchnorm(x1, training=training)
        x1 = tf.nn.relu(x1)

        x2 = self.down1(x1, training=training)  # (bs, 128, 128, 128)
        x3 = self.down2(x2, training=training)  # (bs, 64, 64, 256)

        x4 = self.res1(x3, training=training)
        x4 = self.res2(x4, training=training)
        x4 = self.res3(x4, training=training)
        x4 = self.res4(x4, training=training)
        x4 = self.res5(x4, training=training)
        x4 = self.res6(x4, training=training)
        x4 = self.res7(x4, training=training)
        x4 = self.res8(x4, training=training)
        x4 = self.res9(x4, training=training)

        x5 = self.up1(x4, None, training=training)  # (bs, 128, 128, 128)
        x6 = self.up2(x5, None, training=training)  # (bs, 256, 256, 64)

        x7 = self.last(x6)  # (bs, 256, 256, 3)
        x7 = tf.nn.tanh(x7)

        return x7


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = Downsample(64, 4, False)
        self.down2 = Downsample(128, 4)
        self.down3 = Downsample(256, 4)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    @tf.contrib.eager.defun
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 30, 30, 1)

        return x


class C_GAN():

    def __init__(self, generator_learning_rate=2e-4, discriminator_learning_rate=2e-4):
        self.generator = ResNet9Generator()
        self.name = self.generator.__class__.__name__ + '_'
        self.discriminator = Discriminator()

        self.generator_optimizer = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        self.discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate, beta1=0.5)

        checkpoint_prefix = os.path.join(checkpoint_dir, self.name + "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)
        self.restore_from_checkpoint(checkpoint)

    def restore_from_checkpoint(self, checkpoint):
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint.restore(checkpoint_file)
        print("restored!!", checkpoint_file)

    def predict(self, input_image, validate=False):
        input_image = tf.cast(input_image, tf.float32)
        if validate is True:
            input_image = load_image(input_image)
        else:
            # Input 3 x H x W (values b/w 0 and 255)
            # Rescale and normalize input
            input_image = tf.image.resize_images(input_image, size=[IMG_HEIGHT, IMG_WIDTH],
                                                 align_corners=True,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            input_image = (input_image / 127.5) - 1
        input_image = tf.expand_dims(input_image, 0)

        # Generate output image (values b/w -1 and 1)
        gen_output = self.generator(input_image, training=True)

        gen_output = ((gen_output.numpy() + 1) * 127.5)[0]
        input_image = ((input_image.numpy() + 1) * 127.5)[0]

        return input_image, gen_output