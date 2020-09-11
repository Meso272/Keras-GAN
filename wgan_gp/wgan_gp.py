from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model,load_model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

#import matplotlib.pyplot as plt

import sys
from datasets import *
import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)#original 0.00005
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        # Build and compile the critic
        self.generator.trainable = False
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        

        # The generator takes noise as input and generated imgs
        orig_img =  Input(shape=self.img_shape)
        decomp_img =  Input(shape=self.img_shape)
        recov_img = self.generator(decomp_img)

        # For the combined model we will only train the generator
        

        # The critic takes generated images as input and determines validity
        fake=self.critic(recov_img)
        valid = self.critic(orig_img)
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        validity_interpolated = self.critic(interpolated_img)
        # The combined model  (stacked generator and critic)
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[orig_img, decomp_img],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])

        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        decomp_img_gen =  Input(shape=self.img_shape)
        # Generate images based of noise
        recov_img_gen = self.generator(decomp_img_gen)
        # Discriminator determines validity
        valid = self.critic(recov_img_gen)
        # Defines generator model
        self.generator_model = Model(decomp_img_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)



    def build_generator(self):

        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #model.add(Flatten())


        #model.add(Dense(128 * 2 * 2, activation="relu", input_dim=self.latent_dim))
        #model.add(Reshape((2, 2, 128)))

        model.add(UpSampling2D())
        model.add(Conv2D(256, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        decomp_img = Input(shape=self.img_shape)
        recov_img = model(decomp_img)

        return Model(decomp_img, recov_img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        #model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        orig_train=load_CLDHGH_orig(path="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH/",endnum=50,size=64)
        decomp_train=load_CLDHGH_decomp(path="/home/jliu447/lossycompression/multisnapshot-data-cleaned/CLDHGH_SZ/",endnum=50,size=64)
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, orig_train.shape[0], batch_size)
                orig_imgs = orig_train[idx]
                orig_imgs = np.expand_dims(orig_imgs, axis=3)
                decomp_imgs=decomp_train[idx]
                decomp_imgs = np.expand_dims(decomp_imgs, axis=3)
                # Sample noise as generator input
                #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                recov_imgs = self.generator.predict(decomp_imgs)

                # Train the critic
                d_loss = self.critic_model.train_on_batch([orig_imgs, decomp_imgs],
                                                                [valid, fake, dummy])

                


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(decomp_imgs, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.generator.save("generator.h5")

    


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=40000, batch_size=32, save_interval=1000)
