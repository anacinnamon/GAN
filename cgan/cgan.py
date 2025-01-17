from __future__ import print_function, division

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
from keras.models import load_model


class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 956
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(100,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])

        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, y_train), (_, _) = mnist.load_data()
        (X_train, y_train), (_, _) = self.load_my_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs, labels = X_train[idx], y_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            valid = np.ones((half_batch, 1))
            fake = np.zeros((half_batch, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            valid = np.ones((batch_size, 1))
            # Generator wants discriminator to label the generated images as the intended
            # digits
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save("saved_model/generator_" + str(epoch) + ".h5")
                self.discriminator.save("saved_model/discriminator_" + str(epoch) + ".h5")

    def sample_images(self, epoch):

        list_labels = self.load_label()

        r, c = 8, 6
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 48).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # cv2.imwrite("images/" + str(epoch) + "_" + str(list_labels[int(sampled_labels[cnt])]) + ".png",
                #             gen_imgs[cnt, :, :, 0] * 256)
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("char: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def load_my_data(self):
        list_objects = pickle.load(open("../cinnamon/ETL8G_GAN.pkl", "rb"))

        np.random.shuffle(list_objects)

        list_images = []
        list_labels = []

        for i, ob in enumerate(list_objects):
            # if ob[1] < 10:
            list_images.append(ob[0])
            list_labels.append(ob[1])

        print("load ok")
        return (np.asarray(list_images, dtype=np.uint8),
                np.asarray(list_labels, dtype=np.uint8)), (
                   np.asarray(list_images[0:1], dtype=np.uint8),
                   np.asarray(list_labels[0:1], dtype=np.uint8))

    def load_label(self):
        list_labels = pickle.load(open("../cinnamon/ETL8G_GAN_labels.pkl", "rb"))
        return list_labels

    # def gen_images(self, num_gen):
    #
    #     list_labels = self.load_label()
    #     self.generator.load_weights('saved_model/generator_1000000.h5')
    #
    #     for num in range(num_gen):
    #
    #         r, c = 8, 6
    #         noise = np.random.normal(0, 1, (r * c, 100))
    #         sampled_labels = np.arange(0, 48).reshape(-1, 1)
    #
    #         gen_imgs = self.generator.predict([noise, sampled_labels])
    #
    #         # Rescale images 0 - 1
    #         gen_imgs = 0.5 * gen_imgs + 0.5
    #
    #         fig, axs = plt.subplots(r, c)
    #         cnt = 0
    #         for i in range(r):
    #             for j in range(c):
    #                 cv2.imwrite(
    #                     "gen_images/" + str(int(sampled_labels[cnt])) + "_" + str(
    #                         list_labels[int(sampled_labels[cnt])]) + "_" + str(num) + ".png",
    #                     256 - gen_imgs[cnt, :, :, 0] * 256)
    #                 axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
    #                 axs[i, j].set_title("char: %d" % sampled_labels[cnt])
    #                 axs[i, j].axis('off')
    #                 cnt += 1
    #         fig.savefig("gen_images/%d.png" % num)
    #         # plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=100001, batch_size=32, sample_interval=5000)
    # cgan.gen_images(10)
