import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from glob import glob
import h5py
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, ZeroPadding2D, UpSampling2D
from keras.layers.core import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from keras.utils import to_categorical

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

np.random.seed(10)

random_dim = 100


def load_data():

    arr_image = []

    ts_dos = glob('data/*')
    ts_dos.sort()

    label = 0

    for img_list in tqdm(ts_dos):

        img = Image.open(img_list).convert('RGB')
        #If the data type is compatible with symmetry :
        #img_flip = img.transpose(Image.FLIP_LEFT_RIGHT) 
        img = img.resize((100,100))
        image_arr = np.asarray(img)
        image_arr = image_arr.astype('float32')
        image_arr = image_arr / 255
        image_arr = image_arr.reshape(100, 100, 3)
        arr_image.append(image_arr)

    arr_image = np.array(arr_image)  

    return arr_image

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)
def get_generator(optimizer):
    generator = Sequential()

    generator.add(Dense(64*25*25, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((25, 25, 64)))
    generator.add(UpSampling2D(size=(2, 2)))

    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(Conv2D(32, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(UpSampling2D(size=(2, 2)))

    generator.add(Conv2D(3, kernel_size=(5, 5), padding='same', activation='tanh'))

    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()

    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(100, 100, 3), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())

    discriminator.add(Dense(1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
 
def get_gan_network(discriminator, random_dim, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

def plot_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('generated_image_epoch_%d.png' % epoch)

def train(epochs=1, batch_size=128):
    x_train = load_data()
    batch_count = x_train.shape[0] / batch_size

    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(x_train.shape[0] // batch_size)):
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 100 == 0:
            plot_generated_images(e, generator)

        if e % 1000 == 0:
            generator.save("generator_cats_epoch_%d.h5" % e)
            discriminator.save("discriminator_cats_epoch_%d.h5" % e)

if __name__ == '__main__':
    train(10000, 32)