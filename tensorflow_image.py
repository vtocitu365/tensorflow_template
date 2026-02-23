#!/usr/bin/env python
# coding: utf-8

# # Image Classification

import pandas
import sklearn
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from scipy import ndimage
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from functools import partial
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Model, Input
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(tf.__version__)
print(get_available_devices())

accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
IMAGE_SIZE = [32, 32]

# Prep pixels for tfds datasets load dataset
def prep_pixels2(train, test, target_train, target_test):
    img_rows=28
    img_cols=28
    X_train = train.reshape(train.shape[0], img_rows, img_cols, 1)
    X_test = test.reshape(test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    train_norm = X_train.astype('float32')
    test_norm = X_test.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    target_train = to_categorical(target_train)
    target_test =  to_categorical(target_test)
    return train_norm, test_norm, target_train, target_test


# Prep pixels for tfds datasets load dataset
def prep_pixels(image, label, depth=10):
    img_rows=28
    img_cols=28
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255)
    train_norm = tf.image.resize(image, (32, 32))
    target = tf.one_hot(label, depth=depth)
    return train_norm, target



# CNN model
def val_cnn_model(n_channels=1):
    model = Sequential()
    model.add(Input(shape=(32, 32, n_channels)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(320, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    return model


# CNN optimized for MNIST
def val_cnn_mnist(n_channels=1):
    model = Sequential()
    model.add(Input(shape=(28, 28, n_channels)))
    model.add(Conv2D(6, (5, 5), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(84, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[7]:


# CNN Optimized for CIFAR10
def val_cnn_cifar(n_depth, n_channels=3):
    weight_decay = 1e-4
    model = Sequential()
    model.add(Input(shape=(32, 32, n_channels)))
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(n_depth, activation='softmax'))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=1e-5
    )
    opt = RMSprop(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def val_cnn_cifar100(n_depth, n_channels=3):
    weight_decay = 1e-4
    model = Sequential()
    model.add(Input(shape=(32, 32, n_channels)))
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_depth, activation='softmax'))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=1e-5
    )
    opt = RMSprop(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# RNN network for images
def val_rnn_model(x_train):
    '''i = Input(shape=x_train[0].shape)
    x = LSTM(128)(i)
    x = Dense(10, activation='softmax')(x)
    model=Model(i, x)'''

    model=Sequential()
    model.add(Input(shape=x_train[0].shape))
    model.add(LSTM(128))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])
    return model


# In[9]:


from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model

# Pretrained MobileNet network for image recognition
def val_mn_model(depth, n_channels=3):
    bottom_model = MobileNet(weights='imagenet', include_top=False, input_shape=(32,32, n_channels))
    for layer in bottom_model.layers:
        layer.trainable = False
    top_model = Flatten(name='flatten')(bottom_model.output)#top_model = Dense(1024, activation='relu')(bottom_model.output)
    top_model = Dense(depth, activation='relu')(top_model)
    top_model = Dense(depth, activation='softmax')(top_model)
    model = Model(inputs = bottom_model.inputs, outputs=top_model)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[10]:


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

# Pretrained VGG Network for image recognition
def val_vgg_model(depth, n_channels=3):
    bottom_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32, n_channels))
    for layer in bottom_model.layers:
        layer.trainable = False
    top_model = Flatten(name='flatten')(bottom_model.output)
    top_model = Dense(depth, activation='relu')(top_model)
    top_model = Dense(depth, activation='softmax')(top_model)
    model = Model(inputs = bottom_model.inputs, outputs=top_model)
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

# Pretrained ResNet Network for image recognition
def val_resnet_model(depth, n_channels=3):
    bottom_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32, n_channels))
    for layer in bottom_model.layers:
        layer.trainable = False
    top_model = Flatten(name='flatten')(bottom_model.output)
    top_model = Dense(depth, activation='relu')(top_model)
    top_model = Dense(depth, activation='softmax')(top_model)
    model = Model(inputs = bottom_model.inputs, outputs=top_model)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def training_step(inputs, targets, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss_value = loss_fn(targets, logits)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Update the accuracy metric
    accuracy_metric.update_state(targets, logits)

    return loss_value

@tf.function
def train_step(inputs, targets, model, loss_fn, optimizer):
    # Handles a single batch - safe to use with @tf.function
    loss = training_step(inputs, targets, model, loss_fn, optimizer)
    return loss

def run_training_epoch(dataset, model, loss_fn, optimizer):
    # Regular Python loop over batches - lives outside @tf.function
    training_loss = 0.0
    num_batches = 0

    for inputs, targets in dataset:
        loss = train_step(inputs, targets, model, loss_fn, optimizer)
        training_loss += loss.numpy()
        num_batches += 1

    mean_loss = training_loss / num_batches
    current_accuracy = accuracy_metric.result().numpy()
    accuracy_metric.reset_states()

    return mean_loss, current_accuracy

def evaluate_image_model(train_dataset, val_dataset, num_epochs, n_channels=3, depth=10, model_name='cifar', use_custom_loop=False, dataset_size=50000):
    batch_size = 64
    steps = dataset_size // batch_size   # computed from actual dataset size

    if model_name == 'cifar':
        model = val_cnn_cifar(depth, n_channels)
        print(model.summary())
    elif model_name == 'cifar100':
        model = val_cnn_cifar100(depth, n_channels)
        print(model.summary())
    elif model_name == 'vgg':
        model = val_vgg_model(depth, n_channels)
    elif model_name == 'resnet':
        model = val_resnet_model(depth, n_channels)
    else:
        model = val_mn_model(depth, n_channels)

    if use_custom_loop:
        loss_fn = model.loss
        optimizer = model.optimizer
        for epoch in range(num_epochs):
            mean_loss, accuracy = run_training_epoch(train_dataset, model, loss_fn, optimizer)
            print(f"Epoch {epoch+1}: loss={mean_loss:.4f}, accuracy={accuracy:.4f}")
    else:
        model.fit(train_dataset, epochs=num_epochs, steps_per_epoch=steps, validation_data=val_dataset, verbose=2)

    _, acc = model.evaluate(val_dataset, verbose=2)
    return acc

# Train on RNN network
def evaluate_image_model_rnn(x_train, y_train, x_test, y_test):
    # Expand dimensions to include a channel (grayscale)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    model=val_cnn_mnist()
    print(model.summary())
    train_generator=ImageDataGenerator(rotation_range=7, width_shift_range=0.05, shear_range=0, height_shift_range=0.07, zoom_range=0.05)
    test_generator=ImageDataGenerator()
    train_generator=train_generator.flow(x_train, y_train, batch_size=64)
    test_generator = test_generator.flow(x_test, y_test, batch_size=64)
    model.fit(train_generator, validation_data=test_generator, epochs=5, verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=2)
    return acc


#CNN optimized for CIFAR10
batch_size=64
train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
train = train_ds.map(partial(prep_pixels, depth=10)).cache().shuffle(100).batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat()
test = test_ds.map(partial(prep_pixels, depth=10)).cache().batch(64).prefetch(tf.data.experimental.AUTOTUNE)
epochs=10
with tf.device('/device:GPU:0'):
    evaluate_image_model(train, test, epochs, n_channels=3, depth=10, model_name='cifar', use_custom_loop=True)


# Eurosat - VGG16
train_ds, test_ds = tfds.load('eurosat', split=['train', 'test'], as_supervised=True)
train = train_ds.map(partial(prep_pixels, depth=10)).cache().shuffle(100).batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat()
test = test_ds.map(partial(prep_pixels, depth=10)).cache().prefetch(tf.data.experimental.AUTOTUNE).batch(64)
epochs=20
with tf.device('/device:GPU:0'):
    evaluate_image_model(train, test, epochs, n_channels=3, depth=10, model_name='vgg')


#Eurosat - ResNet50
train_ds, test_ds = tfds.load('eurosat', split=['train','test'], as_supervised=True)
train = train_ds.map(partial(prep_pixels, depth=10)).cache().shuffle(100).batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat()
test = test_ds.map(partial(prep_pixels, depth=10)).cache().prefetch(tf.data.experimental.AUTOTUNE).batch(64)
epochs=20
with tf.device('/device:GPU:0'):
    evaluate_image_model(train, test, epochs, n_channels=3, depth=10, model_name='resnet')


#Eurosat - MobileNet
train_ds, test_ds = tfds.load('eurosat', split=['train','test'], as_supervised=True)
train = train_ds.map(partial(prep_pixels, depth=10)).cache().shuffle(100).batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat()
test = test_ds.map(partial(prep_pixels, depth=10)).cache().prefetch(tf.data.experimental.AUTOTUNE).batch(64)
epochs=20
with tf.device('/device:GPU:0'):
    evaluate_image_model(train, test, epochs, n_channels=3, depth=10, model_name='mn')


# CIFAR100 with a CNN netowork optimized for CIFAR10
train_ds, test_ds = tfds.load('cifar100', split=['train','test'], as_supervised=True)
train = train_ds.map(partial(prep_pixels, depth=100)).cache().shuffle(100).batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat()
test = test_ds.map(partial(prep_pixels, depth=100)).cache().batch(64).prefetch(tf.data.experimental.AUTOTUNE)
epochs=20
with tf.device('/device:GPU:0'):
    evaluate_image_model(train, test, epochs, n_channels=3, depth=100, model_name='cifar100')


#MNIST Dataset using tf.kersas.dataset instead of tfds
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test=x_test/255.0
print(get_available_devices())
with tf.device('/device:GPU:0'):
    evaluate_image_model_rnn(x_train, y_train, x_test, y_test)


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),  # id instead of class label
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    id_val = example['id']
    return image, id_val  # returns (image, id) pairs instead of (image, label)

def val_load_dataset(filenames, labeled=True, ordered=False):
    '''
    Read from TFRecords. For optimal performance, reading from multiple files at once and
    disregarding data order. Order does not matter since we will be shuffling the data anyway.
    '''
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=10) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=10)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


# # Variational Autoencoder
def val_vae(x_train, x_test):
    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(1, (2, 2), activation='relu', padding='same')(x)
    x = Flatten()(x)
    encoded = Dense(2, activation='relu')(x)

    encoder = Model(inputs=inputs, outputs=encoded)

    encoded_inputs = Input(shape=(2,))

    x = Dense(4, activation='relu')(encoded_inputs)
    x = Reshape((2, 2, 1))(x)
    x = Conv2D(4, (2, 2), activation='relu', padding='same')(x)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((7, 7))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    decoder = Model(inputs=encoded_inputs, outputs=decoded)

    x = encoder(inputs)
    x = decoder(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy', 'mse'])

    print(model.summary())

    clr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_delta=0.01,
        cooldown=0,
        min_lr=1e-7,
        verbose=1)

    model.fit(
        x_train,
        x_train,
        batch_size=256,
        epochs=10,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[clr])

    return model, encoder, decoder

# Train on RNN network
def evaluate_image_model_vae(x_train, y_train, x_test, y_test):
    with tf.device('/device:GPU:0'):
        model, encoder, decoder=val_vae(x_train, x_test)
    return

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()
x_train, x_test, y_train, y_test = prep_pixels2(x_train, x_test, y_train, y_test)
with tf.device('/device:GPU:0'):
    evaluate_image_model_vae(x_train, y_train, x_test, y_test)

