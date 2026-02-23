#!/usr/bin/env python
# coding: utf-8

# # Build Network from scratch

# In[43]:


# Feedforward Neural Network
import tensorflow as tf
from random import seed, random
from math import exp
import numpy as np
import pandas as pd
import threading
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.signal import convolve2d
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def create_dataset(xs, ys, n_classes=10):
    xs = tf.cast(xs, tf.float32) / 255.0
    ys = tf.cast(ys, tf.float32)
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(ys)).batch(128)

def val_nn(training_inputs_data, training_outputs_data, test_inputs_data):
    tf.compat.v1.disable_eager_execution()
    training_inputs = tf.compat.v1.placeholder(shape=[None, 3], dtype=tf.float32)
    training_outputs = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32) #Desired outputs for each input
    weights = tf.Variable(initial_value=[[.3], [.1], [.8]], dtype=tf.float32)
    bias = tf.Variable(initial_value=[[1]], dtype=tf.float32)

    af_input = tf.matmul(training_inputs, weights) + bias

    # Activation function of the output layer neuron  
    predictions = tf.nn.sigmoid(af_input)
    # Measuring the prediction error of the network after being trained  
    prediction_error = tf.reduce_mean(tf.square(training_outputs - predictions))
    # Minimizing the prediction error using gradient descent optimizer  

    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)
    # Creating a TensorFlow Session  
    sess = tf.compat.v1.Session()
    # Initializing the TensorFlow Variables (weights and bias)  
    sess.run(tf.compat.v1.global_variables_initializer())

    # Training loop of the neural network  
    for step in range(10000):  
        sess.run(fetches=train_op, feed_dict={training_inputs: training_inputs_data, training_outputs: training_outputs_data})  
        # Class scores of some testing data  

    result = sess.run(fetches=predictions, feed_dict={training_inputs: test_inputs_data})
    # Closing the TensorFlow Session to free resources  
    sess.close()
    return result


# In[44]:
X_train = [[255, 0, 0], [248, 80, 68], [0, 0, 255],[67, 15, 210]]  
Y_train = [[1], [1], [0], [0]] 
X_test = [[5, 2, 13], [7, 9, 0]]
tf.compat.v1.disable_eager_execution()
val_nn(X_train, Y_train, X_test)


# In[45]:


# Backprop Neural Network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = 1.0 / (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            transfer_derivative = neuron['output'] * (1.0 - neuron['output'])
            neuron['delta'] = errors[j] * transfer_derivative


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Test training backprop algorithm
seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
    print(layer)
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
           [{'weights': [0.2550690257394217, 0.49543508709194095]},
            {'weights': [0.4494910647887381, 0.651592972722763]}]]

for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

row = [1, 0, None]
output = forward_propagate(network, row)


# # Build CNN network using Pure Python

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32')# / 255.0
X_test = X_test.astype('float32')# / 255.0

# Flatten labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train.flatten())
y_test = label_encoder.transform(y_test.flatten())

class SimpleCNNCifar:
    def __init__(self):
        self.weights = {
            'conv1': np.random.randn(3, 3, 3, 32) / (3 * 3 * 3),
            'batch_norm1_gamma': np.ones((1, 1, 1, 32)),
            'batch_norm1_beta': np.zeros((1, 1, 1, 32)),
            'conv2': np.random.randn(3, 3, 32, 32) / (3 * 3 * 32),
            'batch_norm2_gamma': np.ones((1, 1, 1, 32)),
            'batch_norm2_beta': np.zeros((1, 1, 1, 32)),
            'conv3': np.random.randn(3, 3, 32, 64) / (3 * 3 * 32),
            'batch_norm3_gamma': np.ones((1, 1, 1, 64)),
            'batch_norm3_beta': np.zeros((1, 1, 1, 64)),
            'conv4': np.random.randn(3, 3, 64, 64) / (3 * 3 * 64),
            'batch_norm4_gamma': np.ones((1, 1, 1, 64)),
            'batch_norm4_beta': np.zeros((1, 1, 1, 64)),
            'conv5': np.random.randn(3, 3, 64, 128) / (3 * 3 * 64),
            'batch_norm5_gamma': np.ones((1, 1, 1, 128)),
            'batch_norm5_beta': np.zeros((1, 1, 1, 128)),
            'conv6': np.random.randn(3, 3, 128, 128) / (3 * 3 * 128),
            'batch_norm6_gamma': np.ones((1, 1, 1, 128)),
            'batch_norm6_beta': np.zeros((1, 1, 1, 128)),
            'dense1': np.random.randn(2048, 10) / 2048,
            'dense1_bias': np.zeros((1, 10)),
            'fc': np.random.randn(2048, 10) / 2048,
            'fc_bias': np.zeros((1, 10)),
        }
        self.epsilon = 1e-5

    def forward_pass(self, images):
        conv1 = self.convolution(images, self.weights['conv1'])
        print(conv1.shape)
        activation1 = self.relu_activation(conv1)
        print(activation1.shape)
        batch_norm1 = self.batch_normalization(activation1, self.weights['batch_norm1_gamma'], self.weights['batch_norm1_beta'])
        conv2 = self.convolution(batch_norm1, self.weights['conv2'])
        activation2 = self.relu_activation(conv2)
        batch_norm2 = self.batch_normalization(activation2, self.weights['batch_norm2_gamma'], self.weights['batch_norm2_beta'])
        pool1 = self.max_pooling(batch_norm2)
        conv3 = self.convolution(pool1, self.weights['conv3'])
        batch_norm3 = self.batch_normalization(conv3, self.weights['batch_norm3_gamma'], self.weights['batch_norm3_beta'])
        activation3 = self.relu_activation(batch_norm3)
        conv4 = self.convolution(activation3, self.weights['conv4'])
        batch_norm4 = self.batch_normalization(conv4, self.weights['batch_norm4_gamma'], self.weights['batch_norm4_beta'])
        activation4 = self.relu_activation(batch_norm4)
        pool2 = self.max_pooling(activation4)
        conv5 = self.convolution(pool2, self.weights['conv5'])
        batch_norm5 = self.batch_normalization(conv5, self.weights['batch_norm5_gamma'], self.weights['batch_norm5_beta'])
        activation5 = self.relu_activation(batch_norm5)
        conv6 = self.convolution(activation5, self.weights['conv6'])
        batch_norm6 = self.batch_normalization(conv6, self.weights['batch_norm6_gamma'], self.weights['batch_norm6_beta'])
        activation6 = self.relu_activation(batch_norm6)
        pool3 = self.max_pooling(activation6)
        flatten = pool3.reshape((pool3.shape[0], -1))
        dense1 = np.dot(flatten, self.weights['dense1']) + self.weights['dense1_bias']
        return dense1

    def batch_normalization(self, x, gamma, beta):
        mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        variance = np.var(x, axis=(0, 1, 2), keepdims=True)
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)
        return gamma * x_normalized + beta


    def max_pooling(self, image, pool_size=(2, 2)):
        return np.max(image.reshape((image.shape[0], image.shape[1] // pool_size[0], pool_size[0], image.shape[2] // pool_size[1], pool_size[1], image.shape[3])), axis=(2, 4))

    def relu_activation(self, x):
        return np.maximum(0, x)


    def convolution(self, image, kernel):
        result = [
            np.sum(
                [convolve2d(image[i, :, :, c], kernel[:, :, c, j], mode='same', boundary='symm')[:, :, np.newaxis]
                 for c in range(image.shape[-1])]
            , axis=0)
            for i in range(len(image))
            for j in range(kernel.shape[-1])
        ]
        result = np.array(result)
        result = result.reshape((len(image), image.shape[1], image.shape[2], -1))
        return result

    def softmax_activation(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y, epochs=5, learning_rate=0.01, batch_size=64):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                end = min(i + batch_size, len(X))
                batch_images = X[i:end]
                batch_labels = y[i:end].astype(int)  # Convert labels to integers

                # Forward pass
                logits = self.forward_pass(batch_images)
                probs = self.softmax_activation(logits)

                # Compute loss
                loss = self.compute_loss(logits, batch_labels)

                # Backward pass
                gradients = self.backward_pass(batch_images, probs, batch_labels)

                # Update weights
                self.update_weights(gradients, learning_rate)

    def compute_loss(self, predictions, labels):
        probs = self.softmax_activation(predictions)
        return np.mean(-np.log(probs[np.arange(len(probs)), labels] + 1e-8))

    def _max_pool_backward(self, d_out, pre_pool, pool_size=(2, 2)):
        batch, h, w, c = pre_pool.shape
        ph, pw = pool_size
        d_input = np.zeros_like(pre_pool)

        for i in range(0, h, ph):
            for j in range(0, w, pw):
                # Find which position in each pool window was the max
                window = pre_pool[:, i:i + ph, j:j + pw, :]  # shape: (batch, ph, pw, c)
                max_vals = np.max(window, axis=(1, 2), keepdims=True)  # shape: (batch, 1, 1, c)
                mask = (window == max_vals)  # True where the max was
                # Pass gradient only to the max position
                d_input[:, i:i + ph, j:j + pw, :] += mask * d_out[:, i // ph:i // ph + 1, j // pw:j // pw + 1, :]

        return d_input

    def _batchnorm_backward(self, d_out, x, gamma):
        mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        var = np.var(x, axis=(0, 1, 2), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)

        # beta gradient: just sum the incoming gradient
        d_beta = np.sum(d_out, axis=(0, 1, 2), keepdims=True)
        # gamma gradient: sum of incoming gradient times normalized input
        d_gamma = np.sum(d_out * x_norm, axis=(0, 1, 2), keepdims=True)
        # gradient w.r.t. the normalized x, then scale by gamma
        d_x = (gamma / np.sqrt(var + self.epsilon)) * d_out

        return d_x, d_gamma, d_beta

    def _conv_backward(self, d_out, x, kernel):
        d_kernel = np.zeros_like(kernel)
        d_x = np.zeros_like(x)
        kh, kw, in_c, out_c = kernel.shape

        for i in range(in_c):
            for j in range(out_c):
                for n in range(len(x)):
                    # Gradient w.r.t. kernel: correlate input with output gradient
                    d_kernel[:, :, i, j] += convolve2d(
                        x[n, :, :, i], d_out[n, :, :, j], mode='valid'
                    )
                    # Gradient w.r.t. input: convolve output gradient with flipped kernel
                    d_x[n, :, :, i] += convolve2d(
                        d_out[n, :, :, j], kernel[:, :, i, j], mode='same', boundary='symm'
                    )

        return d_kernel, d_x

    def update_weights(self, gradients, learning_rate):
        for key in gradients:
            self.weights[key] -= learning_rate * gradients[key]

    def backward_pass(self, images, predictions, labels):
        gradients = {}
        num_samples = len(images)

        # ---- Rerun forward pass to get all intermediate values we need ----
        conv1 = self.convolution(images, self.weights['conv1'])
        act1 = self.relu_activation(conv1)
        bn1 = self.batch_normalization(act1, self.weights['batch_norm1_gamma'], self.weights['batch_norm1_beta'])
        conv2 = self.convolution(bn1, self.weights['conv2'])
        act2 = self.relu_activation(conv2)
        bn2 = self.batch_normalization(act2, self.weights['batch_norm2_gamma'], self.weights['batch_norm2_beta'])
        pool1 = self.max_pooling(bn2)

        conv3 = self.convolution(pool1, self.weights['conv3'])
        bn3 = self.batch_normalization(conv3, self.weights['batch_norm3_gamma'], self.weights['batch_norm3_beta'])
        act3 = self.relu_activation(bn3)
        conv4 = self.convolution(act3, self.weights['conv4'])
        bn4 = self.batch_normalization(conv4, self.weights['batch_norm4_gamma'], self.weights['batch_norm4_beta'])
        act4 = self.relu_activation(bn4)
        pool2 = self.max_pooling(act4)

        conv5 = self.convolution(pool2, self.weights['conv5'])
        bn5 = self.batch_normalization(conv5, self.weights['batch_norm5_gamma'], self.weights['batch_norm5_beta'])
        act5 = self.relu_activation(bn5)
        conv6 = self.convolution(act5, self.weights['conv6'])
        bn6 = self.batch_normalization(conv6, self.weights['batch_norm6_gamma'], self.weights['batch_norm6_beta'])
        act6 = self.relu_activation(bn6)
        pool3 = self.max_pooling(act6)

        flatten = pool3.reshape((num_samples, -1))  # shape: (batch, 2048)
        logits = np.dot(flatten, self.weights['dense1']) + self.weights['dense1_bias']  # shape: (batch, 10)

        # ---- Step 1: Gradient of loss w.r.t. softmax+crossentropy output ----
        # When you combine softmax + cross-entropy loss, the gradient simplifies to:
        # (predicted_probability - 1) for the correct class, predicted_probability for others
        probs = self.softmax_activation(logits)  # shape: (batch, 10)
        d_logits = probs.copy()  # shape: (batch, 10)
        d_logits[np.arange(num_samples), labels] -= 1  # subtract 1 from correct class
        d_logits /= num_samples  # average over batch

        # ---- Step 2: Gradient of dense1 weights ----
        # dense1 is just a matrix multiply: logits = flatten @ dense1 + bias
        # So gradient w.r.t. dense1 = flatten.T @ d_logits
        gradients['dense1'] = np.dot(flatten.T, d_logits)  # shape: (2048, 10)
        gradients['dense1_bias'] = np.sum(d_logits, axis=0, keepdims=True)  # shape: (1, 10)

        # Gradient flowing back into flatten (shape: batch, 2048)
        d_flatten = np.dot(d_logits, self.weights['dense1'].T)

        # ---- Step 3: Unflatten back to pool3 shape ----
        # pool3 has shape (batch, 4, 4, 128) — 4*4*128 = 2048
        d_pool3 = d_flatten.reshape(pool3.shape)

        # ---- Step 4: Gradient through max pooling (pool3 -> act6) ----
        # Max pooling only passes gradient to whichever position was the maximum.
        # We upsample by repeating, then zero out positions that weren't the max.
        d_act6 = self._max_pool_backward(d_pool3, act6)

        # ---- Step 5: Gradient through relu (act6 -> bn6) ----
        # ReLU gradient: pass through where input > 0, zero elsewhere
        d_bn6 = d_act6 * (conv6 > 0)

        # ---- Step 6: Gradient through batch norm (bn6 -> conv6) ----
        d_conv6, gradients['batch_norm6_gamma'], gradients['batch_norm6_beta'] = \
            self._batchnorm_backward(d_bn6, conv6, self.weights['batch_norm6_gamma'])

        # ---- Step 7: Gradient through conv6 weights and into act5 ----
        gradients['conv6'], d_act5 = self._conv_backward(d_conv6, act5, self.weights['conv6'])

        # ---- Step 8: Gradient through relu (act5 -> bn5) ----
        d_bn5 = d_act5 * (conv5 > 0)

        # ---- Step 9: Gradient through batch norm (bn5 -> conv5) ----
        d_conv5, gradients['batch_norm5_gamma'], gradients['batch_norm5_beta'] = \
            self._batchnorm_backward(d_bn5, conv5, self.weights['batch_norm5_gamma'])

        # ---- Step 10: Gradient through conv5 and into pool2 ----
        gradients['conv5'], d_pool2 = self._conv_backward(d_conv5, pool2, self.weights['conv5'])

        # ---- Repeat the same pattern back through pool2, conv4, conv3, pool1, conv2, conv1 ----
        d_act4 = self._max_pool_backward(d_pool2, act4)
        d_bn4 = d_act4 * (conv4 > 0)
        d_conv4, gradients['batch_norm4_gamma'], gradients['batch_norm4_beta'] = \
            self._batchnorm_backward(d_bn4, conv4, self.weights['batch_norm4_gamma'])
        gradients['conv4'], d_act3 = self._conv_backward(d_conv4, act3, self.weights['conv4'])

        d_bn3 = d_act3 * (conv3 > 0)
        d_conv3, gradients['batch_norm3_gamma'], gradients['batch_norm3_beta'] = \
            self._batchnorm_backward(d_bn3, conv3, self.weights['batch_norm3_gamma'])
        gradients['conv3'], d_pool1 = self._conv_backward(d_conv3, pool1, self.weights['conv3'])

        d_act2 = self._max_pool_backward(d_pool1, act2)  # Note: pool1 came from bn2/act2
        d_bn2 = d_act2 * (conv2 > 0)
        d_conv2, gradients['batch_norm2_gamma'], gradients['batch_norm2_beta'] = \
            self._batchnorm_backward(d_bn2, conv2, self.weights['batch_norm2_gamma'])
        gradients['conv2'], d_bn1 = self._conv_backward(d_conv2, bn1, self.weights['conv2'])

        d_act1 = d_bn1 * (conv1 > 0)
        d_conv1, gradients['batch_norm1_gamma'], gradients['batch_norm1_beta'] = \
            self._batchnorm_backward(d_act1, conv1, self.weights['batch_norm1_gamma'])
        gradients['conv1'], _ = self._conv_backward(d_conv1, images, self.weights['conv1'])

        return gradients


# Convert tensors to Pandas DataFrame using threading
def convert_to_dataframe(X, y, start, end, result, cnn_model):
    data = []

    for i in range(start, end):
        image = X[i].reshape(-1, 32, 32, 3)
        label = y[i]

        # Forward pass through the CNN to get features
        features = cnn_model.forward_pass(np.expand_dims(image, axis=0))

        data.append(np.concatenate([features.flatten(), [int(label)]]))

    result.extend(data)

# Function to use threading effectively
def tensors_to_dataframe(X, y, cnn_model, num_threads=4, batch_size=64):
    data = []
    threads = []

    for i in range(0, len(X), batch_size * num_threads):
        for j in range(num_threads):
            start = i + j * batch_size
            end = min(i + (j + 1) * batch_size, len(X))
            thread_result = []
            thread = threading.Thread(target=convert_to_dataframe, args=(X[start:end], y[start:end], start, end, thread_result, cnn_model))
            thread.daemon = True  # Set daemon attribute to True
            thread.start()
            threads.append((thread, thread_result))

    for thread, thread_result in threads:
        try:
            thread.join()
        except Exception as e:
            pass  # Do nothing on exception

        data.extend(thread_result)

    # Create Pandas DataFrame
    columns = [f'feature_{i}' for i in range(data[0].shape[0] - 1)] + ['label']
    df = pd.DataFrame(data, columns=columns)

    return df

# Train CNN model
cnn_model = SimpleCNNCifar()
cnn_model.fit(X_train, y_train, epochs=5)

# Convert tensors to Pandas DataFrame using threading
df_train_cnn = tensors_to_dataframe(X_train, y_train, cnn_model, batch_size=64)

# Split the data into train and test sets
df_train_cnn, df_test_cnn = train_test_split(df_train_cnn, test_size=0.2, random_state=42)

# Separate features and labels
X_train_cnn, y_train_cnn = df_train_cnn.iloc[:, :-1], df_train_cnn['label']
X_test_cnn, y_test_cnn = df_test_cnn.iloc[:, :-1], df_test_cnn['label']

# Continue with your desired classification model or further processing using the CNN features.
# For example, if you want to use the SimpleCNNCifar model for classification:

# Initialize and train SimpleCNNCifar model
cnn_model_cifar = SimpleCNNCifar()
cnn_model_cifar.fit(X_train_cnn, y_train_cnn, epochs=5, learning_rate=0.01)

# Ensure the number of samples in X_test_cnn matches y_test_cnn
X_test_cnn = df_test_cnn.iloc[:, :-1].values.reshape(-1, 32, 32, 3)
y_test_cnn = df_test_cnn['label']

# Make predictions on the test set
y_pred_cnn_cifar = np.argmax(cnn_model_cifar.forward_pass(X_test_cnn), axis=1)

# Calculate accuracy
accuracy_cnn_cifar = accuracy_score(y_test_cnn, y_pred_cnn_cifar)
print(f'SimpleCNNCifar Accuracy: {accuracy_cnn_cifar:.4f}')

'''
This trains a CNN neural network in Tensorflow without using Keras Layers
'''

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the CNN model using subclassed layers
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Convolutional layer 1
        self.conv1 = tf.Variable(tf.random.normal([3, 3, 3, 32]))
        # Convolutional layer 2
        self.conv2 = tf.Variable(tf.random.normal([3, 3, 32, 64]))
        # Fully connected layer
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.Variable(tf.random.normal([8 * 8 * 64, 256]))
        # Output layer
        self.output_layer = tf.Variable(tf.random.normal([256, 10]))

    def call(self, x, training=True):
        # Convolutional layer 1
        conv1 = tf.nn.relu(tf.nn.conv2d(x, self.conv1, strides=[1, 1, 1, 1], padding='SAME'))
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Convolutional layer 2
        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, self.conv2, strides=[1, 1, 1, 1], padding='SAME'))
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten
        flattened = self.flatten(pool2)

        # Fully connected layer
        fc = tf.nn.relu(tf.matmul(flattened, self.fc))

        # Output layer
        output = tf.matmul(fc, self.output_layer)

        return output

# Instantiate the model
model = MyModel()

# Define training parameters
learning_rate = 0.001
epochs = 5
batch_size = 64

# Use tf.data.Dataset for input pipeline parallelization
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)

# Define loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Training the model
for epoch in range(epochs):
    for x_batch, y_batch in train_dataset:
        with tf.device('/device:GPU:0'):  # Specify the GPU device
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                current_loss = loss_fn(y_batch, logits)

            gradients = tape.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate training loss and accuracy
    train_loss = loss_fn(y_train, model(x_train, training=False))
    train_accuracy = np.mean(np.argmax(model(x_train, training=False), axis=1) == np.argmax(y_train, axis=1))

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

# Calculate test accuracy
test_accuracy = np.mean(np.argmax(model(x_test, training=False), axis=1) == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {test_accuracy:.4f}')


# # Tensor Basics
X_train = [[255, 0, 0], [248, 80, 68], [0, 0, 255],[67, 15, 210]]  
Y_train = [[1], [1], [0], [0]] 
X_test = [[5, 2, 13], [7, 9, 0]]
xt=tf.constant(X_train)
xt.shape, tf.rank(xt)
y1 = tf.random.uniform(shape=[5, 300], minval=0, maxval=1, dtype=tf.float64)
y2 = tf.random.uniform(shape=[5, 300], minval=0, maxval=1, dtype=tf.float64)
prob4=tf.matmul(y1, tf.transpose(y2))
prob5=tf.tensordot(y1, tf.transpose(y2), axes=1)
y6 = tf.random.uniform(shape=[224, 224, 3], minval=0, maxval=1, dtype=tf.float64)
prob6=tf.math.reduce_max(y6, axis=0)
y7 = tf.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1, dtype=tf.float64)
prob7 = tf.squeeze(y7, axis=0)
y8 = tf.random.uniform(shape=[10], minval=0, maxval=10, dtype=tf.int64)
prob9=tf.math.argmax(y8)
prob10=tf.one_hot(y8, depth=10)


# In[3]:


tf.constant(X_train)
a = tf.linspace(-1, 1, 10)
a_new = tf.expand_dims(a, axis=1)
a_new
a_transpose=tf.transpose(a_new)
a_randint = tf.random.uniform(shape=[3,4], minval=1, maxval=10, dtype=tf.int32)
a_randn = tf.random.normal(shape=[3,4])
a_zeros = tf.zeros(shape=[3,4])
a_ones = tf.ones(shape=[3,4])
a_fives = tf.fill([3,4], 5.2)
a_eye = tf.eye(5)
x = tf.expand_dims(tf.Variable([1, 2, 0, 4]), axis=1)
mask = x >= 2
a_slice = tf.boolean_mask(x, mask)
masked = tf.greater(x,1)
zeros=tf.zeros_like(x)
a_masked = tf.where(masked, x, zeros)
a_sq_eye = tf.reshape(a_eye, [25])
a_new_transpose = tf.squeeze(a_transpose)
a_concat = tf.concat((tf.cast(a_new, tf.float32), tf.cast(tf.expand_dims(a_sq_eye, axis=1), tf.float32)), axis=0)
a_matmul = tf.matmul(tf.transpose(a_masked), x)

numpy_arr = np.array([10.0, 11.0, 12.0, 13.0])
from_numpy_to_tensor = tf.convert_to_tensor(numpy_arr)
a_min = tf.reduce_min(x)
a_max = tf.reduce_max(x, axis=1)
a_randint, a_randn, a_zeros, a_fives, a_ones, a_eye, a_randint.shape, a_randn.dtype, a_slice, a_masked, a_sq_eye, a_transpose, a_new_transpose, a_concat, a_matmul, from_numpy_to_tensor, a_min, a_max



