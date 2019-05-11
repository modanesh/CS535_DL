"""
Mohamad Hosein Danesh
"""

from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        # DEFINE __init function
        self.W = W
        self.b = b

    def forward(self, x):
        # DEFINE forward function
        self.x = x
        ltransform_output = np.dot(self.x, self.W) + self.b
        return ltransform_output

    def backward(self, grad_output):
        # DEFINE backward function
        # Computing the gradients
        dx = np.dot(grad_output, self.W.T)
        dw = np.dot(self.x.T, grad_output)
        db = np.sum(grad_output, axis=0)
        return dx, dw, db
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self):
        self.relu_output = None

    def forward(self, x):
        # DEFINE forward function
        self.relu_output = np.maximum(0, x)
        return self.relu_output

    def backward(self, grad_output):
        # DEFINE backward function
        # Computing the gradients
        self.relu_output[self.relu_output > 0] = 1
        dx = np.multiply(self.relu_output, grad_output)
        return dx

# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.y = None
        self.sigmoid = None
        self.epsilon = 10e-8

    def forward(self, x):
        # DEFINE forward function
        self.sigmoid = 1.0 / (1.0 + np.exp(-x))
        return self.sigmoid

    def backward(self, y, grad_output):
        # DEFINE backward function
        # Computing the gradients
        self.y = y
        delta = grad_output * (self.sigmoid - self.y)
        return delta

# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units, output_units):
        # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.output_units = output_units

        # Weight and bias initializations
        self.w1 = np.random.uniform(-1.0, 1.0, size=(input_dims, hidden_units))
        self.b1 = np.random.uniform(-1.0, 1.0, size=(1, hidden_units))
        self.w2 = np.random.uniform(-1.0, 1.0, size=(hidden_units, output_units))
        self.b2 = np.random.uniform(-1.0, 1.0, size=(1, output_units))

        # Creating instances of LinearTransform, ReLU, and SigmoidCrossEntropy classes
        self.linearT1 = LinearTransform(self.w1, self.b1)
        self.linearT2 = LinearTransform(self.w2, self.b2)
        self.relu = ReLU()
        self.sigmoidCE = SigmoidCrossEntropy()

        # Momentum initialization
        self.w1_momentum = 0
        self.b1_momentum = 0
        self.w2_momentum = 0
        self.b2_momentum = 0

    # Training the model.
    def train(self, x_train, y_train, batch_size, learning_rate=0.005, momentum=0.6, l2_penalty=0.0):
        # Get the data batches.
        x_batch = x_train[batch * batch_size: (batch + 1) * batch_size, :]
        y_batch = y_train[batch * batch_size: (batch + 1) * batch_size, :]

        # Forward pass. Computing the predictions.
        z1_in = self.linearT1.forward(x_batch)
        z1_out = self.relu.forward(z1_in)
        z2_in = self.linearT2.forward(z1_out)
        prediction = self.sigmoidCE.forward(z2_in)

        # Backward pass. Computing the derivatives.
        delta = self.sigmoidCE.backward(y_batch, 1)
        dx2, dw2, db2 = self.linearT2.backward(delta)
        dx2_r = self.relu.backward(dx2)
        dx1, dw1, db1 = self.linearT1.backward(dx2_r)

        # Updating weights and biases.
        self.w1_momentum = momentum * self.w1_momentum - learning_rate * dw1
        self.b1_momentum = momentum * self.b1_momentum - learning_rate * db1
        self.w2_momentum = momentum * self.w2_momentum - learning_rate * dw2
        self.b2_momentum = momentum * self.b2_momentum - learning_rate * db2

        self.w1 += self.w1_momentum
        self.b1 += self.b1_momentum
        self.w2 += self.w2_momentum
        self.b2 += self.b2_momentum

        # Computing the loss
        loss = -1 * (y_batch * np.log(prediction + 10e-8) + (1.0 - y_batch) * np.log(1.0 - prediction + 10e-8))
        loss += l2_penalty / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        mean_loss = np.mean(loss)

        acc_b = 100 * np.mean((np.round(prediction) == y_batch))

        batch_loss.append(mean_loss)
        batch_accuracy.append(acc_b)

        return acc_b, mean_loss

    def evaluate(self, x, y, batch_size, l2_penalty):
        # INSERT CODE for testing the network
        num_exp = x.shape[0]
        eval_x, eval_y = random_dist_data(x, y, num_exp)
        num_batches = num_exp // batch_size

        eval_loss, eval_accuracy = [], []
        # Make data into different batches.
        for batch in range(num_batches):
            x_batch = eval_x[batch * batch_size: (batch + 1) * batch_size, :]
            y_batch = eval_y[batch * batch_size: (batch + 1) * batch_size, :]

            # Forward pass. Computing the predictions.
            z1_in = self.linearT1.forward(x_batch)
            z1_out = self.relu.forward(z1_in)
            z2_in = self.linearT2.forward(z1_out)
            z2_out = self.sigmoidCE.forward(z2_in)
            y_pred = np.round(z2_out)

            # Computing the loss
            loss = -1 * (y_batch * np.log(z2_out + 10e-8) + (1.0 - y_batch) * np.log(1.0 - z2_out + 10e-8))
            loss += l2_penalty / 2 * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))

            eval_loss.append(np.mean(loss))
            eval_accuracy.append(100 * np.mean(y_pred == y_batch))

        return np.mean(eval_loss), np.mean(eval_accuracy)

# ADD other operations and data entries in MLP if needed

def random_dist_data(x, y, num_exp):
    rnd_ind = np.arange(num_exp)
    np.random.shuffle(rnd_ind)
    eval_x = x[rnd_ind]
    eval_y = y[rnd_ind]

    return eval_x, eval_y

if __name__ == '__main__':
    data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    x_train = data['train_data']
    y_train = data['train_labels']
    x_test = data['test_data']
    y_test = data['test_labels']

    num_examples, input_dims = x_train.shape

    # Data normalization
    x_max_train = np.max(x_train, axis=0)
    x_train = x_train / x_max_train
    x_max_test = np.max(x_test, axis=0)
    x_test = x_test / x_max_test

    x_mean_train = np.mean(x_train, axis=0)
    x_train = (x_train - x_mean_train)
    x_mean_test = np.mean(x_test, axis=0)
    x_test = (x_test - x_mean_test)

    batch_size = 32
    num_epochs = 10

    # Making an MLP class instance.
    mlp = MLP(input_dims=input_dims, hidden_units=32, output_units=1)

    train_loss_set, train_accuracy_set, val_accuracy_set, val_loss_set = [], [], [], []

    for epoch in range(num_epochs):
        train_x, train_y = random_dist_data(x_train, y_train, num_examples)
        batch_loss, batch_accuracy = [], []
        num_batches = num_examples // batch_size
        for batch in range(num_batches):
            # Train the network.
            train_accuracy, train_loss = mlp.train(x_train, y_train, batch_size)

            train_loss_set.append(np.mean(train_loss))
            train_accuracy_set.append(np.mean(train_accuracy))

        # Validating the network at the end of each epoch
        validation_loss, validation_accuracy = mlp.evaluate(x_test, y_test, batch_size, l2_penalty=0)
        val_loss_set.append(validation_loss)
        val_accuracy_set.append(validation_accuracy)
        print("[EPOCH " + str(epoch + 1) + "/" + str(num_epochs) + "]" + "\tTrain Accuracy: " + str(round(np.mean(train_accuracy_set), 3)) + "\tTrain Loss: " + str(round(np.mean(train_loss_set), 3)) + "\tValidation Accuracy: " + str(round(np.mean(val_accuracy_set), 3)) + "\tValidation Loss: " + str(round(np.mean(val_loss_set), 3)))

    # Testing the network when training is done
    test_loss, test_accuracy = mlp.evaluate(x_test, y_test, batch_size, l2_penalty=0)
    print("Test Accuracy: " + str(round(test_accuracy, 3)) + "\tTest Loss: " + str(round(test_loss, 3)))
