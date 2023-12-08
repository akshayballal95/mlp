from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils import *


class MultiLayerPerceptron:
    def __init__(self, layers):
        self.layers = layers

        self.params = {}

    def initialize_parameters(self):
        for l in range(1, len(self.layers)):
            in_size = self.layers[l - 1]
            out_size = self.layers[l]

            self.params["w" + str(l)] = np.random.randn(out_size, in_size) / np.sqrt(
                in_size
            )
            self.params["b" + str(l)] = np.zeros((out_size, 1))

    def model_forward(self, X):
        number_of_layers = len(self.layers) - 1
        a = X
        caches = {}

        for l in range(1, number_of_layers):
            a_prev = a
            weight_string = "w" + str(l)
            bias_string = "b" + str(l)
            w = self.params[weight_string]
            b = self.params[bias_string]
            a, cache = linear_forward_activation(a_prev, w, b, "relu")
            caches[str(l)] = cache

        weight_string = "w" + str(number_of_layers)
        bias_string = "b" + str(number_of_layers)

        w = self.params[weight_string]
        b = self.params[bias_string]

        al, cache = linear_forward_activation(a, w, b, "sigmoid")
        caches[str(number_of_layers)] = cache

        return (al, caches)

    def model_backward(self, al, y, caches):
        grads = {}
        num_layers = len(self.layers) - 1
        dal = -(y / al - (1 - y) / (1 - al))

        current_cache = caches[str(num_layers)]
        da_prev, dw, db = linear_backward_activation(dal, current_cache, "sigmoid")

        weight_string = "dw" + str(num_layers)
        bias_string = "db" + str(num_layers)
        activation_string = "dA" + str(num_layers)

        grads[weight_string] = dw
        grads[bias_string] = db
        grads[activation_string] = da_prev

        for l in reversed(range(1, num_layers)):
            current_cache = caches[str(l)]
            da_prev, dw, db = linear_backward_activation(da_prev, current_cache, "relu")

            weight_string = "dw" + str(l)
            bias_string = "db" + str(l)
            activation_string = "dA" + str(l)

            grads[weight_string] = dw
            grads[bias_string] = db
            grads[activation_string] = da_prev

        return grads

    def update_parameters(self, grads, learning_rate):
        num_of_layers = len(self.layers) - 1

        for l in range(1, num_of_layers + 1):
            weight_string_grad = "dw" + str(l)
            bias_string_grad = "db" + str(l)
            weight_string = "w" + str(l)
            bias_string = "b" + str(l)

            self.params[weight_string] = (
                self.params[weight_string] - learning_rate * grads[weight_string_grad]
            )
            self.params[bias_string] = (
                self.params[bias_string] - learning_rate * grads[bias_string_grad]
            )

    def predict(self, X):
        out, _ = self.model_forward(X)
        return out.argmax(axis=0)

    """
    Train the network using the input data and labels.

    Args:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The true labels.
        iterations (int, optional): The number of iterations for training. Defaults to 20000.
        learning_rate (float, optional): The learning rate for the update. Defaults to 0.1.
    """
    def train(self, X, y, iterations=20000, learning_rate=0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # One-hot encode labels
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).T
        y_val = encoder.fit_transform(y_val.reshape(-1, 1)).T

        X_train = X_train.T
        X_val = X_val.T

        self.initialize_parameters()
        for epoch in range(iterations):
            al, caches = self.model_forward(X_train)
            grads = self.model_backward(al, y_train, caches)
            self.update_parameters(grads, learning_rate)

            train_predictions = self.predict(X_train)
            val_predictions = self.predict(X_val)
            if epoch % 200 == 0:
                print(
                    f"Epoch: {epoch}/{iterations} || Training Accuracy: {accuracy_score(y_train.argmax(axis = 0),train_predictions):.2f} || Validation Accuracy: {accuracy_score(y_val.argmax(axis = 0),val_predictions):.2f}"
                )
