"""
Implémentation des réseaux de neurones
"""

import numpy as np

class Layer:
    """
    interface for neural network layers
    """
    def __init__(self, input_size, output_size, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function

    def back(self, x, delta, alpha):
        """
        abstract, update the weights, return delta for next layer
        """

    def predict(self, x):
        """
        abstract, make a prediction from this layer
        """

class NeuronLayer(Layer):
    """
    simple layer of neurons fully connected
    """
    def __init__(self, input_size, output_size, activation_function):
        Layer.__init__(self, input_size, output_size, activation_function)
        self._weights = 2*np.random.random_sample((input_size, output_size)) - 1
        self._bias = 2*np.random.random_sample((output_size)) - 1

    def back(self, x, delta, alpha):
        """
        update the weights, return delta for next layer
        """
        # compute gradient
        d_bias = delta * self.activation_function(x.dot(self._weights) + self._bias, True)
        d_weights = x.transpose().dot(d_bias)
        # save data for return
        tmp = self._weights.transpose()
        # update weights
        self._weights -= (d_weights * alpha)
        self._bias -= (d_bias * alpha)
        # return next delta
        return d_bias.dot(tmp)

    def predict(self, x):
        """
        make a prediction from this layer
        """
        return self.activation_function(x.dot(self._weights) + self._bias)

class DropoutLayer(Layer):
    """
    layer where some neurons are put to zero
    """
    def __init__(self, probability):
        Layer.__init__(self, None, None, None)
        self.probability = probability

    def back(self, x, delta, alpha):
        """
        do nothing
        """
        return delta

    def predict(self, x):
        """
        put some neurons to zero
        """
        for i in range(len(x)):
            if np.random.random_sample() < self.probability:
                x[i] = 0
        return x

################################################################################

class NeuralNetwork:
    """
    class containing all the layers of a neural network
    """
    def __init__(self, learning_rate):
        self.layers = list()
        self.learning_rate = learning_rate

    def one_step(self, x, y):
        """
        compute one step of gradient descent
        """
        intermediate_values = [x]
        for layer in self.layers:
            intermediate_values.append(layer.predict(intermediate_values[-1]))
        delta = intermediate_values[-1] - y
        for i in range(len(self.layers)):
            delta = self.layers[-i-1].back(intermediate_values[-i-1], delta)

    def predict(self, x):
        """
        compute x through the whole network
        """
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def add_layer(self, layer):
        """
        add a layer to the network
        """
        if self.layers == []:
            self.layers.append(layer)
        elif self.layers[-1].output_size == layer.input_size:
            self.layers.append(layer)
        else:
            raise ValueError(
                f"input_size ({layer.input_size}) does not match last layer \
                output_size ({self.layers[-1].output_size})"
            )
