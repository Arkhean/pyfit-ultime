"""
Implémentation des réseaux de neurones
"""


class Layer:
    """
    interface for neural network layers
    """
    def __init__(self):
        pass

    def train(self):
        """
        """

    def predict(self):
        """
        make a prediction from this layer
        """

class NeuronLayer(Layer):
    """
    simple layer of neurons fully connected
    """
    def __init__(self):
        Layer.__init__(self)

    def train(self):
        """"""

    def predict(self):
        """"""

class DropoutLayer(Layer):
    """
    layer where some neurons are put to zero
    """
    def __init__(self):
        Layer.__init__(self)

    def train(self):
        """"""

    def predict(self):
        """"""

################################################################################

class NeuralNetwork:
    """
    class containing all the layers of a neural network
    """
    def __init__(self):
        pass

    def train(self):
        """"""

    def predict(self):
        """"""

    def add_layer(self):
        """
        add a layer to the network
        """
