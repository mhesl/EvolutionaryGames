import numpy as np

from util import load_generation


class NeuralNetwork:
    def __init__(self, layer_sizes, mode):
        if mode != 'thrust':
            input_layer_size, hidden_layer_size, output_layer_size = layer_sizes
            self.W1 = np.random.normal(size=(hidden_layer_size, input_layer_size))
            self.W2 = np.random.normal(size=(output_layer_size, hidden_layer_size))
            self.B1 = np.zeros((hidden_layer_size, 1))
            self.B2 = np.zeros((output_layer_size, 1))
        else:
            l1, l2, l3, l4 = layer_sizes
            players = load_generation('checkpoint/helicopter/43', '')
            players.sort(key=lambda x: x.fitness)
            self.W1 = players[-1].nn.W1
            self.W2 = np.random.normal(size=(l3, l2))
            self.W3 = np.random.normal(size=(l4, l3))
            self.B1 = players[-1].nn.B1
            self.B2 = np.zeros((l3, 1))
            self.B3 = np.zeros((l4, 1))

    def activation(self, x, type):
        if type == 'RELO':
            return x * (x > 0)
        if type == 'C_RELO':
            return x * (x > 0) + 0.01 * x * (x <= 0)
        if type == 'LINEAR':
            return x
        if type == 'SIGMOID':
            return 1 / (1 + np.e ** (-x))
        if type == 'SOFTMAX':
            ex = np.e ** x
            return ex / np.sum(ex)
        if type == 'TANH':
            exp_pos = np.exp(x)
            exp_neg = np.exp(-x)
            return (exp_pos - exp_neg) / (exp_pos + exp_neg)
        raise ValueError("type didn't found!")

    def forward(self, x, mode):
        O1 = np.dot(self.W1, x) + self.B1
        O1 = self.activation(O1, 'TANH')
        O2 = np.dot(self.W2, O1) + self.B2
        if mode != 'thrust':
            O2 = self.activation(O2, 'SIGMOID')
            return O2
        else:
            O2 = self.activation(O2, 'TANH')
            O3 = np.dot(self.W3, O2) + self.B3
            O3 = self.activation(O3, 'SIGMOID')
            return O3

