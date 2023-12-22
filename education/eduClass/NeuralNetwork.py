import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from education.eduClass.mnist import load_mnist
from education.eduClass.NeuralNetworkFunction import NeuralNetWorkFunction


class NeuralNetwork:
    def __init__(self):
        filePath = "/Users/junghyunsu/PycharmProjects/myWorkSpace/ai_education/education/eduClass/sample_weight.pkl"
        with open(filePath, 'rb') as f:
            self.network = pickle.load(f)

    def get_data(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
        return x_test, t_test

    def predict(self, x):
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = NeuralNetWorkFunction(a1).sigmoid_function()
        a2 = np.dot(z1, W2) + b2
        z2 = NeuralNetWorkFunction(a2).sigmoid_function()
        a3 = np.dot(z2, W3) + b3
        y = NeuralNetWorkFunction(a3).softmax_function()
        return y