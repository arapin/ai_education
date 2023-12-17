import numpy as np
import education.eduClass.NeuralNetworkFunction as nnt


class ThreeStepNeuralNetwork:
    def __init__(self, arg):
        '''
        3층 신경망
        :param arg:
        '''
        self.network = {}
        self.network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
        self.network['b1'] = np.array([0.1, 0.2, 0.3])
        self.network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
        self.network['b2'] = np.array([0.1, 0.2])
        self.network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
        self.network['b3'] = np.array([0.1, 0.2])
        self.arg = arg

    def forward_function(self):
        '''
        3층 신경망 계산 함수
        :return:
        '''
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(self.arg, W1) + b1
        z1 = nnt.NeuralNetWorkFunction(a1).sigmoid_function()
        a2 = np.dot(z1, W2) + b2
        z2 = nnt.NeuralNetWorkFunction(a2).sigmoid_function()
        a3 = np.dot(z2, W3) + b3
        y = nnt.NeuralNetWorkFunction(a3).identity_function()

        return y
