import numpy as np


class Perceptron:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        print("Initializing Perceptron")
    def and_function(self, xx1, xx2):
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = xx1 * w1 + xx2 * w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    def bandcal(self):
        x = np.array([0, 1])
        w = np.array([0.5, 0.5])
        b = -0.7
        print(w * x)
        print(np.sum(w * x))
        print(np.sum(w * x) + b)

    def band(self):
        x = np.array([self.x1, self.x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def nand(self):
        x = np.array([self.x1, self.x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def or_function(self):
        x = np.array([self.x1, self.x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w * x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def xor(self):
        s1 = self.nand()
        s2 = self.or_function()
        y = self.and_function(s1, s2)
        return y
