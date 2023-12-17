import numpy as np
class perceptron:
    def AND(self, x1, x2):
        w1, w2, theta = 0.5, 0.5, 0.7
        tmp = x1*w1 + x2*w2
        if tmp <= theta:
            return 0
        elif tmp > theta:
            return 1

    def BANDCAL(self):
        x = np.array([0, 1])
        w = np.array([0.5, 0.5])
        b = -0.7
        print(w*x)
        print(np.sum(w*x))
        print(np.sum(w*x) + b)

    def BAND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def NAND(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([-0.5, -0.5])
        b = 0.7
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def OR(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.5, 0.5])
        b = -0.2
        tmp = np.sum(w*x) + b
        if tmp <= 0:
            return 0
        else:
            return 1

    def XOR(self,x1, x2):
        s1 = self.NAND(x1, x2)
        s2 = self.OR(x1, x2)
        y = self.AND(s1, s2)
        return y

perc = perceptron()
# print(perc.AND(0, 0))
# print(perc.AND(1, 0))
# print(perc.AND(0, 1))
# print(perc.AND(1, 1))
# perc.BANDCAL()
# print(perc.BAND(0, 0))
# print(perc.BAND(1, 0))
# print(perc.BAND(0, 1))
# print(perc.BAND(1, 1))
# print(perc.NAND(0, 0))
# print(perc.NAND(1, 0))
# print(perc.NAND(0, 1))
# print(perc.NAND(1, 1))
# print(perc.OR(0, 0))
# print(perc.OR(1, 0))
# print(perc.OR(0, 1))
# print(perc.OR(1, 1))
print(perc.XOR(0, 0))
print(perc.XOR(1, 0))
print(perc.XOR(0, 1))
print(perc.XOR(1, 1))