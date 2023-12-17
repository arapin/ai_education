import numpy as np


class NeuralNetWork:
    def __init__(self, x):
        '''
        신경망 클래스
        :param x:
        '''
        self.x = x

    def step_function(self):
        '''
        계층형 함수1
        :return:
        '''
        y = self.x > 0
        return y.astype(np.int64)

    def step_function2(self):
        '''
        계층형 함수2
        :return:
        '''
        return np.array(self.x > 0, dtype=np.int64)

    def sigmoid_function(self):
        '''
        시그모이드 함수
        :return:
        '''
        return 1 / (1 + np.exp(-self.x))

    def relu_function(self):
        '''
        ReLU 함수
        :return:
        '''
        return np.maximum(0, self.x)