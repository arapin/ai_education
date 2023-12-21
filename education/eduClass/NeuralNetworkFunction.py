import numpy as np


class NeuralNetWorkFunction:
    def __init__(self, arg):
        '''
        신경망 함수 클래스
        :param x:
        '''
        self.arg = arg

    def step_function(self):
        '''
        계층형 함수1
        :return:
        '''
        y = self.arg > 0
        return y.astype(np.int64)

    def step_function2(self):
        '''
        계층형 함수2
        :return:
        '''
        return np.array(self.arg > 0, dtype=np.int64)

    def sigmoid_function(self):
        '''
        시그모이드 함수
        :return:
        '''
        return 1 / (1 + np.exp(-self.arg))

    def relu_function(self):
        '''
        ReLU 함수
        :return:
        '''
        return np.maximum(0, self.arg)

    def identity_function(self):
        '''
        항등함수
        :return:
        '''
        return self.arg

    def softmax_function(self):
        '''
        소프트 맥스 함수
        :return:
        '''
        c = np.max(self.arg)
        exp_a = np.exp(self.arg - c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return y
