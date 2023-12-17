import numpy as np

class MultiArrayEdu:
    def __init__(self, arg, arg2):
        '''
        다차원 배열 학습 클래스
        '''
        self.arg = arg
        self.arg2 = arg2
        print("Initializing MultiArrayEdu")

    def array_info(self):
        '''
        1차원 배열 테스트
        :return:
        '''
        print(self.arg)
        print(np.ndim(self.arg))
        print(self.arg.shape)
        print(self.arg.shape[0])

    def array_multiply(self):
        '''
        다차원 배열의 곱하기
        :return:
        '''
        array1 = self.arg[0]
        array2 = self.arg[1]
        array3 = self.arg[2]
        array4 = self.arg[3]

        a = np.array([array1, array2])
        b = np.array([array3, array4])

        print(np.dot(a, b))

    def dis_array_multiply(self):
        '''
        서로 다른 차원의 배열의 합
        :return:
        '''
        print(np.dot(self.arg, self.arg2))