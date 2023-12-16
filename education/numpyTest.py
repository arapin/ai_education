import numpy as np


class NumpyTest:
    def __init__(self):
        print("Initializing NumpyTest")

    def numpy_test(self):
        x = np.array([1.0, 2.0, 3.0])
        print(x)


npTest = NumpyTest()
npTest.numpy_test()
