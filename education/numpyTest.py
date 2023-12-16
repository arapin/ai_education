import numpy as np


class NumpyTest:
    def __init__(self):
        print("Initializing NumpyTest")

    def numpy_test(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        print(x + y)
        print(x - y)
        print(x * y)
        print(x / y)
    def numpy_array_select(self):
        x = np.array([[51, 55], [14, 19], [0, 4]])
        print(x)
        print(x[0])
        print(x[0][1])

    def numpy_array_for(self):
        x = np.array([[51, 55], [14, 19], [0, 4]])
        for row in x:
            print(row)

    def numpy_array_flatten(self):
        x = np.array([[51, 55], [14, 19], [0, 4]])
        X = x.flatten()
        print(X)
        print(X[np.array([0, 2, 4])])
        print(X > 15)
        print(X[X>15])

npTest = NumpyTest()
# npTest.numpy_test()
# npTest.numpy_array_select()
# npTest.numpy_array_for()
npTest.numpy_array_flatten()
