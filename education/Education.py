import numpy as np
import matplotlib.pyplot as plt
import eduClass.ThreeStepNeuralnetwork as tsn
import eduClass.NeuralNetworkFunction as nnt
from eduClass.MnistClass import MnistClass
from eduClass.NeuralNetwork import NeuralNetwork

# perc = pt.Perceptron(0, 0)
# perc2 = pt.Perceptron(1, 0)
# perc3 = pt.Perceptron(0, 1)
# perc4 = pt.Perceptron(1, 1)
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
# print(perc.xor())
# print(perc2.xor())
# print(perc3.xor())
# print(perc4.xor())

# nen = nnt.NeuralNetWork(np.array([-1.0, 1.0, 2.0]))
# print(nen.step_function())
#
# xx = np.arange(-5.0, 5.0, 0.1)
# nen2 = nnt.NeuralNetWork(xx)
# yy = nen2.step_function2()
# plt.plot(xx, yy)
# plt.ylim(-0.1, 1.1)
# plt.show()
#
# nen3 = nnt.NeuralNetWork(np.array([-1.0, 1.0, 2.0]))
# print(nen3.sigmoid_function())
#
# xxx = np.arange(-5.0, 5.0, 0.1)
# nen4 = nnt.NeuralNetWorkFunction(xxx)
# yyy = nen4.sigmoid_function()
# plt.plot(xxx, yyy)
# plt.ylim(-0.1, 1.1)
# plt.show()

# xxx = np.arange(-5.0, 5.0, 0.1)
# nen4 = nnt.NeuralNetWorkFunction(xxx)
# yyy = nen4.relu_function()
# plt.plot(xxx, yyy)
# plt.ylim(-0.1, 5)
# plt.show()

# maec = mae.MultiArrayEdu(np.array([1, 2, 3, 4, 5]), np.array([]))
# maec.array_info()
#
# maec2 = mae.MultiArrayEdu(np.array([[1, 2], [3, 4], [5, 6]]), np.array([]))
# maec2.array_info()
#
# maec3 = mae.MultiArrayEdu(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([]))
# maec3.array_multiply()
#
# maec4 = mae.MultiArrayEdu(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[1, 2], [3, 4], [5, 6]]))
# maec4.dis_array_multiply()
#
# maec5 = mae.MultiArrayEdu(np.array([[1, 2], [3, 4], [5, 6]]), np.array([7, 8]))
# maec5.dis_array_multiply()
#
# maec6 = mae.MultiArrayEdu(np.array([1, 2]), np.array([[1, 3, 5], [2, 4, 6]]))
# maec6.dis_array_multiply()

# 3층 신경망 학습
# tsnc = tsn.ThreeStepNeuralNetwork(np.array([1.0, 0.5]))
# y = tsnc.forward_function()
# print(y)

# 소프트맥스 함수
# nntfc = nnt.NeuralNetWorkFunction(np.array([0.3, 2.9, 4.0])).softmax_function()
# print(nntfc)
# y = nnt.NeuralNetWorkFunction(np.array([0.3, 2.9, 4.0])).softmax_function()
# print(y)
# print(np.sum(y))
# 이미지 출력
# mc = MnistClass()
# mc.img_show()
# mnist 데이터 추론
nn = NeuralNetwork()
accuracy_cnt = 0
# print("Accuracy:" + nn.accuracy_result(accuracy_cnt))
print("Accuracy:" + nn.accuracy_batch_result(accuracy_cnt))
#
# def mean_squared_error(y, t):
#     return 0.5 * np.sum((y-t)**2)
#
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y+delta))

# t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
# print(mean_squared_error(np.array(t), np.array(y)))
# print(cross_entropy_error(np.array(y), np.array(t)))

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return np.sum(x**2)

def function_3(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1
def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

# x = np.arange(0.0, 20.0, 0.1)
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()
# print(numerical_diff(function_1, 5))
# print(numerical_diff(function_1, 10))
# print(numerical_diff(function_tmp1, 3.0))
# print(numerical_diff(function_tmp2, 4.0))
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_3, init_x=init_x, lr=0.1, step_num=100))
print(gradient_descent(function_3, init_x=init_x, lr=10.0, step_num=100))
print(gradient_descent(function_3, init_x=init_x, lr=1e-10, step_num=100))
