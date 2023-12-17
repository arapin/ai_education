import numpy as np
import eduClass.ThreeStepNeuralnetwork as tsn

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
# nen4 = nnt.NeuralNetWork(xxx)
# yyy = nen4.sigmoid_function()
# plt.plot(xxx, yyy)
# plt.ylim(-0.1, 1.1)
# plt.show()

# xxx = np.arange(-5.0, 5.0, 0.1)
# nen4 = nnt.NeuralNetWork(xxx)
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
tsnc = tsn.ThreeStepNeuralNetwork(np.array([1.0, 0.5]))
y = tsnc.forward_function()
print(y)
