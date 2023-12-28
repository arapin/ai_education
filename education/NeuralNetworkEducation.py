import numpy as np
import education.eduClass.TwoLayerNet as tlnc

tln = tlnc.TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# print(tln.params['W1'].shape)
# print(tln.params['b1'].shape)
# print(tln.params['W2'].shape)
# print(tln.params['b2'].shape)
#
# x = np.random.rand(100, 784)
# y = tln.predict(x)
# print(y)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = tln.numerical_gradient(x, t)
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)