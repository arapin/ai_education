import sys, os

sys.path.append(os.pardir)
import numpy as np
from education.eduClass.mnist import load_mnist
from PIL import Image


class MnistClass:
    def __init__(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        img = x_train[0]
        img = img.reshape(28, 28)
        self.image = img

    def img_show(self):
        pil_img = Image.fromarray(np.uint8(self.image))
        pil_img.show()
