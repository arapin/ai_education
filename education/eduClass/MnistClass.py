import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from PIL import Image
class MnistClass:
    def img_show(img):
        pil_img = Image.fromarray(np.uint8(img))
        pil_img.show()
    def mnist_show(self):
        (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
        img = x_train[0]
        img = img.reshape(28, 28)
        self.img_show(img)