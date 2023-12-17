import numpy as np
import matplotlib.pyplot as plt


class PyplotTest:
    def __init__(self):
        print("init pyplotTest")

    def plotTest(self):
        # 데이터 준비
        x = np.arange(0, 6, 0.1)
        y = np.sin(x)

        # 그래프 그리기
        plt.plot(x, y)
        plt.show()

    def plotTest2(self):
        # 데이터 준비
        x = np.arange(0, 6, 0.1)
        y1 = np.sin(x)
        y2 = np.cos(x)

        # 그래프 그리기
        plt.plot(x, y1, label="sin")
        plt.plot(x, y2, linestyle="--", label="cos")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("sin & cos")  # 제목
        plt.legend()
        plt.show()


pt = PyplotTest()
# pt.plotTest()
pt.plotTest2()
