from PyQt5.QtWidgets import *   # type: ignore
from PyQt5.QtCore import pyqtSlot
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import cv2
from A9_C2 import A9_C2

class D1_D6(A9_C2):
    def __init__(self):
        super(D1_D6, self).__init__()

        self.popupButton: QPushButton
        self.kvMenu: QMenu

        self.popupButton.clicked.connect(self.popupClicked)
        self.kvMenu.triggered.connect(self.kvTrigger)

    def konvolusi(self, kernel: np.ndarray):
        tinggi_citra, lebar_citra = self.imageOriginal.shape[:2]
        tinggi_kernel, lebar_kernel = kernel.shape[:2]
        H = tinggi_kernel // 2
        W = lebar_kernel // 2
        out = np.zeros_like(self.imageOriginal)

        for i in range(H, tinggi_citra - H):
            for j in range(W, lebar_citra - W):
                sum = 0
                for k in range(-H, H + 1):
                    for l in range(-W, W + 1):
                        a = self.imageOriginal[i + k, j + l]
                        w = kernel[H + k, W + l]
                        sum += w * a

                out[i, j] = np.clip(sum, 0, 255)

        self.imageResult = out
        self.displayImage(2)

    @pyqtSlot()
    def popupClicked(self):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        fig, axes = plt.subplots(1, 2)
        axes: tuple[Axes, Axes]
        axes[0].imshow(cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2RGB))

        if not hasattr(self, 'imageResult'):
            axes[1].remove()
        elif len(self.imageResult.shape) == 3:
            axes[1].imshow(cv2.cvtColor(self.imageResult, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(cv2.cvtColor(self.imageResult, cv2.COLOR_GRAY2BGR))

        axes[0].axis('off')
        axes[1].axis('off')
        fig.tight_layout()
        fig.show()

    @pyqtSlot(QAction)
    def kvTrigger(self, action):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Mean Filter': self.__mean,
            'Gaussian Filter': self.__gaussian,
            'Median Filter': self.__median,
            'Max Filter': self.__max
        }

        sharpening = ['i', 'ii', 'iii', 'iv', 'v', 'vi']

        if menuText in sharpening:
            self.__sharpening(menuText)
        else:
            mapped.get(menuText)()      # type: ignore

        return True

    def __mean(self):
        kernel = np.full((3, 3), 1/9)
        self.konvolusi(kernel)

    def __gaussian(self):
        kernel = (1.0 / 345) * np.array([[1, 5, 7, 5, 1],
                                         [5, 20, 33, 20, 5],
                                         [7, 33, 55, 33, 7],
                                         [5, 20, 33, 20, 5],
                                         [1, 5, 7, 5, 1]
                                         ])

        self.konvolusi(kernel)

    def __sharpening(self, filter):
        mapped = {
            'i': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            'ii': [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
            'iii': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
            'iv': [[1, -2, 1], [-2, 5, -2], [1, -2, 1]],
            'v': [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
            'vi': [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        }

        kernel = 1.0 * np.array(mapped.get(filter))
        self.konvolusi(kernel)

    def __median(self):
        H, W = self.imageOriginal.shape[:2]
        out = self.imageOriginal.copy()

        for i in range(3, H - 3):
            for j in range(3, W - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = self.imageOriginal[i + k, j + l]
                        neighbors.append(a)

                neighbors.sort(key=lambda x: sum(x))
                median = neighbors[24]
                out[i, j] = median

        self.imageResult = out
        self.displayImage(2)

    def __max(self):
        H, W = self.imageOriginal.shape[:2]
        out = self.imageOriginal.copy()

        for i in range(3, H - 3):
            for j in range(3, W - 3):
                pixel = self.imageOriginal[i, j]
                max_val = 0

                for k in range(-3, 4):
                    for l in range(-3, 4):
                        pixel_value = self.imageOriginal[i + k, j + l]
                        intensity = np.mean(pixel_value)

                        if intensity > max_val:
                            max_val = intensity
                            pixel = pixel_value

                out[i, j] = pixel

        self.imageResult = out
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
