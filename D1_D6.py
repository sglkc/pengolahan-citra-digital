from PyQt5.QtWidgets import *   # type: ignore
from PyQt5.QtCore import pyqtSlot
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import cv2
from A9_C2 import A9_C2, InputDialog

class D1_D6(A9_C2):
    def __init__(self):
        super(D1_D6, self).__init__()

        self.popupButton: QPushButton
        self.kvMenu: QMenu

        self.popupButton.clicked.connect(self.popupClicked)
        self.kvMenu.triggered.connect(self.kvTrigger)

    def konvolusi(self, kernel: np.ndarray):
        image = self.imageOriginal.copy()
        image_H, image_W = image.shape[:2]
        kernel_H, kernel_W = kernel.shape[:2]

        H = kernel_H // 2
        W = kernel_W // 2

        for i in range(H, image_H-H):
            for j in range(W, image_W-W):
                sum =  0
                for k in range(-H, H+1):
                    for l in range(-W, W+1):
                        a = image[i+k, j+l]
                        w = kernel[H+k, W+l]
                        sum += (w * a)

                image[i, j] = sum

        self.imageResult = image
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
        }

        mapped.get(menuText)()      # type: ignore

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

if __name__ == '__main__':
    __import__('aplikasi').main()
