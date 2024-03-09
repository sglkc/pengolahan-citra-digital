from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QDialog
import cv2
import numpy as np

from A9_C2 import InputDialog
from E1_E2 import E1_E2

class F1_F2(E1_E2):
    def __init__(self):
        super(F1_F2, self).__init__()

    @pyqtSlot(QAction)
    def kvTrigger(self, action: QAction):
        if super().kvTrigger(action): return True

        menuText = action.text()
        mapped = {
            'Sobel': lambda: self.__edgeDetect('sobel'),
            'Prewitt': lambda: self.__edgeDetect('prewitt'),
            'Canny': self.__canny,
        }

        try:
            mapped.get(menuText)()      # type: ignore
            return True
        except Exception as error:
            if isinstance(error, TypeError):
                return False

            raise error

    def __edgeDetect(self, method='sobel'):
        img = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Kernel default sobel
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, -2],
                             [-1, 0, -1]])
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

        if method == 'prewitt':
            kernel_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])

        Gx = self.konvolusi(kernel_x, img, False)
        Gy = self.konvolusi(kernel_y, img, False)

        gradien = np.sqrt((Gx * Gx) + (Gy * Gy))    # type: ignore
        normal = cv2.normalize(gradien, gradien, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.imageResult = cv2.cvtColor(normal, cv2.COLOR_GRAY2BGR)
        self.displayImage(2)

    def __canny(self):
        dialog = InputDialog([ ['Weak', 50, 'slider'], ['Strong', 100, 'slider'] ])

        if dialog.exec() == QDialog.Rejected: return

        weak, strong = dialog.getValues(lambda val: int(val) * 2.55)
        img = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gauss = (1.0 / 57) * np.array([
            [0, 1, 2, 1, 0],
            [1, 3, 5, 3, 1],
            [2, 5, 9, 5, 2],
            [1, 3, 5, 3, 1],
            [0, 1, 2, 1, 0],
        ])

        img = self.konvolusi(gauss, img, False)

        kernel_x = np.array([[-1, 0, 1],
             [-2, 0, -2],
             [-1, 0, -1]])

        kernel_y = np.array([[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]])

        Gx = self.konvolusi(kernel_x, img, False)
        Gy = self.konvolusi(kernel_y, img, False)

        magnitude = np.sqrt((Gx ** 2) + (Gy ** 2))      # type: ignore
        magnitude = (magnitude / np.max(magnitude)) * 255
        theta = np.arctan2(Gx, Gy)

        angle = theta * 180 / np.pi
        angle[angle < 0] += 180

        H, W = magnitude.shape[:2]
        Z = np.zeros((H, W), dtype=np.uint8)

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q, r = 255, 255

                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q, r = magnitude[i, j+1], magnitude[i, j-1]
                    elif (22.5 <= angle[i, j] < 67.5):
                        q, r = magnitude[i+1,j-1], magnitude[i-1, j+1]
                    elif (67.5 <= angle[i, j] < 112.5):
                        q, r = magnitude[i+1,j], magnitude[i-1, j]
                    elif (112.5 <= angle[i, j] < 157.5):
                        q, r = magnitude[i-1,j-1], magnitude[i+1, j+1]

                    if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                        Z[i, j] = magnitude[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError:
                    pass

        for i in range(H):
            for j in range(W):
                a = Z[i, j]

                if a > weak:
                    Z[i, j] = weak
                    if a > strong: Z[i, j] = 255
                else:
                    Z[i, j] = 0

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if Z[i, j] == weak:
                    try:
                        if (
                            (Z[i+1, j-1] == strong) or (Z[i+1, j] == strong) or
                            (Z[i+1, j+1] == strong) or (Z[i, j-1] == strong) or
                            (Z[i, j+1] == strong) or (Z[i-1, j-1] == strong) or
                            (Z[i-1, j] == strong) or (Z[i-1, j+1] == strong)
                            ):
                            Z[i, j] = strong
                        else:
                            Z[i, j] = 0
                    except IndexError:
                        pass

        self.imageResult = cv2.cvtColor(Z, cv2.COLOR_GRAY2BGR)
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
