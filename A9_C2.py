import cv2
from cv2.typing import MatLike
from A1_A8 import A1_A8
from PyQt5.QtWidgets import *   # type: ignore
from PyQt5.QtCore import pyqtSlot
from matplotlib import pyplot as plt
import numpy as np

# pyright: reportOptionalMemberAccess=false

class InputDialog(QDialog):
    def __init__(self, inputs=[]):
        super(InputDialog, self).__init__()
        self.setWindowTitle('Pengaturan')
        self.form = QFormLayout()
        self.input: list[QWidget] = []

        for input in inputs:
            self.addInput(input[0], input[1])

    def addInput(self, label: str, placeholder: str):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        self.form.addRow(QLabel(label), edit)
        self.input.append(edit)

    def getValues(self, map_func=None):
        def evaluate(val):
            try: return eval(val)
            except: return False

        mapped = list(map(lambda w: evaluate(w.text()) or w.placeholderText(),
                          self.input))

        return list(map(map_func or int, mapped))

    def exec(self):
        self.ok = QPushButton("OK")
        self.ok.setDefault(True)
        self.ok.clicked.connect(self.accept)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        self.form.addRow(self.cancel, self.ok)
        self.setLayout(self.form)

        return super().exec()

class A9_C2(A1_A8):
    def __init__(self):
        super(A9_C2, self).__init__()

        self.swapButton: QPushButton
        self.hgMenu: QMenu
        self.tfMenu: QMenu
        self.opMenu: QMenu

        self.swapButton.clicked.connect(self.swapImages)    # type: ignore
        self.hgMenu.triggered.connect(self.hgTrigger)       # type: ignore
        self.tfMenu.triggered.connect(self.tfTrigger)       # type: ignore
        self.opMenu.triggered.connect(self.opTrigger)       # type: ignore

    @pyqtSlot()
    def swapImages(self):
        if not hasattr(self, 'imageResult'):
            return self.showMessage('Error', 'Hasil citra masih kosong!')

        if len(self.imageResult.shape) == 3:
            self.imageOriginal = self.imageResult
        else:
            self.imageOriginal = cv2.cvtColor(self.imageResult, cv2.COLOR_GRAY2BGR)

        self.displayImage()

    @pyqtSlot(QAction)
    def hgTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Grayscale': self.__grayscale,
            'RGB': self.__rgb,
            'Equalization': self.__equalization
        }

        mapped.get(menuText)()      # type: ignore

    @pyqtSlot(QAction)
    def tfTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Translasi': self.__translasi,
            'Rotasi': self.__rotasi,
            'Skala': self.__skala,
            'Crop': self.__crop
        }

        mapped.get(menuText)()      # type: ignore

    @pyqtSlot(QAction)
    def opTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        file1 = self.originalLabel.toolTip()
        file2, _ = QFileDialog().getOpenFileName(
            filter='Citra (*.png *.jpeg *.jpg *.bmp)')

        if not file2: return

        image1: MatLike
        image2: MatLike

        if menuText in ['Add', 'Subtract']:
            image1 = cv2.imread(file1, 0)
            image2 = cv2.imread(file2, 0)
        else:
            image1 = cv2.imread(file1, 0)
            image2 = cv2.imread(file2, 1)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        try:
            mapped = {
                'Add': lambda: image1 + image2,         # type: ignore
                'Subtract': lambda: image1 - image2,    # type: ignore
                'AND': lambda: cv2.bitwise_and(image1, image2),
                'OR': lambda: cv2.bitwise_or(image1, image2),
                'XOR': lambda: cv2.bitwise_xor(image1, image2),
            }

            self.imageResult = mapped.get(menuText)()     # type: ignore
            self.displayImage(2)
        except:
            self.showMessage('Error', 'Citra 1 dan 2 harus berukuran sama!')

    def __grayscale(self):
        plt.hist(self.imageOriginal.ravel(), 255, (0, 255))
        plt.show()

    def __rgb(self):
        for i, color in enumerate(('b', 'g', 'r')):
            histo = cv2.calcHist([self.imageOriginal], [i], None, [255], [0, 255])
            plt.plot(histo, color=color)
            plt.xlim([0, 256])

        plt.show()

    def __equalization(self):
        hist, _ = np.histogram(self.imageOriginal.flatten(), 256, (0, 255))
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.imageResult = cdf[self.imageOriginal]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.imageResult.flatten(), 256, (0, 256), color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def __translasi(self):
        dialog = InputDialog([ ["X", "0"], ["Y", "0"] ])

        if dialog.exec() == QDialog.Rejected: return

        X, Y = dialog.getValues()
        H, W = self.imageOriginal.shape[:2]
        H2 = H * X / H
        W2 = W * Y / W
        T = np.array([[1, 0, H2], [0, 1, W2]])
        self.imageResult = cv2.warpAffine(self.imageOriginal, T, (W, H))
        self.displayImage(2)

    def __rotasi(self):
        dialog = InputDialog([ ["Derajat", "0"] ])

        if dialog.exec() == QDialog.Rejected: return

        deg = dialog.getValues()[0]
        H, W = self.imageOriginal.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((W / 2, H / 2), deg, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        W2 = int((H * sin) + (W * cos))
        H2 = int((H * cos) + (W * sin))
        rotationMatrix[0, 2] += (W2 / 2) - W / 2
        rotationMatrix[1, 2] += (H2 / 2) - H / 2
        rot_image = cv2.warpAffine(self.imageOriginal, rotationMatrix, (H, W))
        self.imageResult = rot_image
        self.displayImage(2)

    def __skala(self):
        dialog = InputDialog([ ["Skala X", "1"], ["Skala Y", "1"] ])

        if dialog.exec() == QDialog.Rejected: return

        fx, fy = dialog.getValues()

        try:
            self.imageResult = cv2.resize(self.imageOriginal, None, None,
                fx, fy, cv2.INTER_CUBIC)
            self.displayImage(2)
        except:
            self.showMessage('Error', 'Input harus skala > 0!')

    def __crop(self):
        dialog = InputDialog([
            ["X1", "0"], ["Y1", "0"],
            ["X2", "0"], ["Y2", "0"],
        ])

        if dialog.exec() == QDialog.Rejected: return

        X1, Y1, X2, Y2 = dialog.getValues()
        self.imageResult = self.imageOriginal[Y1:Y2, X1:X2].copy()
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
