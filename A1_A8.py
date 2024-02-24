import cv2
import math
from cv2.typing import MatLike
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *   # type: ignore
from PyQt5.uic import loadUi

# pyright: reportArgumentType=false

class A1_A8(QMainWindow):
    def __init__(self):
        super(A1_A8, self).__init__()
        loadUi('gui.ui', self)

        self.imageOriginal: MatLike
        self.imageResult: MatLike
        self.originalLabel: QLabel
        self.resultLabel: QLabel
        self.loadButton: QPushButton
        self.resultButton: QPushButton
        self.filterCombo: QComboBox
        self.slider: QSlider

        self.loadButton.clicked.connect(self.loadClicked)
        self.filterCombo.currentTextChanged.connect(self.filterComboChanged)
        self.resultButton.clicked.connect(self.resultClicked)

    def showMessage(self, title: str, text: str, icon=QMessageBox.Critical):
        QMessageBox(icon, title, text).exec()

    @pyqtSlot()
    def loadClicked(self):
        file, _ = QFileDialog().getOpenFileName(
            filter='Citra (*.png *.jpeg *.jpg *.bmp)')

        if file:
            self.loadImage(file)
            self.originalLabel.setToolTip(file)

    @pyqtSlot()
    def filterComboChanged(self):
        value = self.filterCombo.currentText()
        mapped = {
            'Grayscale': False,
            'Brightness': True,
            'Contrast': True,
            'Stretching': False,
            'Negative': False,
            'Binary': True,
        }

        self.slider.setEnabled(mapped.get(value))

    @pyqtSlot()
    def resultClicked(self):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        value = self.filterCombo.currentText()
        mapped = {
            'Grayscale': self.__grayscale,
            'Brightness': self.__brightness,
            'Contrast': self.__contrast,
            'Stretching': self.__stretching,
            'Negative': self.__negative,
            'Binary': self.__binary,
        }

        mapped.get(value)() # type: ignore
        self.displayImage(2)

    def __grayscale(self):
        H, W = self.imageOriginal.shape[:2]
        image = np.zeros((H, W), np.uint8)

        for i in range(H):
            for j in range(W):
                image[i, j] = np.clip(
                    0.333 * self.imageOriginal[i, j, 0] +
                    0.333 * self.imageOriginal[i, j, 1] +
                    0.333 * self.imageOriginal[i, j, 2], 0, 255)

        self.imageResult = image

    def __brightness(self):
        self.imageResult = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        H, W = self.imageResult.shape[:2]
        brightness = int(self.slider.value())

        for i in range(H):
            for j in range(W):
                a = self.imageResult.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.imageResult.itemset((i, j), b)

    def __contrast(self):
        self.imageResult = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        H, W = self.imageResult.shape[:2]
        contrast = 1 + int(self.slider.value()) / 100

        for i in range(H):
            for j in range(W):
                a = self.imageResult.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.imageResult.itemset((i, j), b)

    def __stretching(self):
        self.imageResult = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        H, W = self.imageResult.shape[:2]
        minValue = np.min(self.imageResult)
        maxValue = np.max(self.imageResult)

        for i in range(H):
            for j in range(W):
                a = self.imageResult.item(i, j)
                b = float(a - minValue) - float(a - maxValue) * 255

                self.imageResult.itemset((i, j), b)

    def __negative(self):
        self.imageResult = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        H, W = self.imageResult.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.imageResult.item(i, j)
                b = math.ceil(255 - a)

                self.imageResult.itemset((i, j), b)

    def __binary(self):
        self.imageResult = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        H, W = self.imageResult.shape[:2]
        threshold = int(self.slider.value()) * 2.55

        for i in range(H):
            for j in range(W):
                a = self.imageResult.item(i, j)
                b = 255 if a >= threshold else 0

                self.imageResult.itemset((i, j), b)

    def loadImage(self, file):
        self.imageOriginal = cv2.imread(file)
        self.displayImage()

    def displayImage(self, label=1):
        qformat = QImage.Format.Format_Indexed8
        image = self.imageOriginal if label == 1 else self.imageResult

        if len(image.shape) == 3:
            if (image.shape[2]) == 4:
                qformat = QImage.Format.Format_RGBA8888
            else:
                qformat = QImage.Format.Format_RGB888

        img = QImage(image.data,
                     image.shape[1],
                     image.shape[0],
                     image.strides[0],
                     qformat)

        img = img.rgbSwapped()

        if label == 1:
            self.originalLabel.setPixmap(QPixmap.fromImage(img))
        else:
            self.resultLabel.setPixmap(QPixmap.fromImage(img))

if __name__ == '__main__':
    __import__('aplikasi').main()
