import typing
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QDialog, QMenu
import cv2

from A9_C2 import InputDialog
from G1 import G1

class H1_H3(G1):
    def __init__(self):
        super(H1_H3, self).__init__()

        self.thMenu: QMenu
        self.thGlobal: QMenu
        self.thLocalAdapt: QMenu

        self.thGlobal.triggered.connect(self.thGlobalTrigger)   # type: ignore
        self.thLocalAdapt.triggered.connect(self.thLocalAdaptTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def thGlobalTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Binary': cv2.THRESH_BINARY,
            'Binary Invert': cv2.THRESH_BINARY_INV,
            'Trunc': cv2.THRESH_TRUNC,
            'To Zero': cv2.THRESH_TOZERO,
            'To Zero Invert': cv2.THRESH_TOZERO_INV,
        }

        fungsi = typing.cast(int, mapped.get(menuText))
        dialog = InputDialog([ ['Nilai Ambang', 50, 'slider'] ])

        if dialog.exec() == QDialog.Rejected: return

        thres = dialog.getValues(lambda val: int(val) * 2.55)[0]
        # grayscalisasikan
        _, image = cv2.threshold(self.imageOriginal, thres, 255, fungsi)

        self.imageResult = image
        self.displayImage(2)

    @pyqtSlot(QAction)
    def thLocalAdaptTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Mean': cv2.ADAPTIVE_THRESH_MEAN_C,
            'Gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            'Otsu': False
        }

        fungsi = typing.cast(int, mapped.get(menuText))
        image = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)

        if menuText == 'Otsu':
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            image = cv2.adaptiveThreshold(image, 255, fungsi, cv2.THRESH_BINARY, 3, 2)

        self.imageResult = image
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
