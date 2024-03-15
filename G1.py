from types import LambdaType
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QMenu
import cv2

from F1_F2 import F1_F2

class G1(F1_F2):
    def __init__(self):
        super(G1, self).__init__()
        self.mfMenu: QMenu
        self.mfMenu.triggered.connect(self.mfTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def mfTrigger(self, action: QAction):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        menuText = action.text()
        mapped = {
            'Dilasi': lambda i, k: cv2.dilate(i, k),
            'Erosi': lambda i, k: cv2.erode(i, k),
            'Opening': lambda i, k: cv2.morphologyEx(i, cv2.MORPH_OPEN, k),
            'Closing': lambda i, k: cv2.morphologyEx(i, cv2.MORPH_CLOSE, k)
        }

        self.morfologi(mapped.get(menuText))        # type: ignore

    def morfologi(self, fungsi: LambdaType):
        # grayscalisasikan
        # binerisasi

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        image = fungsi(self.imageOriginal, kernel)

        self.imageResult = image
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
