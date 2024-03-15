from types import LambdaType
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QDialog, QMenu
import cv2

from A9_C2 import InputDialog
from G1 import G1

class H1_H3(G1):
    def __init__(self):
        super(H1_H3, self).__init__()
        self.ltMenu: QMenu
        self.ltMenu.triggered.connect(self.ltTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def ltTrigger(self, action: QAction):
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

        self.localThres(mapped.get(menuText))        # type: ignore

    def localThres(self, fungsi: int):
        # grayscalisasikan

        dialog = InputDialog([ ['Nilai Ambang', 50, 'slider'] ])

        if dialog.exec() == QDialog.Rejected: return

        thres = dialog.getValues(lambda val: int(val) * 2.55)[0]
        _, image = cv2.threshold(self.imageOriginal, thres, 255, fungsi)

        self.imageResult = image
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
