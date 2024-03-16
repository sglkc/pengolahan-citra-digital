import typing
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QDialog, QMenu
import cv2

from A9_C2 import InputDialog
from G1 import G1

class H1_H3(G1):
    def __init__(self):
        super(H1_H3, self).__init__()

        self.sgMenu: QMenu
        self.sgGlobalThres: QMenu
        self.sgLocalAdaptThres: QMenu
        self.sgContour: QMenu

        self.sgGlobalThres.triggered.connect(self.sgGlobalThresTrigger)   # type: ignore
        self.sgLocalAdaptThres.triggered.connect(self.sgLocalAdaptThresTrigger)   # type: ignore
        self.sgContour.triggered.connect(self.contour)   # type: ignore

    @pyqtSlot(QAction)
    def sgGlobalThresTrigger(self, action: QAction):
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
    def sgLocalAdaptThresTrigger(self, action: QAction):
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

    @pyqtSlot()
    def contour(self):
        image = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        polygons = []

        opts = {
            'fontFace': cv2.FONT_HERSHEY_DUPLEX,
            'fontScale': 0.75,
            'color': (0, 75, 255),
        }

        putText = lambda text, coords: cv2.putText(image, text, coords, **opts)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, False)
            polygons.append(polygon)

            epsilon = 0.01 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(polygon)
            coords = int(x + (w / 2)), int(y + (h / 2))

            if len(polygon) == 3:
                putText('Segitiga', coords)
            elif len(polygon) == 4:
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    putText('Persegi', coords)
                else:
                    putText('Persegi Panjang', coords)
            elif len(polygon) == 5:
                putText('Pentagon', coords)
            elif len(polygon) == 6:
                putText('Heksagon', coords)
            elif len(polygon) == 7:
                putText('Heptagon', coords)
            elif len(polygon) == 8:
                putText('Oktagon', coords)
            elif len(polygon) == 9:
                putText('Nonagon', coords)
            elif len(polygon) == 10:
                putText('Bintang', coords)
            else:
                putText('Lingkaran', coords)

        for contour in polygons:
            cv2.drawContours(image, [contour], -1, (0, 127, 255), 2)

        self.imageResult = image
        self.displayImage(2)

if __name__ == '__main__':
    __import__('aplikasi').main()
