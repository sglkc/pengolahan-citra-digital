from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QMenu
import cv2
import numpy as np

from G1 import G1

class I1_I2(G1):
    def __init__(self):
        super(I1_I2, self).__init__()

        self.pwMenu: QMenu
        self.pwTracking: QMenu
        self.pwPicker: QMenu

        self.pwMenu.triggered.connect(self.pwTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def pwTrigger(self, action: QAction):
        value = action.text()
        mapped = {
            'Color Tracking': self.__tracking,
            'Color Picker': self.__picker,
        }

        mapped.get(value)() # type: ignore
        return True

    def __tracking(self):
        cam = cv2.VideoCapture(0)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([66, 98, 100])
            upper_color = np.array([156, 232, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) == 27: break

            cam.release()
            cv2.destroyAllWindows()

    def __picker(self):
        def nothing(_): pass

        cam = cv2.VideoCapture(0)

        cv2.namedWindow("Trackbars")
        cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
        cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
        cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
        cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
        cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            lower_color = np.array([l_h, l_s, l_v])
            upper_color = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) == 27: break

            cam.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    __import__('aplikasi').main()
