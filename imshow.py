import cv2
import os
import sys
from PyQt5 import QtCore
from cv2.typing import MatLike

# Patch imshow karena PyQt5 & OpenCV ga bisa bareng (Linux)
def patch():
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            os.path.dirname(os.path.abspath(cv2.__file__)), "qt", "plugins"
        )

def unpatch():
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
            QtCore.QLibraryInfo.PluginsPath
        )

def cv2_createTrackbar(trackbar: str, window: str, value: int, count: int):
    patch()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar, window, value, count, lambda _: None)
    unpatch()

def cv2_imshow(winname: str, mat: MatLike):
    patch()
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, mat)
    unpatch()

if __name__ == '__main__':
    print("Jalankan dari aplikasi.py!!")
    __import__('aplikasi').main()
