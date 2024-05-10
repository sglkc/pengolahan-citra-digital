import cv2
import os
import sys
from PyQt5 import QtCore
from cv2.typing import MatLike

# Patch imshow karena PyQt5 & OpenCV ga bisa bareng (Linux)
def cv2_imshow(winname: str, mat: MatLike):
    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            os.path.dirname(os.path.abspath(cv2.__file__)), "qt", "plugins"
        )

    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, mat)

    if sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
            QtCore.QLibraryInfo.PluginsPath
        )
