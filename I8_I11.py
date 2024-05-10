from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from imshow import cv2_imshow

from I3_I7 import I3_I7

class TooManyFaces(Exception): pass
class NoFaces(Exception): pass

class I8_I11(I3_I7):
    def __init__(self):
        super(I8_I11, self).__init__()

        self.detLandmark: QAction
        self.detSwap: QAction
        self.detSwapLive: QAction
        self.detYawn: QAction

        self.detMenu.triggered.connect(self.detTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def detTrigger(self, action):
        if super().detTrigger(action): return True

        value = action.text()
        mapped = {
            'Facial Landmark': self.__haar,
            'Face Swap': self.__hog,
            'Face Swap Live': self.__hogCustom,
            'Face Yawn': self.__detection,
        }

        mapped.get(value)() # type: ignore
        return True

    def __facialLandmark(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        def get_landmarks(im):
            rects = detector(im, 1)
            if len(rects) > 1:
                raise TooManyFaces
            if len(rects) == 0:
                raise NoFaces
            return np.matrix([[p.x, p.y] for p in predictor(im,
            rects[0]).parts()])
            # def annotate_landmarks(im, landmarks):
            # im = im.copy()
            # for idx, point in enumerate(landmarks):
            # pos = (point[0, 0], point[0, 1])
            # cv2.putText(im, str(idx), pos,
            # fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            # fontScale=0.4,
            # color=(0, 0, 255))
            # cv2.circle(im, pos, 3, color=(0, 255, 255))
            # return im
            # image = cv2.imread('jungkook.jfif')
            # landmarks = get_landmarks(image)
            # image_with_landmarks = annotate_landmarks(image, landmarks)
            # cv2.imshow('Result', image_with_landmarks)
            # cv2.imwrite('image_with_landmarks.jpg', image_with_landmarks)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    def __faceSwap(self):
        pass

    def __faceSwapLive(self):
        pass

    def __faceYawn(self):
        pass

if __name__ == '__main__':
    __import__('aplikasi').main()
