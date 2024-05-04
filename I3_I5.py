from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QMenu
import cv2
import numpy as np
from skimage import data, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import imutils

from I1_I2 import I1_I2

class I3_I5(I1_I2):
    def __init__(self):
        super(I3_I5, self).__init__()

        self.detMenu: QMenu
        self.detAstronot: QAction
        self.detPedestrian: QAction
        self.detCars: QAction
        self.detFace: QAction

        self.detMenu.triggered.connect(self.detTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def detTrigger(self, action):
        value = action.text()
        mapped = {
            'Cars Detection': self.__detection,
            'HoG Astronot': self.__hog,
            'HoG Pejalan Kaki': self.__hogCustom,
            'HAAR Face Detection': self.__haar,
        }

        mapped.get(value)() # type: ignore
        return True

    @pyqtSlot()
    def __detection(self):
        cam = cv2.VideoCapture("cars.mp4")
        car_cascade = cv2.CascadeClassifier("cars.xml")

        while True:
            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            cv2.imshow('video', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            cam.release()
            cv2.destroyAllWindows()

    @pyqtSlot()
    def __hog(self):
        image = data.astronaut()
        _, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True,
                    multichannel=True)
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True,
                                       sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap="gray")
        ax1.set_title('Input image')
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) # type: ignore
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap="gray")
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    @pyqtSlot()
    def __hogCustom(self):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(hog.getDefaultPeopleDetector()) # type: ignore
        img = cv2.imread("pedestrian.png")
        img = imutils.resize(img, width=min(400, img.shape[0]))

        (regions, _) = hog.detectMultiScale(img, winStride=(4,4), padding=(4,4), scale=1.05)

        for (x,y,w,h) in regions:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow("image", img)
        cv2.waitKey()

    @pyqtSlot()
    def __haar(self):
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        image = cv2.imread("face.png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces == ():
            print("No faces found")

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
            cv2.imshow("Face Detection", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    __import__('aplikasi').main()
