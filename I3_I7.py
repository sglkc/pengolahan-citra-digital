from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QMenu
import cv2
from skimage import data, exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import imutils
import numpy as np
from imshow import cv2_imshow

from I1_I2 import I1_I2

class I3_I7(I1_I2):
    def __init__(self):
        super(I3_I7, self).__init__()

        self.detMenu: QMenu
        self.detAstronot: QAction
        self.detPedestrian: QAction
        self.detCars: QAction
        self.detFace: QAction
        self.detWalking: QAction

        self.detMenu.triggered.connect(self.detTrigger)   # type: ignore

    @pyqtSlot(QAction)
    def detTrigger(self, action):
        value = action.text()
        mapped = {
            'Cars Detection': self.__detection,
            'HoG Astronot': self.__hog,
            'HoG Pejalan Kaki': self.__hogCustom,
            'HAAR Face Detection': self.__haar,
            'Walking Detection': self.__walkingDetection,
            'Hough Circle Transform': self.__houghCircle,
        }

        try:
            mapped.get(value)()      # type: ignore
            return True
        except Exception as error:
            if isinstance(error, TypeError):
                return False

            raise error

    @pyqtSlot()
    def __detection(self):
        cam = cv2.VideoCapture("detection/mobil.mp4")
        car_cascade = cv2.CascadeClassifier("detection/haarcascade_car.xml")

        while True:
            _, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

            cv2_imshow('video', frame)

            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

    @pyqtSlot()
    def __hog(self):
        image = data.astronaut()
        _, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True,
                    channel_axis=-1)
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
        img = cv2.imread("detection/pedestrian.jpeg")
        img = imutils.resize(img, width=min(400, img.shape[0]))

        (regions, _) = hog.detectMultiScale(img, winStride=(4,4), padding=(4,4), scale=1.05)

        for (x,y,w,h) in regions:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2_imshow("image", img)
        cv2.waitKey()

    @pyqtSlot()
    def __haar(self):
        face_classifier = cv2.CascadeClassifier('detection/haarcascade_frontalface_default.xml')
        image = cv2.imread("detection/face.jpeg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces.size == 0: # type: ignore
            print("No faces found")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
                cv2_imshow("Face Detection", image)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

    @pyqtSlot()
    def __walkingDetection(self):
        body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
        cap = cv2.VideoCapture('detection/walking.mp4')

        while cap.isOpened():
            _, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
            interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2_imshow('Pedestrians', frame)
                if cv2.waitKey(1) == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

    @pyqtSlot()
    def __houghCircle(self):
        img = cv2.imread('detection/opencv.png', 0)
        img = cv2.medianBlur(img,5)
        cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,20 ,
                                   param1=50 ,param2=50 ,minRadius=5 ,maxRadius=0)

        circles = np.uint16(np.around(circles)) # type: ignore

        for i in circles[0,:]: # type: ignore
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        cv2_imshow('detected circles',cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    __import__('aplikasi').main()
