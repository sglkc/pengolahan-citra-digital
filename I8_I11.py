# pyright: reportAttributeAccessIssue=false
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QAction, QFileDialog
from cv2.typing import MatLike
import numpy as np
import dlib
import cv2
import os
from imshow import cv2_imshow

from I3_I7 import I3_I7

class TooManyFaces(Exception): pass
class NoFaces(Exception): pass

class I8_I11(I3_I7):
    def __init__(self):
        super(I8_I11, self).__init__()

        self.PREDICTOR_PATH = 'detection/shape_predictor_68_face_landmarks.dat'
        self.SCALE_FACTOR = 1
        self.FEATHER_AMOUNT = 11
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))
        self.ALIGN_POINTS = (
            self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        self.OVERLAY_POINTS = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS +
                self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        self.detector = dlib.get_frontal_face_detector()
        self.CASCADE_PATH = './detection/haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        self.detLandmark: QAction
        self.detSwap: QAction
        self.detSwapLive: QAction
        self.detYawn: QAction

    @pyqtSlot(QAction)
    def detTrigger(self, action):
        if super().detTrigger(action): return True

        value = action.text()
        mapped = {
            'Facial Landmark': self.__facialLandmark,
            'Face Swap': self.__faceSwap,
            'Face Swap Live': self.__faceSwapLive,
            'Face Yawn': self.__faceYawn,
        }

        mapped.get(value)() # type: ignore
        return True

    def get_landmarks(self, image, dlibOn=True):
        if dlibOn:
            rects = self.detector(image, 1)

            if len(rects) > 1: return "error"
            if len(rects) == 0: return "error"

            return np.matrix(
                [[p.x, p.y] for p in self.predictor(image, rects[0]).parts()]
            )
        else:
            rects = self.cascade.detectMultiScale(image, 1.3, 5)

            if len(rects) > 1: return "error"
            if len(rects) == 0: return "error"

            x, y, w, h = rects[0]
            rect = dlib.rectangle(x, y, x + w, y + h)
            return np.matrix(
                [[p.x, p.y] for p in self.predictor(image, rect).parts()]
            )

    def annotate_landmarks(self, image, landmarks):
        result = image.copy()

        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(result, pos, 3, color=(255, 0, 255))
            cv2.putText(result, str(idx), pos,
                        fontScale=0.4, color=(0, 0, 255),
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

        return result

    def draw_convex_hull(self, image, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(image, points, color)

    def get_face_mask(self, image: MatLike, landmarks):
        image = np.zeros(image.shape[:2], dtype=np.float64)

        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(image, landmarks[group], color=1)


        ksize = (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT)
        image = np.array([image, image, image]).transpose((1, 2, 0))
        image = (cv2.GaussianBlur(image, ksize, 0) > 0) * 1.0 # type: ignore

        return cv2.GaussianBlur(image, ksize, 0)

    def transformation_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, _, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.vstack([
            np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)),
            np.matrix([0., 0., 1.])
        ])

    def read_im_and_landmarks(self, image, dlibOn=True):
        im = image
        im = cv2.resize(im, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        im = cv2.resize(im, (im.shape[1] * self.SCALE_FACTOR,
                             im.shape[0] * self.SCALE_FACTOR))

        s = self.get_landmarks(im, dlibOn)

        if s == "error": return "error"

        return im, s

    def warp_im(self, im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)

        return output_im

    def correct_colours(self, im1, im2, landmarks1):
        blur_amount = self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[self.LEFT_EYE_POINTS], axis=0) -
                np.mean(landmarks1[self.RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)

        if blur_amount % 2 == 0:
            blur_amount += 1

        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))

    def swappy(self, image1, image2, dlibOn=True):
        get_im_lm1 = self.read_im_and_landmarks(image1, dlibOn)
        get_im_lm2 = self.read_im_and_landmarks(image2, dlibOn)

        if get_im_lm1 == "error" or get_im_lm2 == "error":
            print("Error tidak ada atau terlalu banyak muka")
            return image1

        im1, landmarks1 = get_im_lm1
        im2, landmarks2 = get_im_lm2
        M = self.transformation_from_points(landmarks1[self.ALIGN_POINTS],
                                            landmarks2[self.ALIGN_POINTS])
        mask = self.get_face_mask(im2, landmarks2)
        warped_mask = self.warp_im(mask, M, im1.shape)
        combined_mask = np.max(
            [self.get_face_mask(im1, landmarks1), warped_mask],
            axis=0
        )
        warped_im2 = self.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = self.correct_colours(im1, warped_im2, landmarks1)
        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        cv2.imwrite('swap.jpg', output_im)
        image = cv2.imread('swap.jpg')
        os.remove('swap.jpg')

        return image

    def __facialLandmark(self):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        image = self.imageOriginal
        landmarks = self.get_landmarks(image)

        self.imageResult = self.annotate_landmarks(image, landmarks)
        self.displayImage(2)

    def __faceSwap(self):
        if not hasattr(self, 'imageOriginal'):
            return self.showMessage('Error', 'Citra masih kosong!')

        file, _ = QFileDialog().getOpenFileName(
            filter='Citra (*.png *.jpeg *.jpg *.bmp)')

        image1 = self.imageOriginal
        image2 = cv2.imread(file)
        swap1 = self.swappy(image1, image2)
        swap2 = self.swappy(image2, image1)

        cv2_imshow('1', swap1)
        cv2_imshow('2', swap2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __faceSwapLive(self):
        file, _ = QFileDialog().getOpenFileName(
            filter='Citra (*.png *.jpeg *.jpg *.bmp)')

        capture = cv2.VideoCapture(0)
        filter_image = cv2.imread(file)
        dlibOn = True

        while True:
            _, frame = capture.read()
            frame = cv2.resize(frame, None, fx=0.75, fy=0.75,
                               interpolation=cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)
            cv2_imshow('Face Swap Live', self.swappy(frame, filter_image, dlibOn))

            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

    def __faceYawn(self):
        capture = cv2.VideoCapture(0)
        dlibOn = True

        def top_lip(landmarks):
            top_lip_pts = []

            for i in range(50, 53): top_lip_pts.append(landmarks[i])
            for i in range(61, 64): top_lip_pts.append(landmarks[i])

            # top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
            top_lip_mean = np.mean(top_lip_pts, axis=0)
            return int(top_lip_mean[:, 1])

        def bottom_lip(landmarks):
            bottom_lip_pts = []

            for i in range(65, 68): bottom_lip_pts.append(landmarks[i])
            for i in range(56, 59): bottom_lip_pts.append(landmarks[i])

            # bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
            bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
            return int(bottom_lip_mean[:, 1])

        def mouth_open(image):
            landmarks = self.get_landmarks(image, dlibOn)

            if landmarks == "error": return image, 0

            image_with_landmarks = self.annotate_landmarks(image, landmarks)
            top_lip_center = top_lip(landmarks)
            bottom_lip_center = bottom_lip(landmarks)
            lip_distance = abs(top_lip_center - bottom_lip_center)

            return image_with_landmarks, lip_distance

        yawns = 0
        yawn_status = False

        while True:
            _, frame = capture.read()
            landmarks, lip_distance = mouth_open(frame)
            frame = cv2.flip(frame, 1)
            prev_yawn_status = yawn_status

            if lip_distance > 25:
                yawn_status = True
                cv2.putText(frame, "Subjek sedang menguap", (50, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Count:" + str(yawns + 1), (50, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            else:
                yawn_status = False

            if prev_yawn_status and not yawn_status:
                yawns += 1

            cv2_imshow('Landmarks', landmarks)
            cv2_imshow('Face Swap Live', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    __import__('aplikasi').main()
