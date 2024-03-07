from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import *   # type: ignore
import cv2
from matplotlib import pyplot as plt
import numpy as np

from D1_D6 import D1_D6

class E1_E2(D1_D6):
    def __init__(self):
        super(E1_E2, self).__init__()

    @pyqtSlot(QAction)
    def tfTrigger(self, action: QAction):
        if super().tfTrigger(action): return True

        menuText = action.text()
        mapped = {
            'Low Pass': self.__dftLowPass,
            'High Pass': self.__dftHighPass,
        }

        mapped.get(menuText)()      # type: ignore
        return False

    def __dftLowPass(self):
        x = np.arange(256)
        y = np.sin((2 * np.pi) * (x / 3))

        y += max(y)

        img = np.array(
            [
                [
                    y[j] * 127 for j in range(256)
                ] for _ in range(256)
            ],
            dtype=np.uint8)

        # plt.imshow(img)

        img = cv2.cvtColor(self.imageOriginal, cv2.COLOR_BGR2GRAY)      # type: ignore
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)    # type: ignore
        dft_shift = np.fft.fftshift(dft)

        spectrum = 20 * np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])))
        rows, cols = img.shape[:2]
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
        f_ishift = np.fft.fftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        self.imageResult = cv2.cvtColor(
            cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
            cv2.COLOR_GRAY2BGR)

        self.displayImage(2)

        fig = plt.figure(figsize=(8, 8))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_axis_off()
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Citra Masukan')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_axis_off()
        ax2.imshow(spectrum, cmap='gray')
        ax2.set_title('FFT Citra')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_axis_off()
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.set_title('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_axis_off()
        ax4.imshow(img_back, cmap='gray')
        ax4.set_title('Inverse Fourier')

        plt.tight_layout()
        plt.show()

    def __dftHighPass(self):
        pass

if __name__ == '__main__':
    __import__('aplikasi').main()
