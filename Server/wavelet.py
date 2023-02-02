import numpy as np
import pywt
import cv2


def waveTrans(img, mode, level):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY )
    img =  np.float32(img)
    img /= 255
    # Decomposition
    coeffs = pywt.wavedec2(img, mode, level)
    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    # Reconstruction
    img_H = pywt.waverec2(coeffs_H, mode);
    img_H *= 255
    img_H =  np.uint8(img_H)
    return img_H