import math
import numpy as np
from scipy.signal import convolve2d
import cv2
from skimage import morphology

def top_hat(image, openingRadius):

  opening_kernel = morphology.disk(openingRadius)

  openedImage=morphology.opening(image, opening_kernel)

  return image-openedImage

def read_gif(path):
  cap = cv2.VideoCapture(path)
  ret, maskImg = cap.read()

  cap.release()
  return maskImg

def get_matched_filter(sgementLength, sigma, orientation):
    kernelSize = math.ceil(np.sqrt((6 * (math.ceil(sigma)) + 1) ** 2 + sgementLength ** 2))
    matchedFilter = np.zeros((kernelSize, kernelSize))
    matchedFilter2 = np.zeros((kernelSize, kernelSize))

    halfKernel = round((kernelSize - 1) / 2);
    row = 0

    for j in range(halfKernel, -halfKernel, -1):
        col = 0
        for i in range(-halfKernel, halfKernel, 1):
            xRotated = i * np.cos(orientation) + j * np.sin(orientation)
            yRotated = -i * np.sin(orientation) + j * np.cos(orientation)
            if abs(xRotated) > 3.5 * math.ceil(sigma):
                matchedFilter[row, col] = 0
            elif abs(yRotated) > (sgementLength - 1) / 2:
                matchedFilter[row, col] = 0
            else:
                matchedFilter[row, col] = -np.exp(-.5 * np.power((xRotated / sigma), 2)) / (np.sqrt(2 * np.pi) * sigma)

            col += 1
        row += 1

    ##subtract mean
    mean = np.sum(matchedFilter.reshape(-1)) / np.sum(matchedFilter.reshape(-1) < 0)
    matchedFilter[matchedFilter < 0] = matchedFilter[matchedFilter < 0] - mean

    return matchedFilter



def matched_filter_1(img, segmentLength, sigma, rotationStep):
    matchedFilterBank = np.zeros((img.shape[0], img.shape[1], int(180 / rotationStep) + 1))
    i = 0
    for theta in np.arange(0, np.pi, rotationStep * np.pi / 180):
        matchedFilter = get_matched_filter(segmentLength, sigma, theta)
        matchedFilterBank[:, :, i] = convolve2d(img, matchedFilter, mode='same')
        i += 1

    return np.amax(matchedFilterBank, 2)