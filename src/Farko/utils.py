import math
from scipy.signal import convolve2d
import numpy as np
import cv2
from skimage import morphology

def read_gif(path):
  cap = cv2.VideoCapture(path)
  ret, maskImg = cap.read()

  cap.release()
  return maskImg

def top_hat(image, openingRadius):

  opening_kernel = morphology.disk(openingRadius)

  openedImage=morphology.opening(image, opening_kernel)

  return image-openedImage

def get_gaussian_filter(sgementLength, sigma, orientation, derivative):
    kernelSize = math.ceil(np.sqrt((6 * (math.ceil(sigma)) + 1) ** 2 + sgementLength ** 2))
    gaussianFilter = np.zeros((kernelSize, kernelSize))

    halfKernel = round((kernelSize - 1) / 2);
    row = 0

    for j in range(halfKernel, -halfKernel, -1):
        col = 0

        for i in range(-halfKernel, halfKernel, 1):
            xRotated = i * np.cos(orientation) + j * np.sin(orientation)
            yRotated = -i * np.sin(orientation) + j * np.cos(orientation)
            if abs(xRotated) > 3.5 * math.ceil(sigma):
                gaussianFilter[row, col] = 0
            elif abs(yRotated) > (sgementLength - 1) / 2:
                gaussianFilter[row, col] = 0
            else:
                gaussianFilter[row, col] = -np.exp(-.5 * np.power((xRotated / sigma), 2)) / (np.sqrt(2 * np.pi) * sigma)
                if derivative:
                    gaussianFilter[row, col] *= xRotated / (sigma ** 2)

            col += 1

        row += 1

    ##subtract mean

    return gaussianFilter




def matched_filter(img, segmentLength, sigma, rotationStep, weighted_coefficient):
    matchedFilterBank = np.zeros((img.shape[0], img.shape[1], int(180 / rotationStep) + 1))
    gaussianDerivativeBank = np.zeros((img.shape[0], img.shape[1], int(180 / rotationStep) + 1))

    i = 0
    for theta in np.arange(0, np.pi, rotationStep * np.pi / 180):
        # Get matched Filter
        matchedFilter = get_gaussian_filter(segmentLength, sigma, theta, False)
        mean = np.sum(matchedFilter.reshape(-1)) / np.sum(matchedFilter.reshape(-1) < 0)
        matchedFilter[matchedFilter < 0] = matchedFilter[matchedFilter < 0] - mean

        # Get gaussian Derivative
        gaussianDerivativeFilter = get_gaussian_filter(segmentLength, sigma, theta, True)

        matchedFilterBank[:, :, i] = convolve2d(img, matchedFilter, mode='same')

        gaussianDerivativeBank[:, :, i] = convolve2d(img, gaussianDerivativeFilter, mode='same')
        i += 1

    matchedFilterMax = np.amax(matchedFilterBank, 2)
    gaussianFilterMax = np.amax(gaussianDerivativeBank, 2)
    # cv2_imshow(matchedFilterMax)

    # Ienhanced=(matchedFilterMax-matchedFilterMax.min())/matchedFilterMax.max()
    # Ienhanced*=255
    # Ienhanced=(Ienhanced-Ienhanced.min())/Ienhanced.max()
    # Ienhanced*=255
    # print(Ienhanced.max())

    # cv2_imshow(Ienhanced)

    # #MCET-HHO
    # histogram, bin_edges = np.histogram(Ienhanced, bins=256, range=(0, 255))
    # histogram=histogram/(Ienhanced.shape[0]*Ienhanced.shape[1])
    # histogram[0]=0
    # hho=BaseHHO(30,250,3,histogram)
    # gbest,_=hho.train()

    # gbest[gbest<0]=0
    # gbest[gbest>255]=255
    # th=sorted(gbest)

    # Ienhanced[Ienhanced>th[2]]=0
    # Ienhanced[Ienhanced<th[0]]=0
    # Ienhanced[Ienhanced!=0]=255

    # cv2_imshow(Ienhanced)

    gaussianMaxBlur = cv2.blur(gaussianFilterMax, (31, 31))
    # cv2_imshow(gaussianMaxBlur)

    gaussianMaxBlur = (gaussianMaxBlur - gaussianMaxBlur.min()) / gaussianMaxBlur.max()

    weight = np.mean(matchedFilterMax) * weighted_coefficient

    gaussianFinal = weight * (1 + gaussianMaxBlur)
    # cv2_imshow(gaussianFinal)

    return (matchedFilterMax >= gaussianFinal).astype('uint8') * 255