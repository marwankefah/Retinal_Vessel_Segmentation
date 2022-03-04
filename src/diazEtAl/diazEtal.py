import numpy as np
import cv2
from skimage import morphology
from BaseHHo import *
from HomomorphicFilter import *

def diazEtAl(image, mask):
    kernel = np.ones((5, 5))
    maskGrayEroded = cv2.erode(mask, kernel, iterations=2)

    replaceBlackRing = image * maskGrayEroded

    replaceBlackRing[replaceBlackRing == 0] = np.mean(replaceBlackRing)

    closing_kernel = np.ones((9, 9))
    closedImage = morphology.closing(replaceBlackRing, closing_kernel)
    bottomHatImage = closedImage - replaceBlackRing

    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    img_filtered = homo_filter.filter(I=bottomHatImage, filter_params=2)
    medianImage = cv2.medianBlur(img_filtered, 3)

    bottomHatImage = medianImage
    # cv2_imshow(bottomHatImage)

    print(bottomHatImage.max(), bottomHatImage.min())

    # lateral
    lateralKernel = [[-0.025, -0.025, -0.025, -0.025, -0.025],
                     [-0.025, -0.075, -0.075, -0.075, -0.025],
                     [-0.025, -0.075, 1, -0.075, -0.025],
                     [-0.025, -0.075, -0.075, -0.075, -0.025],
                     [-0.025, -0.025, -0.025, -0.025, -0.025]]

    Ienhanced = bottomHatImage + convolve2d(bottomHatImage, lateralKernel, mode='same')

    Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
    Ienhanced *= 255
    Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
    Ienhanced *= 255

    # cv2_imshow(Ienhanced)

    # MCET-HHO
    histogram, bin_edges = np.histogram(Ienhanced, bins=256, range=(0, 255))
    histogram = histogram / (Ienhanced.shape[0] * Ienhanced.shape[1])
    histogram[0] = 0
    hho = BaseHHO(30, 250, 3, histogram)
    gbest, _ = hho.train()

    # binarizing based on best threshold
    gbest[gbest < 0] = 0
    gbest[gbest > 255] = 255
    th = sorted(gbest)
    print(th)
    Ienhanced[Ienhanced > th[2]] = 0
    Ienhanced[Ienhanced < th[0]] = 0
    Ienhanced[Ienhanced != 0] = 255

    kernel = np.ones((5, 5), np.uint8)
    Ienhanced = morphology.closing(Ienhanced, kernel)

    # cv2_imshow(Ienhanced)
    cleaned = morphology.remove_small_objects(Ienhanced.astype(bool), min_size=30, connectivity=8).astype(int)

    cleaned[cleaned == True] = 255

    return cleaned


def diazEtAl_noMask(image):
    kernel = np.ones((5, 5))

    replaceBlackRing = image

    replaceBlackRing[replaceBlackRing == 0] = np.mean(replaceBlackRing)

    closing_kernel = np.ones((9, 9))
    closedImage = morphology.closing(replaceBlackRing, closing_kernel)
    bottomHatImage = closedImage - replaceBlackRing

    homo_filter = HomomorphicFilter(a=0.75, b=1.25)
    img_filtered = homo_filter.filter(I=bottomHatImage, filter_params=2)
    medianImage = cv2.medianBlur(img_filtered, 3)

    bottomHatImage = medianImage
    # cv2_imshow(bottomHatImage)

    print(bottomHatImage.max(), bottomHatImage.min())

    # lateral
    lateralKernel = [[-0.025, -0.025, -0.025, -0.025, -0.025],
                     [-0.025, -0.075, -0.075, -0.075, -0.025],
                     [-0.025, -0.075, 1, -0.075, -0.025],
                     [-0.025, -0.075, -0.075, -0.075, -0.025],
                     [-0.025, -0.025, -0.025, -0.025, -0.025]]

    Ienhanced = bottomHatImage + convolve2d(bottomHatImage, lateralKernel, mode='same')

    Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
    Ienhanced *= 255
    Ienhanced = (Ienhanced - Ienhanced.min()) / Ienhanced.max()
    Ienhanced *= 255

    # cv2_imshow(Ienhanced)

    # MCET-HHO
    histogram, bin_edges = np.histogram(Ienhanced, bins=256, range=(0, 255))
    histogram = histogram / (Ienhanced.shape[0] * Ienhanced.shape[1])
    histogram[0] = 0
    hho = BaseHHO(30, 250, 3, histogram)
    gbest, _ = hho.train()

    # binarizing based on best threshold
    gbest[gbest < 0] = 0
    gbest[gbest > 255] = 255
    th = sorted(gbest)
    print(th)
    Ienhanced[Ienhanced > th[2]] = 0
    Ienhanced[Ienhanced < th[0]] = 0
    Ienhanced[Ienhanced != 0] = 255

    kernel = np.ones((5, 5), np.uint8)
    Ienhanced = morphology.closing(Ienhanced, kernel)

    # cv2_imshow(Ienhanced)
    cleaned = morphology.remove_small_objects(Ienhanced.astype(bool), min_size=30, connectivity=8).astype(int)

    cleaned[cleaned == True] = 255

    return cleaned
