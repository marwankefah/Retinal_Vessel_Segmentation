import numpy as np
import cv2
from scipy.signal import convolve2d
from skimage import morphology
from .utils import *


def farko_method_2(image, maskImg):
    imageGreen = image[:, :, 1]
    # Create mask if not available
    # mask=image[:,:,1]>=20
    # mask=mask.astype(int)
    # print(data)

    # maskImg=read_gif(mask[1])

    maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY) / 255

    erosionKernel = morphology.disk(3)

    erodedMask = cv2.erode(maskImg, erosionKernel, iterations=2)

    # cv2_imshow(imageGreen)

    # histogram Eq adaptive
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clacheImg = clahe.apply(imageGreen)
    # cv2_imshow(clacheImg)

    # Remove black ring
    replaceBlackRing = imageGreen * erodedMask
    meanImage = np.mean(replaceBlackRing)
    replaceBlackRing[replaceBlackRing == 0] = meanImage

    # Top hat
    imgComplement = 255 - replaceBlackRing
    # cv2_imshow(imgComplement)

    topHatedImg = top_hat(imgComplement, 10) * erodedMask
    # cv2_imshow(topHatedImg)
    topHatedImg = topHatedImg.astype('uint8')

    # Otsu
    ret, thresh1 = cv2.threshold(topHatedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh1 = thresh1 * erodedMask.astype('uint8')

    # cv2_imshow(thresh1)

    # Cleaning small objects less than 100
    cleaned = morphology.remove_small_objects(thresh1.astype(bool), min_size=100, connectivity=8)

    cleanedImg = cleaned.astype('uint8') * 255

    # cv2_imshow(cleanedImg)

    # is it close the min pixel(backgorund) or to the max pixel(vessel)
    filteredImage = np.abs(imgComplement - imgComplement.max()) < np.abs(imgComplement - meanImage)
    filteredImage = filteredImage.astype('uint8') * cleanedImg
    # cv2_imshow(filteredImage)

    medianCleaned = cv2.medianBlur(cleanedImg, 3)
    # cv2_imshow(medianCleaned)

    filteredImage = cv2.bitwise_or(medianCleaned, filteredImage)
    # filteredImage
    # cv2_imshow(filteredImage)

    # finding thin vessels
    thinVessels = matched_filter(clacheImg, 4, 1, 7, 2.3)

    # cv2_imshow(thinVessels)

    closingKernel = np.ones((3, 3))
    closedThinVessels = morphology.binary_closing(thinVessels, closingKernel)
    # cv2_imshow(closedThinVessels.astype(int)*255)

    cleanedThin = morphology.remove_small_objects(closedThinVessels.astype(bool), min_size=30, connectivity=8)

    # cleanedThin=cleanedThin.astype('uint8')*255
    # cv2_imshow(cleanedThin)

    finalKernael = np.ones((3, 3))
    finalKernael[1, 1] = 0
    finalConv = convolve2d(cleanedThin.astype('uint8'), finalKernael, mode='same')
    finalImage = (finalConv > 0).astype('uint8') * filteredImage.astype('uint8') * erodedMask

    # erodedMask=cv2.erode(erodedMask, erosionKernel, iterations=1)
    finalImage = cv2.bitwise_or(cleanedThin.astype('uint8') * 255, finalImage.astype('uint8')) * erodedMask

    # finalImage=cv2.bitwise_or(cleanedThin.astype('uint8')*255,filteredImage.astype('uint8'))*erodedMask

    # cv2_imshow(finalImage.astype(int))
    return finalImage

def farko_method(data):
    image = cv2.imread(data[0])
    imageGreen = image[:, :, 1]
    # Create mask if not available
    # mask=image[:,:,1]>=20
    # mask=mask.astype(int)
    print(data)

    maskImg = read_gif(data[1])
    maskImg = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY) / 255

    erosionKernel = morphology.disk(3)

    erodedMask = cv2.erode(maskImg, erosionKernel, iterations=2)

    # cv2_imshow(imageGreen)

    # histogram Eq adaptive
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clacheImg = clahe.apply(imageGreen)
    # cv2_imshow(clacheImg)

    # Remove black ring
    replaceBlackRing = imageGreen * erodedMask
    meanImage = np.mean(replaceBlackRing)
    replaceBlackRing[replaceBlackRing == 0] = meanImage

    # Top hat
    imgComplement = 255 - replaceBlackRing
    # cv2_imshow(imgComplement)

    topHatedImg = top_hat(imgComplement, 10) * erodedMask
    # cv2_imshow(topHatedImg)
    topHatedImg = topHatedImg.astype('uint8')

    # Otsu
    ret, thresh1 = cv2.threshold(topHatedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh1 = thresh1 * erodedMask.astype('uint8')

    # cv2_imshow(thresh1)

    # Cleaning small objects less than 100
    cleaned = morphology.remove_small_objects(thresh1.astype(bool), min_size=100, connectivity=8)

    cleanedImg = cleaned.astype('uint8') * 255

    # cv2_imshow(cleanedImg)

    # is it close the min pixel(backgorund) or to the max pixel(vessel)
    filteredImage = np.abs(imgComplement - imgComplement.max()) < np.abs(imgComplement - meanImage)
    filteredImage = filteredImage.astype('uint8') * cleanedImg
    # cv2_imshow(filteredImage)

    medianCleaned = cv2.medianBlur(cleanedImg, 3)
    # cv2_imshow(medianCleaned)

    filteredImage = cv2.bitwise_or(medianCleaned, filteredImage)
    # filteredImage
    # cv2_imshow(filteredImage)

    # finding thin vessels
    thinVessels = matched_filter(clacheImg, 4, 1, 7, 2.3)

    # cv2_imshow(thinVessels)

    closingKernel = np.ones((3, 3))
    closedThinVessels = morphology.binary_closing(thinVessels, closingKernel)
    # cv2_imshow(closedThinVessels.astype(int)*255)

    cleanedThin = morphology.remove_small_objects(closedThinVessels.astype(bool), min_size=30, connectivity=8)

    # cleanedThin=cleanedThin.astype('uint8')*255
    # cv2_imshow(cleanedThin)

    finalKernael = np.ones((3, 3))
    finalKernael[1, 1] = 0
    finalConv = convolve2d(cleanedThin.astype('uint8'), finalKernael, mode='same')
    finalImage = (finalConv > 0).astype('uint8') * filteredImage.astype('uint8') * erodedMask

    # erodedMask=cv2.erode(erodedMask, erosionKernel, iterations=1)
    finalImage = cv2.bitwise_or(cleanedThin.astype('uint8') * 255, finalImage.astype('uint8')) * erodedMask

    # finalImage=cv2.bitwise_or(cleanedThin.astype('uint8')*255,filteredImage.astype('uint8'))*erodedMask

    # cv2_imshow(finalImage.astype(int))
    return finalImage



def farko_method_noMask(image):
    imageGreen = image[:, :, 1]
    # Create mask if not available
    # mask=image[:,:,1]>=20
    # mask=mask.astype(int)
    # print(data)

    # maskImg=read_gif(mask[1])
    # maskImg=cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)/255

    # cv2_imshow(imageGreen)

    # histogram Eq adaptive
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clacheImg = clahe.apply(imageGreen)
    # cv2_imshow(clacheImg)

    # Remove black ring
    replaceBlackRing = imageGreen
    meanImage = np.mean(replaceBlackRing)
    replaceBlackRing[replaceBlackRing == 0] = meanImage

    # Top hat
    imgComplement = 255 - replaceBlackRing
    # cv2_imshow(imgComplement)

    topHatedImg = top_hat(imgComplement, 10)
    # cv2_imshow(topHatedImg)
    topHatedImg = topHatedImg.astype('uint8')

    # Otsu
    ret, thresh1 = cv2.threshold(topHatedImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh1 = thresh1

    # cv2_imshow(thresh1)

    # Cleaning small objects less than 100
    cleaned = morphology.remove_small_objects(thresh1.astype(bool), min_size=100, connectivity=8)

    cleanedImg = cleaned.astype('uint8') * 255

    # cv2_imshow(cleanedImg)

    # is it close the min pixel(backgorund) or to the max pixel(vessel)
    filteredImage = np.abs(imgComplement - imgComplement.max()) < np.abs(imgComplement - meanImage)
    filteredImage = filteredImage.astype('uint8') * cleanedImg
    # cv2_imshow(filteredImage)

    medianCleaned = cv2.medianBlur(cleanedImg, 3)
    # cv2_imshow(medianCleaned)

    filteredImage = cv2.bitwise_or(medianCleaned, filteredImage)
    # filteredImage
    # cv2_imshow(filteredImage)

    # finding thin vessels
    thinVessels = matched_filter(clacheImg, 4, 1, 7, 2.3)

    # cv2_imshow(thinVessels)

    closingKernel = np.ones((3, 3))
    closedThinVessels = morphology.binary_closing(thinVessels, closingKernel)
    # cv2_imshow(closedThinVessels.astype(int)*255)

    cleanedThin = morphology.remove_small_objects(closedThinVessels.astype(bool), min_size=30, connectivity=8)

    # cleanedThin=cleanedThin.astype('uint8')*255
    # cv2_imshow(cleanedThin)

    finalKernael = np.ones((3, 3))
    finalKernael[1, 1] = 0
    finalConv = convolve2d(cleanedThin.astype('uint8'), finalKernael, mode='same')
    finalImage = (finalConv > 0).astype('uint8') * filteredImage.astype('uint8')

    # erodedMask=cv2.erode(erodedMask, erosionKernel, iterations=1)
    finalImage = cv2.bitwise_or(cleanedThin.astype('uint8') * 255, finalImage.astype('uint8'))

    # finalImage=cv2.bitwise_or(cleanedThin.astype('uint8')*255,filteredImage.astype('uint8'))*erodedMask

    # cv2_imshow(finalImage.astype(int))
    return finalImage