"""
Created on Feb 23, 2015

@author: Jorge
"""

"""
Install ImageJ

conda install SimpleITK
conda install ipython?
conda install matplotlib
conda install binstar
pip install pynrrd
conda install -c https://conda.binstar.org/asmeurer ipython-notebook
conda install pyqt
conda install opencv
"""

import sys
import nrrd
import os
import math
import time
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

import cv2
import cv2.cv

from skimage import color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max

from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

from skimage import filter


def plot(image):
    """ Use Matplotlib to display a simpleITK image (among others)
    :param image: image to display (ex: sitkImage)
    """
    array = sitk.GetArrayFromImage(image)
    plota(array)


def plota(array):
    """ Use Matplotlib to display a numpy array in grayscale
    :param array:
    :return:
    """
    f = plt.figure()
    plt.imshow(array, cmap=cm.Greys_r)
    refresh()
    return f.number


def arr(sitkImage):
    """ Get a numpy array from a simpleItk image
    :param imageArray:
    :return:
    """
    return sitk.GetArrayFromImage(sitkImage)


def cl(id=0):
    """ Close all Matplotlib windows. If an id is passed, close just that window
    :param id: Number of window to close (0 for all the windows)
    :return:
    """
    if id == 0:
        plt.close('all')
    else:
        plt.close(id)


def refresh(interval=0.1):
    """ Refresh the Matplotlib window so that it's not freezed and we can use it (to workaround the problem with PyDev)
    :param interval: number of seconds that we can use the Matplotlib window (we cannot use Python in the meantime!)
    """
    plt.pause(interval)


def generateSeed(image, seedIndex):
    """ Generate a binary labelmap image with a seed starting in an index. It will build a small circular seed
    :param image: original image
    :param seedIndex: index of the seed
    :return: binary labelmap image with the same size as the original one
    """
    imSize = image.GetSize()
    seedImage = sitk.Image(imSize, sitk.sitkUInt8)
    seedImage.CopyInformation(image)
    seedImage[seedIndex] = 1
    seedIndex = sitk.BinaryDilate(seedImage, 3)
    return seedIndex


def applyLevelset(image, labelmap, numberOfIterations, curvatureScaling, propagationScaling, lowerThreshold,
                  upperThreshold, maximumRMSError):
    """ simpleITK levelset segmentation.
    :param image: original image
    :param labelmap: labelmap that contains a seed to grow the levelset
    :param numberOfIterations: levelset number of iterations (algorithm parameter)
    :param curvatureScaling: levelset curvature scaling (algorithm parameter)
    :param propagationScaling: levelset propagation scaling (algorithm parameter)
    :param lowerThreshold: levelset lower threshold (algorithm parameter)
    :param upperThreshold: levelset upper threshold (algorithm parameter)
    :param maximumRMSError: levelset maximun RMS error (algorithm parameter)
    :return: labelmap as a simpleITK image with the same size as the original one and value=1 in the grown region
    """
    init_ls = sitk.SignedMaurerDistanceMap(labelmap, insideIsPositive=True, useImageSpacing=True)
    lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
    lsFilter.SetLowerThreshold(lowerThreshold)
    lsFilter.SetUpperThreshold(upperThreshold)
    # Additional parameters
    lsFilter.SetMaximumRMSError(maximumRMSError)
    lsFilter.SetNumberOfIterations(numberOfIterations)
    lsFilter.SetCurvatureScaling(curvatureScaling)
    lsFilter.SetPropagationScaling(propagationScaling)
    lsFilter.ReverseExpansionDirectionOn()

    im = lsFilter.Execute(init_ls, sitk.Cast(image, sitk.sitkFloat32))
    return im > 0



def adaptIntensities(imageArray, thresholdMin=-50, thresholdMax=150):
    """ Adapt the intensities of an image numpy array and convert to 255 gray levels (uint8)
    :param imageArray: numpy array
    :param thresholdMin: values below this level will be set to this value
    :param thresholdMax: values beneath this level will be set to this value
    :return: 256 gray levels thresholded image. The type of the array will be uint8
    """
    imageArray[imageArray <= thresholdMin] = thresholdMin
    imageArray[imageArray >= thresholdMax] = thresholdMax
    imageArray = 254*(imageArray.astype(np.float32)-thresholdMin)/(thresholdMax-thresholdMin)
    # plota(imageArray)
    # imageArray[sitk.RescaleIntensity(sitk.sitkUInt8)]
    imageArray = imageArray.astype(np.uint8)

    return imageArray


def detectAortaSitk(imageArray, boundingBoxMin, boundingBoxMax, seedAorta):
    """ Detect the aorta in a 2D axial image (as a numpy array)
    :param imageArray: numpy uint8 array
    :param boundingBoxMin: top-left XY coordinate for the box where the aorta will be searched
    :param boundingBoxMax: bottom-right XY coordinate for the box where the aorta will be searched
    :param seedAorta: XY index that is known to be inside the aorta. This index will be used to select the best
            candidate circle
    :return: list with 3 elems that represent the circle that contains the aorta (or None if no circle found):
        - X coordinate of the circle center
        - Y coordinate of the circle center
        - Radious of the circle
    """
    # Crop the imageArray (2D)
    img = imageArray[boundingBoxMin[Y]:boundingBoxMax[Y], boundingBoxMin[X]:boundingBoxMax[X]]
    img = cv2.medianBlur(img, 5)

    f = sitk.BinaryMorphologicalClosingImageFilter()
    f.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
    f.SetKernelRadius(2)
    i2 = img >= 80
    i2 = i2.astype('uint8')
    i2 = sitk.GetImageFromArray(i2)
    filteredImage = f.Execute(i2)
    img = img * sitk.GetArrayFromImage(filteredImage)

    # Load picture and detect edges
    image = img_as_ubyte(img)
    edges = filter.canny(image, sigma=1, low_threshold=80, high_threshold=170)
    # plota(edges)

    # Detect two radii
    hough_radii = np.arange(10, aortaMaxRadius)
    hough_res = hough_circle(edges, hough_radii)

    centers = []
    accums = []
    radii = []

    for radius, h in zip(hough_radii, hough_res):
        # For each radius, extract 3 circles
        npeaks = 3
        peaks = peak_local_max(h, num_peaks=npeaks)
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * npeaks)

    # Remove the circles that are outside our seed
    filteredCenters = []
    filteredAccums = []
    filteredRadii = []
    for i in range(len(centers)):
        # Calculate the distance from the center to the seed
        distance = math.sqrt(pow(centers[i][1]+boundingBoxMin[X]-seedAorta[X], 2) + pow(centers[i][0]+boundingBoxMin[Y]-seedAorta[Y], 2))

        if radii[i] >= distance:
            # The circle contains the seed. Mark as valid
            filteredCenters.append(centers[i])
            filteredAccums.append(accums[i])
            filteredRadii.append(radii[i])

    image2 = color.gray2rgb(image)

    # Get the best circle among the selected ones
    idx = np.argmax(filteredAccums)
    radius = filteredRadii[idx]
    cx, cy = circle_perimeter(filteredCenters[idx][1], filteredCenters[idx][0], radius)
    image2[cy, cx] = (220, 20, 20)


    # for idx in np.argsort(filteredAccums)[::-1][:5]:
    #     center_x, center_y = filteredCenters[idx]
    #     radius = filteredRadii[idx]
    #     cx, cy = circle_perimeter(center_y, center_x, radius)
    #     image2[cy, cx] = (220, 20, 20)
    # plota(image2)

    # Return X,Y,Radius
    return [filteredCenters[idx][1], filteredCenters[idx][0], filteredRadii[idx]]


def drawCircles(imageArray, circles):
    """ Draw a list of circles using opencv in a numpy array
    :param imageArray: numpy array
    """
    for circ in circles:
        drawCircle(imageArray, circ)


def drawCircle(imageArray, circle):
    """ Draw a circle in in a numpy array using opencv. The circle will be in the format [X,Y,Radious]
    :param imageArray: numpy array
    """
    cv2.circle(imageArray, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)


def getSkeletonIntersection(array, seed):
    """ Given a numpy array that represents a structure skeleton (values 0 and 1), try to detect the intersection point
    that is closest to the [X,Y] index stored in "seed"
    :param array: numpy 2D array with the skeleton
    :param seed: [X,Y] values of the point that should be the closest one to the candidate intersections.
    :return: tuple with X,Y coordinates of the intersection point (or None if not found a suitable candidate)
    """
    # Get the number of neighbours for each pixel
    N = np.zeros(array.shape, dtype=int)
    N[1:-1, 1:-1] += (array[:-2,  :-2]  + array[:-2, 1:-1] + array[:-2, 2:] +
                      array[1:-1, :-2] + array[1:-1, 2:] + array[2: , :-2] +
                      array[2:, 1:-1]  + array[2:, 2:])

    # Keep just the ones that are part of the array
    N = N * array

    # Filter those that have a high connectivity (at least N=4)
    points = np.where(N > 3)

    # Return the closest point to the seed
    # TODO: could "seed" just be replaced by the top point?
    index = -1
    minDistance = sys.maxint
    for i in range(len(points[0])):
        distance = abs(seed[1]-points[0][i]) + abs(seed[0]-points[1][i])
        if distance < minDistance:
            minDistance = distance
            index = i
    if index != -1:
        return points[1][index], points[0][index]
    # No suitable candidate found
    return None


def executePipeline(imageFullPath, boundingBoxMin, boundingBoxMax, seedAorta, seedPA, displayResults=True):
    """ Execute all the pipeline to calculate the PA:A ratio.

    These are the steps that are followed:
    1) Read the 3D image in the path "imageFullPath"
    2) Adjust the gray levels for a better segmentation of Aorta and Pulmonary Artery (PA)
    3) Detect the aorta as a circle using scikit
    4) Grow a level set in the PA once that we have set a boundary for the aorta (otherwise they could grow as the same
    structure because there is no space between them)
    5) Get the PA skeleton
    6) Measure the ratio of the PA (at the moment just 15 pixels beneath the intersection point in the skeleton). The
    ratio will be measured in an horizontal line at this point.
    7) Calculate the PA:A ratio
    :param imageFullPath: full path to the original 3D volume
    :param boundingBoxMin: top-left XYZ coordinate of the 2D slice that we will choose as the working slice
    :param boundingBoxMax: bottom-right XYZ coordinate of the 2D slice we will choose as the working slice
    :param seedAorta: XYZ index of a point that is known to be inside the aorta
    :param seedPA: XYZ index of a point that is known to be inside the PA
    :return: tuple with the following information:
        - If everything goes good:
            - Case id
            - PAA
            - PA diameter
            - Y coordinate where the PA diameter has been measured
            - Aorta diameter
            - Aorta center ([X,Y] coordinates)
        - Otherwise:
            - Case id
            - Error message
    """
    # TODO: get seedAorta and seedPA automatically from the bounding box

    # factor = 1.5
    numberOfIterations = 1000
    maximumRMSError = 0.02
    
    curvatureScaling = 1
    propagationScaling = 1

    # Set thresholds for a bigger/smaller segmented area
    # stats = getImageStats(imageArray, labelmap)
    # lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
    # upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
    lowerThreshold = 60
    upperThreshold = 170


    originalImage = sitk.ReadImage(imageFullPath)
    # Select just the slice that we are going to work in
    slice = boundingBoxMin[Z]

    originalImageSlice = originalImage[:, :, slice]
    imageArray = adaptIntensities(sitk.GetArrayFromImage(originalImageSlice))
    # circ2 = detectAorta(imageArray, boundingBoxMin, boundingBoxMax, seedAorta)
    circ = detectAortaSitk(imageArray, boundingBoxMin, boundingBoxMax, seedAorta)

    if circ is None:
        print("Aorta not detected in case ", imageFullPath)
        return

    # Draw the circle in the original image
    imageArrayWithAorta = imageArray
    circ[0] = circ[0]+boundingBoxMin[X]
    circ[1] = circ[1]+boundingBoxMin[Y]
    drawCircle(imageArrayWithAorta, circ)
    # plota(imageArrayWithAorta)

    # Convert the arrays in itk images to work with simpleitk filters
    image = sitk.GetImageFromArray(imageArrayWithAorta)
    # Generate seed in the labelmap image (get the first two dimensions in the seed because we are now working in 2D)
    labelmapSeed = generateSeed(image, seedPA[X:Z])

    labelmap = applyLevelset(image, labelmapSeed, numberOfIterations, curvatureScaling, propagationScaling,
                             lowerThreshold, upperThreshold, maximumRMSError)

    # Skeleton
    paDilated = sitk.BinaryDilate(labelmap > 0)
    skeleton = sitk.BinaryThinning(paDilated)

    skeletonArray = sitk.GetArrayFromImage(skeleton)
    intersection = getSkeletonIntersection(skeletonArray, seedPA)

    # Draw the skeleton and the intersection point
    labelmapArray = sitk.GetArrayFromImage(labelmap)
    labelmapArray[skeletonArray == 1] = 2    # (label 2 to display results)
    cv2.circle(labelmapArray, (intersection[0], intersection[1]), 5, (4, 0, 0), -3)

    # Draw the line that will be used to measure the PA radious
    paLevel = intersection[1]-15
    cv2.line(labelmapArray, (0, paLevel), (imageArray.shape[1], paLevel), (5,0,0))

    # Draw the aorta circle just to display results (label 3)
    cv2.circle(labelmapArray, (circ[0], circ[1]), circ[2], (3, 0, 0), 1)

    labelmap = sitk.GetImageFromArray(labelmapArray)

    # f = sitk.ScalarToRGBColormapImageFilter()
    # f.SetColormap(f.Blue)
    # i = f.Execute(skeleton)
    # sitk.Show(i)

    # output2 = sitk.LabelOverlay(image, labelmap)
    # plot(output2)

    caseId = os.path.basename(imageFullPath).replace(".nhdr", "")
    if displayResults:
        # Show the final results
        output = sitk.LabelOverlay(image, labelmap)
        plot(output)
        plt.title('Results ' + caseId)
        refresh()

    # Calculate the PA radius
    pa = sitk.GetArrayFromImage(paDilated)
    # Get the minimun index for the PA (center of the aorta+radious)
    idx = circ[0] + circ[2]
    nz = np.nonzero(pa[paLevel, idx:])
    minPA = np.min(nz) + idx
    maxPA = np.max(nz) + idx
    paDiameter = maxPA-minPA
    aortaDiameter = circ[2] * 2
    ratio = paDiameter / float(aortaDiameter)

    if displayResults:
        print("PA diameter: {0}".format(paDiameter))
        print("Aorta diameter: {0}".format(aortaDiameter))
        print("PA:A ratio: {0}".format(ratio))

    return caseId, ratio, paDiameter, paLevel, aortaDiameter, [circ[0], circ[1]]

def save(imageName):
    plt.savefig(os.path.join(homepath, 'tempdata/results', os.path.basename(imageName)) + '-' +
                time.strftime("%Y-%m-%d_%H-%M") + '.png')

#############################################################################################

# FIXED PARAMETERS
X = 0
Y = 1
Z = 2

np_X = 2
np_Y = 1
np_Z = 0

homepath = os.path.expanduser("~")

aortaMaxRadius = 35


if __name__ == '__main__':
    print("main")

else:
    print("KK")

#
# Shitty
imageName = homepath + '/tempdata/023960002463_INSP_B35f_L1_ECLIPSE.nhdr'
slice = 214
# XYZ
seedAorta = [250,210, slice]
seedPA = [295, 222, slice]
boundingBoxMin = [228,193, slice]
boundingBoxMax = [311,251, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)


# So so ==> OK aorta (fail skeleton intersection)
imageName = homepath + '/tempdata/10002K_INSP_STD_BWH_COPD.nhdr'
slice = 391
# XYZ
seedAorta = [249, 175, slice]
seedPA = [294,186, slice]
boundingBoxMin = [178, 141, slice]
boundingBoxMax = [361, 287, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)

# WRONG ==> ok
imageName = homepath + '/tempdata/10005Q_INSP_STD_NJC_COPD.nhdr'
slice = 325
# XYZ
seedAorta = [225,200, slice]
seedPA = [268, 203, slice]
boundingBoxMin = [201,170, slice]
boundingBoxMax = [324,273, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)

# OK  ==> aorta is kind of too big
imageName = homepath + '/tempdata/10004O_INSP_STD_BWH_COPD.nhdr'
slice = 415
# XYZ
seedAorta = [223,227, slice]
seedPA = [272, 246, slice]
boundingBoxMin = [187,198, slice]
boundingBoxMax = [298,296, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)

# OK
imageName = homepath + '/tempdata/10006S_INSP_STD_BWH_COPD.nhdr'
slice = 399
# XYZ
seedAorta = [228,202, slice]
seedPA = [272, 210, slice]
boundingBoxMin = [184,171, slice]
boundingBoxMax = [310,268, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)

# Ok
imageName = homepath + '/tempdata/12257B_INSP_STD_UIA_COPD.nhdr'
slice = 442
# XYZ
seedAorta = [238,220, slice]
seedPA = [287, 248, slice]
boundingBoxMin = [216,199, slice]
boundingBoxMax = [312,273, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)

# WRONG ==> OK
imageName = homepath + '/tempdata/10015T_INSP_STD_BWH_COPD.nhdr'
slice = 458
# XYZ
seedAorta = [248,200, slice]
seedPA = [291, 212, slice]
boundingBoxMin = [214,169, slice]
boundingBoxMax = [319,250, slice]
executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedAorta, seedPA)
save(imageName)



##############################################
# LEGACY CODE
##############################################
# def readImages(imageName, labelmapName, dataFolder, slice):
#     """Read a slice from the original imageArray and the label map.
#     It returns a tuple like:
#     - OriginalImage
#     - OriginalLabelmap
#     - ImageArray
#     - LabelmapArray
#     """
#     inputImagePath = dataFolder + imageName
#     imageArray = sitk.ReadImage(inputImagePath)
#     imageArray = imageArray[:,:,slice]
#
#     seedImagePath = dataFolder + labelmapName
#     labelmapImage = sitk.ReadImage(seedImagePath)
#     labelmapImage = labelmapImage[:,:,slice]
#
#     return imageArray, labelmapImage, sitk.GetArrayFromImage(imageArray), sitk.GetArrayFromImage(labelmapImage)


# def extractAxialSlice(filePath, sliceNumber):
#     ct_data, imageInfo = nrrd.read(filePath)
#
#     # create a new ct file with that slice number
#     ct_data_slice = ct_data[:,:,sliceNumber:sliceNumber+1]
#     outputFilePath = '%s_%i.nrrd' % (os.path.splitext(filePath)[0], sliceNumber)
#     nrrd.write(outputFilePath, ct_data_slice)


# def preprocessImage(img):
#     """Apply a filter to preprocess the imageArray (median, gausian, etc.).
#     It returns the filtered imageArray."""
#     filter = sitk.MedianImageFilter()
#     img = filter.Execute(img)
#     return img

# def getImageStats(imageArray, labelmap):
#     """Get some standard stats for the imageArray"""
#     stats = sitk.LabelStatisticsImageFilter()
#     stats.Execute(imageArray, labelmap)
#     return stats


# def displayResults(originalImage, labelmap, resultImage, imageName='', labelmapName='', skeletonImage=None,
#                   iterations=0, curvature=0, propagation=0):
#     # Display results
#     fig, axes = plt.subplots(nrows=1, ncols=2)
#
#     # Show original imageArray and the starting labelmap
#     castedImage = sitk.Cast(sitk.RescaleIntensity(originalImage), labelmap.GetPixelID())
#     im = sitk.LabelOverlay(castedImage, labelmap)
#     # axes[0].imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(originalImage, labelmap > 0)))
#     axes[0].imshow(sitk.GetArrayFromImage(im))
#     axes[0].set_title('Seed')
#
#     # Show original imageArray and the result of the levelset
#     castedResult = sitk.Cast(sitk.RescaleIntensity(resultImage), labelmap.GetPixelID())
#     im = sitk.LabelOverlay(castedImage, castedResult)
#     axes[1].imshow(sitk.GetArrayFromImage(im))
#     # axes[1].imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(originalImage, resultImage > 0)))
#     axes[1].set_title('Segment')
#
#     fig.suptitle("Seed: %s.\nIterations: %d\nCurvature: %f\nExpansion: %f" % \
#                              (labelmapName, iterations, curvature, propagation))
#     refresh()

# def detectAorta(imageArray, boundingBoxMin, boundingBoxMax, seed):
#     """ Detect aorta
#     :param imageArray: whole imageArray in 255 gray levels
#     :param seedY: y coord of a seed inside the circle
#     :param seedX: x coord of a seed inside the circle
#     :return: tuple (CenterX,CenterY,ratio) for the circle. None if it doesn't find it
#     """
#
#     # Crop the imageArray (2D)
#     img = imageArray[boundingBoxMin[Y]:boundingBoxMax[Y], boundingBoxMin[X]:boundingBoxMax[X]]
#     img = cv2.medianBlur(img, 5)
#
#     f = sitk.BinaryMorphologicalClosingImageFilter()
#     f.SetKernelType(sitk.BinaryMorphologicalClosingImageFilter.Ball)
#     f.SetKernelRadius(2)
#     i2 = img >= 80
#     i2 = i2.astype('uint8')
#     i2 = sitk.GetImageFromArray(i2)
#     filteredImage = f.Execute(i2)
#     img = img * sitk.GetArrayFromImage(filteredImage)
#
#     # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#     # canny = cv2.Canny(img, 150, 150); plota(canny)
#
#     # Get all the circles (each circle is an array [X, Y, ratio]
#     circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 150, param1=150, param2=10, minRadius=10,
#                                maxRadius=aortaMaxRadius);
#
#     # Draw circles (debugging purposese)
#     # imageArrayWithAorta = img.copy()
#     # drawCircles(imageArrayWithAorta, circles)
#     # plota(imageArrayWithAorta)
#     radius = 0
#     result = None
#     # Return the biggest
#     for circ in circles[0,:]:
#         # print ("Detection coords: %f,%f" % (circ[X], circ[Y]))
#         #  print ("Real coords: %f,%f" % (circ[X]+boundingBoxMin[X], circ[Y]+boundingBoxMin[Y]))
#         distance = math.sqrt(pow(circ[0]+boundingBoxMin[X]-seed[X], 2) + pow(circ[Y]+boundingBoxMin[Y]-seed[Y], 2))
#         # print(distance, circ[2], radius)
#
#         # if distance < circ[2] and distance > radius:
#         if circ[2] > distance > radius:
#             # The circles contains the seed. This is the right one
#             radius = distance
#             result = circ
#
#     return result