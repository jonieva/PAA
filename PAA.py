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

from lxml import etree
import sklearn

from cipObjectDetector import cipObjectDetector

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

def plotBinary(binarySitkimage):
    f = sitk.ScalarToRGBColormapImageFilter()
    f.SetColormap(f.Blue)
    i = f.Execute(binarySitkimage)
    sitk.Show(i)


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


def detectAortaScikit(imageArray, boundingBoxMin, boundingBoxMax, seedAorta):
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
    #img = cv2.medianBlur(img, 5)

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

    if len(filteredRadii) == 0:
        # Aorta not found
        return None
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


def getSkeletonIntersection(skeletonArray, boundingBoxMin, boundingBoxMax):
    """ Given a numpy array that represents a structure skeleton (values 0 and 1), try to detect the intersection point
    that is closest to the [X,Y] index stored in "seed"
    :param skeletonArray: numpy 2D array with the skeleton
    :param seed: [X,Y] values of the point that should be the closest one to the candidate intersections.
    :return: 2 tuples with X,Y coordinates of the intersection point and the top point of the skeleton
    (or None if not a suitable candidate found)
    """
    # Get the number of neighbours for each pixel
    ab = skeletonArray.astype(bool)
    # Discard all those points outside the bounding box
    ab[0:boundingBoxMin[1], :] = 0  # Top
    ab[boundingBoxMax[1]:, :] = 0   # Bottom
    ab[:, 0:boundingBoxMin[0]] = 0  # Left
    ab[:, boundingBoxMax[0]:] = 0   # Right

    b = np.zeros(skeletonArray.shape, dtype=int)

    # First, sum the top-right-bottom-left neighbours
    b[1:-1, 1:-1] = ab[:-2, 1:-1].astype(int) + ab[1:-1, 2:].astype(int) + \
                    ab[2:, 1:-1].astype(int) + ab[1:-1, :-2].astype(int)

    # Add the diagonals that have not a "normal" neighbour in the same direction.
    # Ex: 0 0    0 1
    #     0 1    0 1
    #     YES    NO (because it already exists a neighbour on the right or the bottom)
    b[1:-1, 1:-1] += (np.logical_and(ab[:-2, :-2], np.logical_not(np.logical_or(ab[:-2, 1:-1], ab[1:-1, :-2]))).astype(int)) + \
                     (np.logical_and(ab[:-2:, 2:], np.logical_not(np.logical_or(ab[:-2, 1:-1], ab[1:-1, 2:]))).astype(int)) + \
                     (np.logical_and(ab[2:, 2:], np.logical_not(np.logical_or(ab[1:-1, 2:], ab[2:, 1:-1]))).astype(int)) + \
                     (np.logical_and(ab[2:, :-2], np.logical_not(np.logical_or(ab[2:, 1:-1], ab[1:-1, :-2]))).astype(int))

    # Get the point that has the lower y coordinate
    topPoint = np.argwhere(ab)[0]

    # Filter those that have a high connectivity (at least b=3) and are part od the skeleton
    b[ab == False] = 0
    points = np.where(b >= 3)

    # Return the closest point to the top point of the skeleton
    index = -1
    minDistance = sys.maxint
    for i in range(len(points[0])):
        #
        distance = abs(topPoint[0]-points[0][i]) + abs(topPoint[1]-points[1][i])
        if distance < minDistance:
            minDistance = distance
            index = i
    if index != -1:
        return (points[1][index], points[0][index]), (topPoint[1], topPoint[0])
    # No suitable candidate found
    return None, None


def calculatePADiameter(paLabelmapArray, paBifurcationPoint, topSkeletonPoint):
    """
    Get the diameter and the intersection points for the pulmonary artery
    :param paLabelmapArray: labelmap of the PA (numpy array)
    :param paBifurcationPoint: X,Y coordinates from PA bifurcation
    :param topSkeletonPoint: X,Y coordinates with the point is on the top of the PA skeleton
    :return: tuple (diameter, (x,y) for left intersection, (x,y) for right intersection)
    """
    # TODO: change doc
    # Calculate the middle point between the top of the skeleton and the bifurcation point
    maxx = max(paBifurcationPoint[0], topSkeletonPoint[0])
    minx = min(paBifurcationPoint[0], topSkeletonPoint[0])
    cx = int(round(float(maxx - minx) / 2 + minx))
    # We assume that the top point of the skeleton is always beneath the bifurcation
    cy = int(round((float(topSkeletonPoint[1] - paBifurcationPoint[1]) / 2) + paBifurcationPoint[1]))

    # Calculate the line equations for a perpendicular line that crosses the middle point.
    # To that end, we have to calculate first the equations of the line that connect the two points
    if topSkeletonPoint[0] == paBifurcationPoint[0]:
        # Both points are in the same vertical line. The perpendicular line is horizontal (b = x0; m=0) in the mid point
        mp = 0
        bp = cy
    elif topSkeletonPoint[1] == paBifurcationPoint[1]:
        # Error case (horizontal line). Both points shouldn't be in the same y coordinate (the perpendicular line would be vertical)
        raise Exception("The PA bifurcation point and the top skeleton point have the same y coordinate")
    else:
        # Regular case
        m = (paBifurcationPoint[1] - topSkeletonPoint[1]) / float(paBifurcationPoint[0] - topSkeletonPoint[0])  # m = (y0-y1)/(x0-x1)
        #b = topSkeletonPoint[1] - m * topSkeletonPoint[0]   # b = y0-m*x0

        mp = -1/m   # The perpendicular line will have mp = -1/m
        bp = cy - mp*cx     # With mp calculated, calculate the new slope bp (y = mx + b ==> b = y - mx

    (p0, p1) = getIntersectionPoints(paLabelmapArray, (cx, cy), mp, bp, 100)

    if p0 is None:
        raise Exception("PA diameter could not be calculated")
    paDiameter = distance(p0, p1)
    #return paDiameter / (aortaCircle[2] * 2)
    return paDiameter, p0, p1

def getIntersectionPoints(paLabelmapArray, point, m, b, delta):
    """ Measure the diameter of an structure that intersects with a line of equation y = mx+b in at least 2 points
    :param paLabelmapArray: labelMap image where the values of the structure have a value != 0
    :param point: X,Y coords of the "center" of the structure
    :param m: slope of the line
    :param b: intercept of the line
    :param delta: range to look for the intersection points ([pointX-delta, pointX+delta])
    :return: distance measured or None if no
    """
    # Sample the point in a range [cx-delta, cx]

    sampleX = np.array(range(delta, 0, -1), np.int)
    sampleX = point[0] - sampleX
    sampleY = np.rint(m * sampleX + b)
    sampleY = sampleY.astype(np.int)
    sample = paLabelmapArray[sampleY, sampleX]
    # Get the maximum index (in case there is more than one intersection point)
    r = np.where(sample == 0)
    if len(r[0]) == 0:
        # Not intersection point found
        return None
    else:
        # Get the coordinates of the first intersection (X,Y format)
        index = r[0].max()
        p0 = [sampleX[index], sampleY[index]]

    # Right size (range [cx,cx+delta])
    sampleX = np.array(range(delta), np.int)
    sampleX = sampleX + point[0]
    sampleY = np.rint(m * sampleX + b)
    sampleY = sampleY.astype(np.int)
    sample = paLabelmapArray[sampleY, sampleX]

    # Get the minimum index (in case there is more than one intersection point)
    r = np.where(sample == 0)
    if len(r[0]) == 0:
        # Not intersection point found
        return None
    else:
        # Get the coordinates of the first intersection (X,Y format)
        index = r[0].min()
        p1 = [sampleX[index], sampleY[index]]

    # Return the two coordinates
    return p0, p1


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)


def executePipeline(originalImage, boundingBoxMin, boundingBoxMax, seedPA, seedAorta, plotImage=True, saveImageOutputPath=None):
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
    :param originalImage: simpleITK image
    :param boundingBoxMin: top-left XYZ coordinate of the 2D slice that we will choose as the working slice
    :param boundingBoxMax: bottom-right XYZ coordinate of the 2D slice we will choose as the working slice
    :param seedAorta: XYZ index of a point that is known to be inside the aorta
    :param seedPA: XYZ index of a point that is known to be inside the PA
    :param plotImage: plot the result image
    :param displayResults: display the results from PA:A ratio
    :param saveImageOutputPath: when not None, save a result image in this FOLDER (the name of the file depends on the
    case and the current date and time)
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


    # originalImage = sitk.ReadImage(imageFullPath)
    # Select just the slice that we are going to work in
    slice = boundingBoxMin[Z]

    originalImageSlice = originalImage[:, :, slice]
    # Thresholding
    imageArray = adaptIntensities(sitk.GetArrayFromImage(originalImageSlice))
    # Smoothing
    imageArray = cv2.medianBlur(imageArray, 5)
    # circ2 = detectAorta(imageArray, boundingBoxMin, boundingBoxMax, seedAorta)
    aortaCircle = detectAortaScikit(imageArray, boundingBoxMin, boundingBoxMax, seedAorta)

    if aortaCircle is None:
        print("Aorta not detected in case ", imageFullPath)
        return

    # Draw the circle in the original image
    imageArrayWithAorta = imageArray
    aortaCircle[0] = aortaCircle[0]+boundingBoxMin[X]
    aortaCircle[1] = aortaCircle[1]+boundingBoxMin[Y]
    drawCircle(imageArrayWithAorta, aortaCircle)
    # plota(imageArrayWithAorta)

    # Convert the arrays in itk images to work with simpleitk filters
    image = sitk.GetImageFromArray(imageArrayWithAorta)
    # Generate seed in the labelmap image (get the first two dimensions in the seed because we are now working in 2D)
    labelmapSeed = generateSeed(image, seedPA[X:Z])

    labelmap = applyLevelset(image, labelmapSeed, numberOfIterations, curvatureScaling, propagationScaling,
                             lowerThreshold, upperThreshold, maximumRMSError)

    # Skeleton. First dilate to close holes.
    #paDilated = sitk.BinaryDilate(labelmap > 0)
    #paDilated = sitk.BinaryFillhole(labelmap > 0)
    f = sitk.BinaryDilateImageFilter()
    f.SetKernelType(f.Ball)
    f.SetKernelRadius(5)
    paDilated = f.Execute(labelmap > 0)

    skeleton = sitk.BinaryThinning(paDilated)

    skeletonArray = sitk.GetArrayFromImage(skeleton)
    paBifurcationPoint, topSkeletonPoint = getSkeletonIntersection(skeletonArray, boundingBoxMin, boundingBoxMax)

    labelmapArray = sitk.GetArrayFromImage(labelmap)
    (paDiameter, pa0, pa1) = calculatePADiameter(labelmapArray, paBifurcationPoint, topSkeletonPoint)

    # Draw the skeleton and the intersection point

    labelmapArray[skeletonArray == 1] = 2    # (label 2 to display results)
    cv2.circle(labelmapArray, (paBifurcationPoint[0], paBifurcationPoint[1]), 5, (4, 0, 0), -2)

    # Draw the line that will be used to measure the PA radious
    #paLevel = intersection[1]-15
    #cv2.line(labelmapArray, (0, paLevel), (imageArray.shape[1], paLevel), (5,0,0))

    cv2.line(labelmapArray, (pa0[0], int(pa0[1])), (pa1[0], int(pa1[1])), (5, 0, 0))
    # Draw the aorta circle just to display results (label 3)
    cv2.circle(labelmapArray, (aortaCircle[0], aortaCircle[1]), aortaCircle[2], (3, 0, 0), 1)

    labelmap = sitk.GetImageFromArray(labelmapArray)


    # output2 = sitk.LabelOverlay(image, labelmap)
    # plot(output2)

    caseId = os.path.basename(imageFullPath).replace(".nhdr", "")
    output = sitk.LabelOverlay(image, labelmap)
    saveImage = (saveImageOutputPath is not None)
    if plotImage or saveImage:
        # Show the final results
        plot(output)
        plt.title('Results ' + caseId)
        refresh()


    # Calculate the PAA ratio
    # calculatePAARatio(aortaCircle, intersection[:2], intersection[2:])
    # pa = sitk.GetArrayFromImage(paDilated)
    # # Get the minimun index for the PA (center of the aorta+radious)
    # idx = aortaCircle[0] + aortaCircle[2]
    # nz = np.nonzero(pa[paLevel, idx:])
    # minPA = np.min(nz) + idx
    # maxPA = np.max(nz) + idx
    # paDiameter = maxPA-minPA
    aortaDiameter = aortaCircle[2] * 2
    paaRatio = paDiameter / float(aortaDiameter)

    # if displayResults:
    #     print("PA diameter: {0}".format(paDiameter))
    #     print("Aorta diameter: {0}".format(aortaDiameter))
    #     print("PA:A ratio: {0}".format(paaRatio))

    if saveImage:
        plt.savefig(os.path.join(saveImageOutputPath, os.path.basename(imageFullPath)) + '-' +
                time.strftime("%Y-%m-%d_%H-%M") + '.png')
        cl()

    return caseId, paaRatio, paDiameter, pa0, pa1, aortaDiameter, (aortaCircle[0], aortaCircle[1])


def save(imageName):
    plt.savefig(os.path.join(homepath, 'tempdata/results', os.path.basename(imageName)) + '-' +
                time.strftime("%Y-%m-%d_%H-%M") + '.png')
    cl()


def loadCaseParameters(caseXmlFullPath):
    """ Load the xml that contains the bounding boxes for a case
    Returns the bounding boxes and the seeds for PulmonaryArtery structure
    :param folder:
    :param caseName:
    :return:
    - [X,Y,Z] bounding box top left
    - [X,Y,Z] bounding box bottom right
    - [X,Y,Z] seed Pulmonary artery
    - [X,Y,Z] seed Aorta
    """
    with open(caseXmlFullPath) as f:
        xml = f.read()

    root = etree.fromstring(xml)
    # Find the node whose Description is PulmonaryArteryAxial
    e = root.xpath('BoundingBox/Description[text()="PulmonaryArteryAxial"]')
    if len(e) == 0:
        raise Exception("PulmonaryArteryAxial node not found in " + caseXmlFullPath)
    # Get the parent node (BoundingBox)
    boundingBoxNode = e[0].getparent()
    # Get the top-left coordinates
    boundingBoxMin = []
    for i in boundingBoxNode.findall("Start/value"):
        boundingBoxMin.append(int(i.text))
    # Get the bottom-right coordinates (offset from previous one)
    boundingBoxMax = []
    index = 0
    for i in boundingBoxNode.findall("Size/value"):
        boundingBoxMax.append(int(i.text) + boundingBoxMin[index])
        index += 1
    # Calculate seeds as offsets of the bounding boxes
    seedPA = [boundingBoxMax[0]-30, boundingBoxMax[1]-50, boundingBoxMax[2]]
    seedAorta = [boundingBoxMin[0]+30, boundingBoxMin[1]+30, boundingBoxMin[2]]

    return boundingBoxMin, boundingBoxMax, seedPA, seedAorta

def loadParametersFromAutomaticDetector(sitkImage):
    detector = cipObjectDetector(verbose=True)
    detector.loadVolumeFromSITKImage(sitkImage)
    detector.loadCascade(os.path.join(homepath, "Projects/acil/acil_python/ObjectDetection/Detectors/ChestCTPulmonaryArteryAxial.xml"))
    coords = detector.detect(useLocationConstraints=True)
    boundingBoxMin = [int(i) for i in coords[:3]]
    boundingBoxMax = [int(i) for i in coords[3:]]
    # Calculate seeds as offsets of the bounding boxes
    seedPA = [boundingBoxMax[0]-30, boundingBoxMax[1]-50, boundingBoxMax[2]]
    seedAorta = [boundingBoxMin[0]+30, boundingBoxMin[1]+30, boundingBoxMin[2]]

    return boundingBoxMin, boundingBoxMax, seedPA, seedAorta


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
paStructureId = "PulmonaryArteryAxial"


if __name__ == '__main__':
    casesListFile = homepath + "/tempdata/structuresDetection/cases.txt"
    with open(casesListFile) as f:
        for case in f.readlines():
            try:
                case = case.rstrip()
                structures = loadCaseParameters("{0}/tempdata/structuresDetection/{1}_structures.xml".format(homepath, case))
                imageFullPath = "{0}/tempdata/structuresDetection/{1}.nhdr".format(homepath, case)
                image = sitk.ReadImage(imageFullPath)
                #detector.loadVolumeFromSITKImage(image)
                structures = loadParametersFromAutomaticDetector(image)
                executePipeline(image, structures[0], structures[1], structures[2], structures[3], plotImage=True,
                                saveImageOutputPath=os.path.join(homepath, "tempdata/results"))
            except Exception as ex:
                print("CASE {0} FAILED: ".format(case), ex)

# else:
#     case = '10047G_INSP_STD_BWH_COPD'
#     imageFullPath = '{0}/tempdata/structuresDetection/{1}.nhdr'.format(homepath, case)
#     image = sitk.ReadImage(imageFullPath)
#     #structures = loadCaseParameters("{0}/tempdata/structuresDetection/{1}_structures.xml".format(homepath, case))
#
#     structures = loadParametersFromAutomaticDetector(image)
#
#     boundingBoxMin = structures[0]
#     boundingBoxMax = structures[1]
#     seedPA = structures[2]
#     seedAorta = structures[3]
#
#     executePipeline(image, structures[0], structures[1], structures[2], structures[3])


# slice = 214
# # XYZ
# seedAorta = [250,210, slice]
# seedPA = [295, 222, slice]
# boundingBoxMin = [228,193, slice]
# boundingBoxMax = [311,251, slice]
# executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
# save(imageName)


    # #
    # # Shitty
    # imageName = homepath + '/tempdata/023960002463_INSP_B35f_L1_ECLIPSE.nhdr'
    # slice = 214
    # # XYZ
    # seedAorta = [250,210, slice]
    # seedPA = [295, 222, slice]
    # boundingBoxMin = [228,193, slice]
    # boundingBoxMax = [311,251, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    #
    # # So so ==> OK aorta (fail skeleton intersection)
    # imageName = homepath + '/tempdata/10002K_INSP_STD_BWH_COPD.nhdr'
    # slice = 391
    # # XYZ
    # seedAorta = [249, 175, slice]
    # seedPA = [294,186, slice]
    # boundingBoxMin = [178, 141, slice]
    # boundingBoxMax = [361, 287, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    # # WRONG ==> ok
    # imageName = homepath + '/tempdata/10005Q_INSP_STD_NJC_COPD.nhdr'
    # slice = 325
    # # XYZ
    # seedAorta = [225,200, slice]
    # seedPA = [268, 203, slice]
    # boundingBoxMin = [201,170, slice]
    # boundingBoxMax = [324,273, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    # # OK  ==> aorta is kind of too big
    # imageName = homepath + '/tempdata/10004O_INSP_STD_BWH_COPD.nhdr'
    # slice = 415
    # # XYZ
    # seedAorta = [223,227, slice]
    # seedPA = [272, 246, slice]
    # boundingBoxMin = [187,198, slice]
    # boundingBoxMax = [298,296, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    # # OK
    # imageName = homepath + '/tempdata/10006S_INSP_STD_BWH_COPD.nhdr'
    # slice = 399
    # # XYZ
    # seedAorta = [228,202, slice]
    # seedPA = [272, 210, slice]
    # boundingBoxMin = [184,171, slice]
    # boundingBoxMax = [310,268, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    # # Ok
    # imageName = homepath + '/tempdata/12257B_INSP_STD_UIA_COPD.nhdr'
    # slice = 442
    # # XYZ
    # seedAorta = [238,220, slice]
    # seedPA = [287, 248, slice]
    # boundingBoxMin = [216,199, slice]
    # boundingBoxMax = [312,273, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)
    #
    # # WRONG ==> OK
    # imageName = homepath + '/tempdata/10015T_INSP_STD_BWH_COPD.nhdr'
    # slice = 458
    # # XYZ
    # seedAorta = [248,200, slice]
    # seedPA = [291, 212, slice]
    # boundingBoxMin = [214,169, slice]
    # boundingBoxMax = [319,250, slice]
    # executePipeline(imageName, boundingBoxMin, boundingBoxMax, seedPA, seedAorta)
    # save(imageName)



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