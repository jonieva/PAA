'''
Created on Feb 23, 2015

@author: Jorge
'''

'''
Install ImageJ

conda install SimpleITK
conda install ipython?
conda install matplotlib
conda install binstar
pip install pynrrd
conda install -c https://conda.binstar.org/asmeurer ipython-notebook
conda install pyqt
conda install opencv
'''

import nrrd
import os
import math
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2
import cv2.cv

# class PAALogic:
#   def __init__(self):
#     pass

Z=2
Y=1
X=0

def extractAxialSlice(filePath, sliceNumber):
  ct_data, imageInfo = nrrd.read(filePath)

  # create a new ct file with that slice number
  ct_data_slice = ct_data[:,:,sliceNumber:sliceNumber+1]
  outputFilePath = '%s_%i.nrrd' % (os.path.splitext(filePath)[0], sliceNumber)
  nrrd.write(outputFilePath, ct_data_slice)


def plot(imageArray):
  array = sitk.GetArrayFromImage(imageArray)
  plota(array)

def plota(array):
  f = plt.figure()
  plt.imshow(array, cmap = cm.Greys_r)
  refresh()
  return f.number

def arr(imageArray):
  return sitk.GetArrayFromImage(imageArray)

def cl():
  plt.close('all')

def r(interval=0.1):
  refresh(interval)

def refresh(interval=0.1):
  plt.pause(interval)






# Read main imageArray
# inputImagePath = '/Users/Jorge/tempdata/12257B_INSP_STD_UIA_COPD_Easy1.png'
# imageVector = sitk.ReadImage(inputImagePath)
# imageArray=sitk.VectorIndexSelectionCast(imageVector)



def readImages(imageName, labelmapName, dataFolder, slice):
  """Read a slice from the original imageArray and the label map.
  It returns a tuple like:
  - OriginalImage
  - OriginalLabelmap
  - ImageArray
  - LabelmapArray
  """
  inputImagePath = dataFolder + imageName
  imageArray = sitk.ReadImage(inputImagePath)
  imageArray = imageArray[:,:,slice]

  seedImagePath = dataFolder + labelmapName
  labelmapImage = sitk.ReadImage(seedImagePath)
  labelmapImage = labelmapImage[:,:,slice]

  return (imageArray, labelmapImage, sitk.GetArrayFromImage(imageArray), sitk.GetArrayFromImage(labelmapImage))

def generateSeed(image, seedIndex):
  ''' Return a sitk image with a seedIndex starting from the middle of the image
  :param image:
  :return:
  '''
  imSize = image.GetSize()
  seedImage = sitk.Image(imSize, sitk.sitkUInt8)
  seedImage.CopyInformation(image)
  seedImage[seedIndex] = 1
  seedIndex = sitk.BinaryDilate(seedImage, 3)
  return seedIndex

def preprocessImage(img):
  """Apply a filter to preprocess the imageArray (median, gausian, etc.).
  It returns the filtered imageArray."""
  filter = sitk.MedianImageFilter()
  img = filter.Execute(img)
  return img


def applyLevelset(image, labelmap, numberOfIterations, curvatureScaling, propagationScaling, lowerThreshold, upperThreshold, maximumRMSError):
  ''' Apply a region growing algorithm to grow the initial seed.
  :param imageArray: original or preprocessed imageArray
  :return: a labelmap imageArray with the same size as 'imageArray' and value=1 in the grown region
  '''
  # img = sitk.ConnectedThreshold(image1=imageArray,
  #                                             seedList=seeds,
  #                                             lower=0,
  #                                             upper=300,
  #                                             replaceValue=1)
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

def getImageStats(imageArray, labelmap):
  """Get some standard stats for the imageArray"""
  stats = sitk.LabelStatisticsImageFilter()
  stats.Execute(imageArray, labelmap)
  return stats



def applyLevelSet(imageArray, labelmap, curvatureScaling, propagationScaling, iterations, maximumRMSError, lower_threshold, upper_threshold):
  """Apply a levelset with 'imageArray' and 'labelmap' and return a result imageArray"""
  # Init distances
  init_ls = sitk.SignedMaurerDistanceMap(labelmap, insideIsPositive=True, useImageSpacing=True)

  # Init filter
  lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
  lsFilter.SetLowerThreshold(lower_threshold)
  lsFilter.SetUpperThreshold(upper_threshold)
  # Additional parameters
  lsFilter.SetMaximumRMSError(maximumRMSError)
  lsFilter.SetNumberOfIterations(iterations)
  lsFilter.SetCurvatureScaling(curvatureScaling)
  lsFilter.SetPropagationScaling(propagationScaling)
  lsFilter.ReverseExpansionDirectionOn()

  # Execute filter
  resultImage = lsFilter.Execute(init_ls, sitk.Cast(imageArray, sitk.sitkFloat32))
  return resultImage



def displayResults(originalImage, labelmap, resultImage, imageName='', labelmapName='', skeletonImage=None, iterations=0, curvature=0, propagation=0):
  # Display results
  fig, axes = plt.subplots(nrows=1, ncols=2)

  # Show original imageArray and the starting labelmap
  castedImage = sitk.Cast(sitk.RescaleIntensity(originalImage), labelmap.GetPixelID())
  im = sitk.LabelOverlay(castedImage, labelmap)
  #axes[0].imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(originalImage, labelmap > 0)))
  axes[0].imshow(sitk.GetArrayFromImage(im))
  axes[0].set_title('Seed')

  # Show original imageArray and the result of the levelset
  castedResult = sitk.Cast(sitk.RescaleIntensity(resultImage), labelmap.GetPixelID())
  im = sitk.LabelOverlay(castedImage, castedResult)
  axes[1].imshow(sitk.GetArrayFromImage(im))
  #axes[1].imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(originalImage, resultImage > 0)))
  axes[1].set_title('Segment')

  fig.suptitle("Seed: %s.\nIterations: %d\nCurvature: %f\nExpansion: %f" % \
               (labelmapName, iterations, curvature, propagation))
  refresh()

  # bt = sitk.BinaryDilate(ls > 0)
  # sk = sitk.BinaryThinning(bt)
  # resultImage = sitk.LabelOverlay(originalImage,sk)
  # axes[2].set_title('Skeleton')
  # axes[2].imshow(sitk.GetArrayFromImage(resultImage))
  # refresh()

# idx = (552,432)
# seg = sitk.Image(imageArray.GetSize(), sitk.sitkUInt8)
# seg.CopyInformation(imageArray)
# seg[idx] = 1
# seg = sitk.BinaryDilate(seg, 3)


# sega = sitk.GetArrayFromImage(seg)
#print(stats)
#

def adaptIntensities(imageArray):
  ''' Adapt the intensities of an imageArray and convert to 255
  :param imageArray:
  :return:
  '''
#img = sitk.GetArrayFromImage(originalImage)
  thresholdMin = -150
  thresholdMax = 250
  imageArray[imageArray <= thresholdMin] = thresholdMin
  imageArray[imageArray >= thresholdMax] = thresholdMax
  imageArray = 254*(imageArray.astype(np.float32)-thresholdMin)/(thresholdMax-thresholdMin)
#   imageArray[sitk.RescaleIntensity(sitk.sitkUInt8)]
  imageArray = imageArray.astype(np.uint8)

  return imageArray

def detectAorta(imageArray, boundingBoxMin, boundingBoxMax, seed):
  ''' Detect aorta
  :param imageArray: whole imageArray in 255 gray levels
  :param seedY: y coord of a seed inside the circle
  :param seedX: x coord of a seed inside the circle
  :return: tuple (CenterX,CenterY,ratio) for the circle. None if it doesn't find it
  '''

  # Crop the imageArray (2D)
  img = imageArray[boundingBoxMin[Y]:boundingBoxMax[Y], boundingBoxMin[X]:boundingBoxMax[X]]
  img = cv2.medianBlur(img,5)

  #img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
  #canny = cv2.Canny(img, 100, 100); plota(canny)

  # Get all the circles (each circle is an array [X, Y, ratio]
  circles = cv2.HoughCircles(img,cv2.cv.CV_HOUGH_GRADIENT,1,100,param1=100,param2=10,minRadius=10,maxRadius=aortaMaxRadius)

  # Draw circles (debugging purposese)
  #imageArrayWithAorta = img.copy()
  #drawCircles(imageArrayWithAorta, circles[0])
  #plota(imageArrayWithAorta)
  radius = 0
  result = None
  # Return the biggest
  for circ in circles[0,:]:
#     print ("Detection coords: %f,%f" % (circ[X], circ[Y]))
#     print ("Real coords: %f,%f" % (circ[X]+boundingBoxMin[X], circ[Y]+boundingBoxMin[Y]))
    distance = math.sqrt(pow(circ[0]+boundingBoxMin[X]-seed[X],2) + pow(circ[Y]+boundingBoxMin[Y]-seed[Y],2))
#     print("Ratio: %f. Distance: %f" % (circ[2], d))   
    if distance < circ[2] and distance > radius:
      # The circles contains the seed. This is the right one
      #print (d)
      radius = distance
      result = circ
     
  return result

def drawCircles(imageArray, circles):
  for circ in circles: drawCircle(imageArray, circ)

def drawCircle(imageArray, circle):
  cv2.circle(imageArray,(circle[0],circle[1]),circle[2],(0,0,0),2)
  #plota(imageArray)

def executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax):
  #factor = 1.5
  numberOfIterations=1000
  maximumRMSError = 0.02
  
  curvatureScaling = 1
  propagationScaling = 1

  # Set thresholds for a bigger/smaller segmented area
  # stats = getImageStats(imageArray, labelmap)
  # lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
  # upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
  lowerThreshold = 80
  upperThreshold = 150


  originalImage = sitk.ReadImage(imageName)
  # Select just the slice that we are going to work in
  slice = boundingBoxMin[Z]

  originalImageSlice = originalImage[:,:,slice]
  imageArray = adaptIntensities(sitk.GetArrayFromImage(originalImageSlice))
  circ = detectAorta(imageArray, boundingBoxMin, boundingBoxMax, seedAorta)

  if circ is None:
    print("Aorta not detected!")
    return

  # Draw the circle in the original image
  imageArrayWithAorta = imageArray
  circ[0] = circ[0]+boundingBoxMin[X]
  circ[1] = circ[1]+boundingBoxMin[Y]
  drawCircle(imageArrayWithAorta, circ)
  #plota(imageArrayWithAorta)

  # Convert the arrays in itk images to work with simpleitk filters
  image = sitk.GetImageFromArray(imageArrayWithAorta)
  # Generate seed in the labelmap image (get the first two dimensions in the seed because we are now working in 2D)
  labelmapSeed = generateSeed(image, seedPA[X:Z])

  labelmap = applyLevelset(image, labelmapSeed, numberOfIterations, curvatureScaling, propagationScaling, lowerThreshold, upperThreshold, maximumRMSError)



  # Skeleton
  bin = sitk.BinaryDilate(labelmap > 0)
  skeleton = sitk.BinaryThinning(bin)

  skeletonArray = sitk.GetArrayFromImage(skeleton)
  labelmapArray = sitk.GetArrayFromImage(labelmap)
  labelmapArray[skeletonArray] = 2  # (label 2 to display results)

  # Draw the aorta circle just to display results (label 3)
  cv2.circle(labelmapArray,(circ[0],circ[1]),circ[2],(3,0,0),1)

  labelmap = sitk.GetImageFromArray(labelmapArray)

  # f = sitk.ScalarToRGBColormapImageFilter()
  # f.SetColormap(f.Cool)
  # i = f.Execute(labelmap)
  # sitk.Show(i)


  output2 = sitk.LabelOverlay(image, labelmap)
  plot(output2)
  plt.title('Results ' + os.path.basename(imageName))
  refresh()

  output3 = sitk.LabelOverlay(image, skeleton); plot(output3)

#dataFolder = homepath + '/tempdata/'
#imageName='12257B_INSP_STD_UIA_COPD.nhdr'
#labelmapName='12257B_INSP_STD_UIA_COPD-label.nrrd'

#imageName=homepath + '/Dropbox/acil-biomarkers/BiomarkerHackathon/10002K_INSP_STD_BWH_COPD_PulmonaryArteryBifurcationAxial_crop.nhdr'

homepath = os.path.expanduser("~")

aortaMaxRadius = 35

imageName = homepath + '/tempdata/structuresDetection/10002K_INSP_STD_BWH_COPD.nhdr'
slice = 391
# XYZ
seedAorta = [249, 175, slice]
seedPA = [294,186, slice]
boundingBoxMin = [178, 141, slice]
boundingBoxMax = [361, 287, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)


imageName = homepath + '/tempdata/structuresDetection/10005Q_INSP_STD_NJC_COPD.nhdr'
slice = 330
# XYZ
seedAorta = [225,200, slice]
seedPA = [268, 203, slice]
boundingBoxMin = [201,184, slice]
boundingBoxMax = [324,273, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)

imageName = homepath + '/tempdata/structuresDetection/10004O_INSP_STD_BWH_COPD.nhdr'
slice = 415
# XYZ
seedAorta = [223,227, slice]
seedPA = [272, 246, slice]
boundingBoxMin = [187,198, slice]
boundingBoxMax = [298,296, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)

imageName = homepath + '/tempdata/structuresDetection/10006S_INSP_STD_BWH_COPD.nhdr'
slice = 399
# XYZ
seedAorta = [228,202, slice]
seedPA = [265, 215, slice]
boundingBoxMin = [184,171, slice]
boundingBoxMax = [310,268, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)

imageName = homepath + '/tempdata/023960002463_INSP_B35f_L1_ECLIPSE.nhdr'
slice = 215
# XYZ
seedAorta = [250,210, slice]
seedPA = [295, 222, slice]
boundingBoxMin = [227,188, slice]
boundingBoxMax = [316,251, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)

imageName = homepath + '/tempdata/12257B_INSP_STD_UIA_COPD.nhdr'
slice = 442
# XYZ
seedAorta = [238,220, slice]
seedPA = [287, 248, slice]
boundingBoxMin = [216,199, slice]
boundingBoxMax = [282,442, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)


imageName = homepath + '/tempdata/10015T_INSP_STD_BWH_COPD.nhdr'
slice = 458
# XYZ
seedAorta = [248,200, slice]
seedPA = [291, 212, slice]
boundingBoxMin = [214,169, slice]
boundingBoxMax = [319,250, slice]
executePipeline(imageName, seedAorta, seedPA, boundingBoxMin, boundingBoxMax)









# rows = 2
# cols = 2
#
# p.ioff()
#
# images = [[0 for i in range(rows)] for j in range(cols)]
# lsFilter.SetNumberOfIterations(500)
# img_T1f = sitk.Cast(imageArray, sitk.sitkFloat32)
# ls = init_ls
# niter = 0
# fig, axes = p.subplots(nrows=rows, ncols=cols)
# for i in range(rows):
#   for j in range(cols):
#     ls = lsFilter.Execute(ls, img_T1f)
#     niter += lsFilter.GetNumberOfIterations()
#     t = "LevelSet after "+str(niter)+" iteratiocd CIPns and RMS "+str(lsFilter.GetRMSChange())
#     #t = "I %i; RMS %f" % (niter, lsFilter.GetRMSChange())
#     #print(t)
#     #fig = myshow3d(sitk.LabelOverlay(img_T1_255, ls > 0), zslices=range(idx[2]-zslice_offset,idx[2]+zslice_offset+1,zslice_offset), dpi=20, title=t)
#     images[i][j] = sitk.GetArrayFromImage(sitk.LabelOverlay(imageArray, ls > 0))
#     axes[i,j].set_title(t)
#     axes[i,j].imshow(images[i][j])
#     p.pause(0.1)
    #plt.pause(0.1)

# plt.switch_backend('Qt4Agg')


# for i in range(rows):
#   for j in range(cols):
#     axes[i,j].imshow(images[i][j])
#     p.pause(0.1)


# filename = "/Data/jonieva/tempdata/Cases to fix/10002K_INSP_STD_BWH_COPD_structuresDetection.nhdr"
# i = sitk.ReadImage(filename)
# a = sitk.GetArrayFromImage(i)
