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
import SimpleITK as sitk

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import cv2
import cv2.cv

# class PAALogic:
#   def __init__(self):
#     pass



def extractAxialSlice(filePath, sliceNumber):
  ct_data, imageInfo = nrrd.read(filePath)

  # create a new ct file with that slice number
  ct_data_slice = ct_data[:,:,sliceNumber:sliceNumber+1]
  outputFilePath = '%s_%i.nrrd' % (os.path.splitext(filePath)[0], sliceNumber)
  nrrd.write(outputFilePath, ct_data_slice)


def plot(image):
  array = sitk.GetArrayFromImage(image)
  plota(array)

def plota(array):
  f = plt.figure()
  plt.imshow(array, cmap = cm.Greys_r)
  refresh()
  return f.number

def arr(image):
  return sitk.GetArrayFromImage(image)

def cl():
  plt.close('all')

def r(interval=0.1):
  refresh(interval)

def refresh(interval=0.1):
  plt.pause(interval)






# Read main image
# inputImagePath = '/Users/Jorge/tempdata/12257B_INSP_STD_UIA_COPD_Easy1.png'
# imageVector = sitk.ReadImage(inputImagePath)
# image=sitk.VectorIndexSelectionCast(imageVector)



def readImages(imageName, labelmapName, dataFolder, slice):
  """Read a slice from the original image and the label map.
  It returns a tuple like:
  - OriginalImage
  - OriginalLabelmap
  - ImageArray
  - LabelmapArray
  """
  inputImagePath = dataFolder + imageName
  image = sitk.ReadImage(inputImagePath)
  image = image[:,:,slice]

  seedImagePath = dataFolder + labelmapName
  labelmapImage = sitk.ReadImage(seedImagePath)
  labelmapImage = labelmapImage[:,:,slice]

  return (image, labelmapImage, sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(labelmapImage))

def generateSeed(image):
  ''' Return a sitk image with a seed starting from the middle of the image
  :param image:
  :return:
  '''
  imSize = image.GetSize()
  idx = (imSize[0]/2,imSize[1]/2)
  seed = sitk.Image(imSize, sitk.sitkUInt8)
  seed.CopyInformation(image)
  seed[idx] = 1
  seed = sitk.BinaryDilate(seed, 3)
  return seed

def preprocessImage(img):
  """Apply a filter to preprocess the image (median, gausian, etc.).
  It returns the filtered image."""
  filter = sitk.MedianImageFilter()
  img = filter.Execute(img)
  return img


def growSeed(image, labelmap):
  ''' Apply a region growing algorithm to grow the initial seed.
  :param image: original or preprocessed image
  :param seed: coordinates of the seed/s (ex: [(290,254]) )
  :return: a labelmap image with the same size as 'image' and value=1 in the grown region
  '''
  # img = sitk.ConnectedThreshold(image1=image,
  #                                             seedList=seeds,
  #                                             lower=0,
  #                                             upper=300,
  #                                             replaceValue=1)
  init_ls = sitk.SignedMaurerDistanceMap(labelmap, insideIsPositive=True, useImageSpacing=True)
  lsFilter = sitk.ThresholdSegmentationLevelSetImageFilter()
  lsFilter.SetLowerThreshold(0)
  lsFilter.SetUpperThreshold(300)
  # Additional parameters
  lsFilter.SetMaximumRMSError(maximumRMSError)
  lsFilter.SetNumberOfIterations(iterations)
  lsFilter.SetCurvatureScaling(curvatureScaling)
  lsFilter.SetPropagationScaling(propagationScaling)
  lsFilter.ReverseExpansionDirectionOn()

  im = lsFilter.Execute()
  return im > 0

def getImageStats(image, labelmap):
  """Get some standard stats for the image"""
  stats = sitk.LabelStatisticsImageFilter()
  stats.Execute(image, labelmap)
  return stats



def applyLevelSet(image, labelmap, curvatureScaling, propagationScaling, iterations, maximumRMSError, lower_threshold, upper_threshold):
  """Apply a levelset with 'image' and 'labelmap' and return a result image"""
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
  resultImage = lsFilter.Execute(init_ls, sitk.Cast(image, sitk.sitkFloat32))
  return resultImage





def detectCircles(image, topleftY=0, topleftX=0, height=0, width=0):
  border = 5
  if height == 0 or width==0:
    croppedImage = image
  else:
    croppedImage = image[topleftY-border:topleftY+height+border, topleftX:topleftX+width+border]

  # Convert the image to a uint8 array that opencv can handle
  imgRescaled = sitk.RescaleIntensity(croppedImage, sitk.sitkUInt8)
  imgArray = sitk.GetArrayFromImage(imgRescaled)
  imgArray = imgArray.astype('uint8')

  # Smoothing
  #imgArray = cv2.medianBlur(imgArray,5)
  # Negative
  imgArray = 255 - imgArray
  # circles = cv2.HoughCircles(croppedImage,cv2.CV_HOUGH_GRADIENT,1,80,
  # param1=10,param2=5,minRadius=3,maxRadius=10)
  circles = cv2.HoughCircles(imgArray, cv2.cv.CV_HOUGH_GRADIENT, 1,50, param1=5,param2=5,minRadius=5,maxRadius=100)





  # rect is defined as [z,x,y,w,h]
def detectAorta(image):
  ''' Detect the aorta in a filtered slice
  :param imgArray: numpy array in 2D
  :return: label map where aorta=1
  '''
  # Approx. coordinates where the Aorta should be contained, having in mind that numpy arrays have [zyx] coordinates.
  topleftY = 180
  topleftX = 190
  height = 95
  width = 100
  border=5

  #cimg = cv2.cvtColor(imgArray,cv2.COLOR_GRAY2BGR)
  # Crop the image to fit the aorta bounding box aprox.
  #croppedImage = imgArray[topleftY-border:(rect[0]+rect[2]-border), rect[1]+border:(rect[1]+rect[3]-border)]
  croppedImage = image[topleftY-border:topleftY+height+border, topleftX:topleftX+width+border]

  # Convert the image to a uint8 array that opencv can handle
  imgRescaled = sitk.RescaleIntensity(croppedImage, sitk.sitkUInt8)
  imgArray = sitk.GetArrayFromImage(imgRescaled)
  imgArray = imgArray.astype('uint8')

  # Smoothing
  #imgArray = cv2.medianBlur(imgArray,5)
  # Negative
  #imgArray = 255 - imgArray
  plota(imgArray)
  cl(); canny = cv2.Canny(th, 30, 30, 3);plota(canny)

  # circles = cv2.HoughCircles(croppedImage,cv2.CV_HOUGH_GRADIENT,1,80,
  # param1=10,param2=5,minRadius=3,maxRadius=10)
  circles = cv2.HoughCircles(th, cv2.cv.CV_HOUGH_GRADIENT, 1,600, param1=30,param2=5,minRadius=10,maxRadius=100)

  cimg = imgArray

  if not circles is None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      # draw the outer circle
      cv2.circle(cimg,(i[0]+topleftY+border,i[1]+topleftY+border),i[2],(0,255,0),2)
      # draw the center of the circle
      cv2.circle(cimg,(i[0]+topleftX+border,i[1]+topleftX+border),2,(0,0,255),3)
    #cv2.imwrite(outputName, cimg)
    plota(cimg)

def displayResults(originalImage, labelmap, resultImage, imageName='', labelmapName='', skeletonImage=None, iterations=0, curvature=0, propagation=0):
  # Display results
  fig, axes = plt.subplots(nrows=1, ncols=2)

  # Show original image and the starting labelmap
  castedImage = sitk.Cast(sitk.RescaleIntensity(originalImage), labelmap.GetPixelID())
  im = sitk.LabelOverlay(castedImage, labelmap)
  #axes[0].imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(originalImage, labelmap > 0)))
  axes[0].imshow(sitk.GetArrayFromImage(im))
  axes[0].set_title('Seed')

  # Show original image and the result of the levelset
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
# seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
# seg.CopyInformation(image)
# seg[idx] = 1
# seg = sitk.BinaryDilate(seg, 3)


# sega = sitk.GetArrayFromImage(seg)
#print(stats)
#


#dataFolder = '/Data/jonieva/tempdata/'
#imageName='12257B_INSP_STD_UIA_COPD.nhdr'
#labelmapName='12257B_INSP_STD_UIA_COPD-label.nrrd'
#slice=448

imageName='/Volumes/Mac500/Dropbox/ACIL-Biomarkers/BiomarkerHackathon/10002K_INSP_STD_BWH_COPD_PulmonaryArteryBifurcationAxial_slice.nhdr'
slice=0


originalImage = sitk.ReadImage(imageName)[:,:,0]
originalLabelmap = generateSeed(originalImage)
#imgs = [originalImage, originalLabelmap, sitk.GetArrayFromImage(originalImage), sitk.GetArrayFromImage(originalLabelmap)]

# Read the image and the label map used as a seed. It returns: (image, labelmap, imageArray, labelmapArray)
#imgs = readImages(imageName, labelmapName, dataFolder, slice)
#originalImage = imgs[0]
# Set manually a value as a seed. I don't know why it's not saved in Slicer
#imgs[1][290,240] = 1
#originalLabelmap = imgs[1]


# Preprocess the image
image = preprocessImage(originalImage)
#labelmap = growSeed(image, [(290,240)])
#labelmap = growSeed(image, originalLabelmap)
labelmap = originalLabelmap




factor = 1.5
iterations=100
maximumRMSError = 0.02

curvatureScaling = 1
propagationScaling = 1

# Set thresholds for a bigger/smaller segmented area
# stats = getImageStats(image, labelmap)
# lower_threshold = stats.GetMean(1)-factor*stats.GetSigma(1)
# upper_threshold = stats.GetMean(1)+factor*stats.GetSigma(1)
lower_threshold = 0
upper_threshold = 300


#def getCirclesGerman(image):
  #originalImageArray = sitk.GetArrayFromImage(image)
 # img = np.squeeze(originalImageArray).T
def adaptIntensities(image):
  ''' Adapt the intensities of an image and convert to 255
  :param image:
  :return:
  '''
#img = sitk.GetArrayFromImage(originalImage)
  image[image <= -150] = -150;
  image[ image >= 150] = 150;
  image = 254*(image.astype(np.float32)-150)/300
  image = image.astype(np.uint8)

  return image

def detectAorta(image, seedX, seedY):
  ''' Detect aorta
  :param image: whole image in 255 gray levels
  :param seedY: y coord of a seed inside the circle
  :param seedX: x coord of a seed inside the circle
  :return:
  '''

  image = cv2.medianBlur(image,5)

  #canny = cv2.Canny(img, 100, 100); plota(canny)

  circles = cv2.HoughCircles(image,cv2.cv.CV_HOUGH_GRADIENT,1,100,param1=100,param2=10,minRadius=10,maxRadius=60)




  cimg = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
  if circles.any():

    #circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      d = math.sqrt(pow(i[0]-seedY,2) + pow(i[1]-seedX,2))
      if d < i[2]:
        # The circles contains the seed. This is the right one




      # draw the outer circle
      cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
      # draw the center of the circle
      cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)


plota(cimg)
# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# Before the levelSet, construct a small circle with a BinaryDilate operation. Otherwise the results of the levelset are weird
#labelmap = sitk.BinaryDilate(labelmap, 3)

#resultImage = applyLevelSet(image,labelmap,curvatureScaling, propagationScaling, iterations, maximumRMSError, lower_threshold, upper_threshold)
#displayResults(image, labelmap, resultImage, 'My image', 'My labelmap', None, iterations, curvatureScaling, propagationScaling)
#displayResults(image, labelmap, resultImage, 'My image', 'My labelmap', None, iterations, curvatureScaling, propagationScaling)

#circleDetectorFilter = sitk.co









# rows = 2
# cols = 2
#
# p.ioff()
#
# images = [[0 for i in range(rows)] for j in range(cols)]
# lsFilter.SetNumberOfIterations(500)
# img_T1f = sitk.Cast(image, sitk.sitkFloat32)
# ls = init_ls
# niter = 0
# fig, axes = p.subplots(nrows=rows, ncols=cols)
# for i in range(rows):
#   for j in range(cols):
#     ls = lsFilter.Execute(ls, img_T1f)
#     niter += lsFilter.GetNumberOfIterations()
#     t = "LevelSet after "+str(niter)+" iterations and RMS "+str(lsFilter.GetRMSChange())
#     #t = "I %i; RMS %f" % (niter, lsFilter.GetRMSChange())
#     #print(t)
#     #fig = myshow3d(sitk.LabelOverlay(img_T1_255, ls > 0), zslices=range(idx[2]-zslice_offset,idx[2]+zslice_offset+1,zslice_offset), dpi=20, title=t)
#     images[i][j] = sitk.GetArrayFromImage(sitk.LabelOverlay(image, ls > 0))
#     axes[i,j].set_title(t)
#     axes[i,j].imshow(images[i][j])
#     p.pause(0.1)
    #plt.pause(0.1)

# plt.switch_backend('Qt4Agg')


# for i in range(rows):
#   for j in range(cols):
#     axes[i,j].imshow(images[i][j])
#     p.pause(0.1)











