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
'''

import nrrd
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


class PAALogic:
def __init__(self):
    pass

  def extractAxialSlice(self, filePath, sliceNumber):
    ct_data, imageInfo= nrrd.read(filePath)

    # create a new ct file with that slice number
    ct_data_slice = ct_data[:,:,sliceNumber:sliceNumber+1]
    outputFilePath = '%s_%i.nrrd' % (os.path.splitext(filePath)[0], sliceNumber)
    nrrd.write(outputFilePath, ct_data_slice)

  def process(self, inputImagePath):
    image = sitk.ReadImage(inputImagePath)
    # reader = sitk.ImageFileReader()
    # reader.SetFileName (inputImagePath)
    # image = reader.Execute()
    # image = sitk.ReadImage(inputImagePath)
    nda = sitk.GetArrayFromImage(image)
    plt.imshow(nda)









p = PAALogic()
p.process('/Users/Jorge/tempdata/10766M_INSP_STD_NJC_COPD_493.png')