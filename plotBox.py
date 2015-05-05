

__author__ = 'jonieva'

# from lxml import etree
import SimpleITK as sitk
import numpy as np
import cip_python.utils.geometry_topology_data as geom
from scipy import misc
import os.path as path
import argparse

def displayPng(caseFullPath, caseXmlFullPath, structureCode='all', outputFolderFullPath=''):
    """ Display a bounding box for the specified structure ('all' will create a png for each structure).
    :param caseFullPath:
    :param caseXmlFullPath:
    :param outputFolderFullPath: if blank, the image will be saved in the same directory as the current folder
    :param structureCode:
    :return:
    """

    geometryTopologyData = geom.GeometryTopologyData.fromFile(caseXmlFullPath)

    # with open(caseXmlFullPath) as f:
    #     xml = f.read()
    #
    # root = etree.fromstring(xml)

    boundingBoxes = []
    boundingBoxesSizes = []
    structures = []

    if structureCode != 'all':
        for bb in geometryTopologyData.boundingBoxes:
            if bb.description == structureCode:
                # Structure found. Get the coordinates
                boundingBoxes.append(map(int, bb.start))
                boundingBoxesSizes.append(map(int, bb.size))
                structures.append(structureCode)
                break
    else:
        for bb in geometryTopologyData.boundingBoxes:
            # Structure found. Get the coordinates
            boundingBoxes.append(map(int, bb.start))
            boundingBoxesSizes.append(map(int, bb.size))
            structures.append(bb.description)

    # Read the image into a numpy array
    image = sitk.ReadImage(caseFullPath)
    # Convert to colors (required by the bounding box)
    imageColor = sitk.ScalarToRGBColormap(image)
    # Convert to numpy array
    imageColorArray = sitk.GetArrayFromImage(imageColor)
    # Get the id of the case
    caseId = path.splitext(path.basename(caseFullPath))[0]

    for i in range(len(boundingBoxes)):
        # Check the plane that is going to be saved
        if boundingBoxesSizes[i][2] == 0:
            # Axial
            imageSlice = imageColorArray[boundingBoxes[i][2], :, :, :].copy()
            flipUd = False
            dim1 = 0
            dim2 = 1
        elif boundingBoxesSizes[i][1] == 0:
            # Coronal
            imageSlice = imageColorArray[:, boundingBoxes[i][1], :, :].copy()
            flipUd = True
            dim1 = 0
            dim2 = 2
        elif boundingBoxesSizes[i][0] == 0:
            # Sagittal
            imageSlice = imageColorArray[:, :, boundingBoxes[i][0], :].copy()
            flipUd = True
            dim1 = 1
            dim2 = 2
        else:
            raise Exception("Corrupt structure: " + structures[i])

        # "Draw" the bounding box
        imageSlice[boundingBoxes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1]+boundingBoxesSizes[i][dim1], 0] = 255
        imageSlice[boundingBoxes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1]+boundingBoxesSizes[i][dim1], 1] = 0
        imageSlice[boundingBoxes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1]+boundingBoxesSizes[i][dim1], 2] = 0

        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 0] = 255
        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 1] = 0
        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 2] = 0

        imageSlice[boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 0] = 255
        imageSlice[boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 1] = 0
        imageSlice[boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1]:boundingBoxes[i][dim1] + boundingBoxesSizes[i][dim1], 2] = 0

        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1], 0] = 255
        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1], 1] = 0
        imageSlice[boundingBoxes[i][dim2]:boundingBoxes[i][dim2] + boundingBoxesSizes[i][dim2], boundingBoxes[i][dim1], 2] = 0

        if flipUd:
            # Flip the image (SimpleITK reads the coordinates in different order)
            imageSlice = np.flipud(imageSlice)

        # Save image in outputFolderPath
        outputFileName = path.join(outputFolderFullPath, "{0}_{1}.png".format(caseId, structures[i]))

        misc.imsave(outputFileName, imageSlice)


if __name__ == '__main__':
    # Parse the input parameters
    parser = argparse.ArgumentParser(prog='plotBox', description="Generate png files with different structures bounded")
    parser.add_argument("-f", dest="caseFullPath", required=True, metavar='caseFullPath', help="Full path to the volume (nrrd file or header)")
    parser.add_argument("-x", dest="caseXmlFullPath", required=True, metavar='caseXmlFullPath', help="Full path to the xml file that contains the detected structures")
    parser.add_argument("-s", dest="structureCode", required=True, metavar='structureCode', help="Structure text code (all = all the available structures)")
    parser.add_argument("-o", dest="outputFolder", metavar='outputFolder', default='', help="Folder where the images are saved (default: current directory)")

    args = parser.parse_args()
    displayPng(args.caseFullPath, args.caseXmlFullPath, args.structureCode, args.outputFolder)


