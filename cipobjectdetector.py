'''
   cipObjectDetector - class to perform object detection using OpenCV in medical images
   inputs:  volume image (tested with nrrds)
3            xml file describing the detector, created with opencv_traincascade and added the location constraints and the window level in whom to perform the detection
   outputs: coordinates: format: (x1, y1, z1, x2, y2, z2)
            detections:  format: (nimage, u, v, uw, vh)
            image with the detection
'''



import cv2
#import cv2.cv as cv
import numpy as np
import SimpleITK as sitk
import time
import os
from   sklearn.cluster import DBSCAN
#import matplotlib.pyplot as plt
from xml.dom import minidom


# TODO: squeeze or enlarge the image if still there is no detection
# TODO: make a version with Opencv 3.0, so that the weights of the detections are taken into account when clustering
# TODO: generate testing code for the class




class cipObjectDetector:
    """Class to perfrom object detection."""

    def __init__(self, verbose=False):

        # Data stored into the detector
        self.cascade           = None  # OpenCV casacade enhanced with 3D det params
        self.volume            = None  # Resampled numpy volume [voxelEdgeSizeXvoxelEdgeSizeXvoxelEdgeSize]
        self.sitkImage         = None  # Resampled sitk image as previous
        self.volumeOriginal    = None  # Original numpy volume
        self.sitkImageOriginal = None  # Original sitkImage
        self.verbose           = verbose

        # Output of the detection
        self.dets                 = None  # Storage of all the detections [layer, u, v, uw, vl]
        self.detectionCluster     = None  # Storage of the detections of the selected cluster: [layer, u, v, uw, vl]
        self.detection            = None  # The final detection in format [layer, u, v, uw, vl]
        self.detectionCoordinates = None  # The final detection in format [x1,y1,z1,x2,y2,z2]
        self.objectCoordinates    = None  # The final detection without the detection margins

        # Parameters of the detector - obtained from the cascade
        self.minDetectionLimits   = None #
        self.maxDetectionLimits   = None
        self.direction            = None # One of 'axial', 'coronal', 'sagittal'
        self.voxelEdgeSize        = 0.65 # One float for isotropic voxels
        self.pixelConversionFactor = np.array([1,1,1])
        self.limMin      = -200         # Mini value when converting to uchar
        self.limMax      =  200         # Idem but max
        self.scaleFactor =    4         # Downsampling factor of the image:
        self.canonicalSampleWidth    = 130 # In the upsampled image
        self.canonicalSampleHeight   = 260
        self.useLocationConstraints  = True
        self.locationConstraintsType = "x1y1z1x2y2z2"
        self.locationConstraintsMargin = 0.0
        self.alwaysPresent = 1
        self.referenceStandardMargins = np.array([0,0,0])

        # Paths of the main object
        self.cascadePath = None    # Name of the cascade
        self.volumePath  = None     # Name of the volume to detect

    def SITKLoadNRRD(self, volumeName, verbose=False, outputIsotropicVoxelSize=0.65):
        # Loads the volume and turns it into an numpy array
        start = time.time()
        if verbose:
            print("Loading: " + volumeName)
        stkimage  = sitk.Cast(sitk.ReadImage( volumeName ), sitk.sitkInt16)
        voloriginalt  = self.npyArrayFromSITKImage(stkimage)
        end = time.time()
        if verbose:
            print("Loading time %f" % (end - start))

        return (voloriginalt, stkimage)

    def ResampleSITKImage(self, sitkImageIn, outputIsotropicVoxelSize=0.65, verbose=False):
        # Resamples the volume
        start = time.time()
        pixelConversionFactor = np.array([1,1,1])
        ovSize        = outputIsotropicVoxelSize
        imageSpacing  = np.array(sitkImageIn.GetSpacing()).astype(float)
        imageSize     = np.array(sitkImageIn.GetSize()).astype(float)
        outputSize    = np.round(imageSize*imageSpacing/ovSize).astype(int)
        outputSpacing = np.array([ovSize,ovSize,ovSize])
        resampler     = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(sitkImageIn.GetDirection())
        resampler.SetOutputOrigin   (sitkImageIn.GetOrigin())
        resampler.SetSize           ( outputSize )
        resampler.SetOutputSpacing  (outputSpacing)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        pixelConversionFactor = imageSpacing / ovSize
        stkimageOutput = resampler.Execute(sitkImageIn)
        volt = self.npyArrayFromSITKImage(stkimageOutput)
        end = time.time()
        if verbose:
            print("Resampling time %f" % (end - start))
        return (volt, stkimageOutput, pixelConversionFactor)

    # Convenient data transformation functions
    def SITKImageFromNpyArray(self, npArray, sitkImageTemplate=None):
        sitkImage = sitk.GetImageFromArray(npArray.astype(np.int16).transpose([2,0,1]))
        if sitkImageTemplate != None:
            sitkImage.SetOrigin    (sitkImageTemplate.GetOrigin()   )
            sitkImage.SetDirection (sitkImageTemplate.GetDirection())
            sitkImage.SetSpacing   (sitkImageTemplate.GetSpacing()  )
        return sitkImage

    def npyArrayFromSITKImage(self, sitkImage):
        npyArray = sitk.GetArrayFromImage(sitkImage)
        npyArray = npyArray.transpose([1,2,0])
        return npyArray



    def getImageCoordinatesFromDetectionCoordinates(self,
            rectangleCoordinates, inOriginalImage = False):
        """Returns [x1 y1 z1 x2 y2 z2] from [z, u, v, uw, vh]"""
        assert rectangleCoordinates.size >= 5, "ObjectDetection::getImageCoordinatesFromDetectionCoordinates: the rectangle has less than 5 coordinates"
        (w, u, v, ul, vl) = rectangleCoordinates
        img = self.getImage(w)
        (hi,wi) = img.shape
        if self.direction == 'axial':
            x1 = u;    y1 = v;     z1=w;
            x2 = u+ul; y2 = v+vl;  z2=w;
        elif self.direction == 'coronal':
            x1=u;     y1=w; z1=hi-v;
	    x2=u+ul;  y2=w; z2=hi-v-vl;
        elif self.direction == 'sagittal':
            x1=w; y1=u;     z1=hi-v;
            x2=w; y2=u+ul;  z2=hi-v-vl;
        if inOriginalImage:
            coords = np.round(np.array([x1/self.pixelConversionFactor[0], y1/self.pixelConversionFactor[1], \
                z1/self.pixelConversionFactor[2], x2/self.pixelConversionFactor[0],\
                y2/self.pixelConversionFactor[1], z2/self.pixelConversionFactor[2]]))
        else:
            coords = np.round(np.array([x1,y1,z1,x2,y2,z2]))
        # Get min to be min and max to be max in case something is screwed up
        for i in range(0,3):
            a =  coords[i]; b = coords[i+3];
            coords[i]   = np.min((a,b))
            coords[i+3] = np.max((a,b))
        return coords

    def getObjectCoordinatesFromImageCoordinates(self, imageCoordinates):
        """Returns [x1o y1o z1o x2o y2o z2o] from [x1 y1 z1 x2 y2 z2]. Removes the detection margins"""
        [x1,y1,z1,x2,y2,z2] = imageCoordinates
        assert (x2 >= x1) and (y2 >= y1) and (z2 >= z1), "cipObjectDetector::getObjectCoordinatesFromImageCoordinates:: error in input coordinates"
        if self.direction == 'axial':
            z1o = z1; z2o = z2;
            x1o = x1 + self.referenceStandardMargins[0]*float(x2-x1)/self.canonicalSampleWidth
            x2o = x2 - self.referenceStandardMargins[0]*float(x2-x1)/self.canonicalSampleWidth
            y1o = y1 + self.referenceStandardMargins[1]*float(y2-y1)/self.canonicalSampleHeight
            y2o = y2 - self.referenceStandardMargins[1]*float(y2-y1)/self.canonicalSampleHeight
        elif self.direction == 'coronal':
            y1o = y1; y2o = y2;
            x1o = x1 + self.referenceStandardMargins[0]*float(x2-x1)/self.canonicalSampleWidth
            x2o = x2 - self.referenceStandardMargins[0]*float(x2-x1)/self.canonicalSampleWidth
            z1o = z1 + self.referenceStandardMargins[2]*float(z2-z1)/self.canonicalSampleHeight
            z2o = z2 - self.referenceStandardMargins[2]*float(z2-z1)/self.canonicalSampleHeight
        elif self.direction == 'sagittal':
            x1o = x1; x2o = x2;
            y1o = y1 + self.referenceStandardMargins[1]*float(y2-y1)/self.canonicalSampleWidth
            y2o = y2 - self.referenceStandardMargins[1]*float(y2-y1)/self.canonicalSampleWidth
            z1o = z1 + self.referenceStandardMargins[2]*float(z2-z1)/self.canonicalSampleHeight
            z2o = z2 - self.referenceStandardMargins[2]*float(z2-z1)/self.canonicalSampleHeight
        return np.round(np.array([x1o,y1o,z1o,x2o,y2o,z2o])).astype(np.int)

    def getDetectionCoordinatesFromImageCoordinates(self, imageCoordinates, inOriginalImage = False):
        """Returns [z u1 v1 u2 v2] from [x1 y1 z1 x2 y2 z2]"""
        [x1,y1,z1,x2,y2,z2] = imageCoordinates
        assert (x2 >= x1) and (y2 >= y1) and (z2 >= z1), "cipObjectDetector::getDetectionCoordinatesFromImageCoordinates:: error in input coordinates"
        u1=0; u2 = 0; v1=0; v2 = 0;
        if self.direction == 'axial':
            z = z1
            u1 = x1;     v1 = y1;
            u2 = x2;     v2 = y2;
        elif self.direction == 'coronal':
            z = y1;
            if inOriginalImage:
                img = self.getOriginalImage(z)
            else:
                img = self.getImage(z)
            (hi,wi) = img.shape
            u1 = x1;     v1 = hi-z2;
	    u2 = x2;     v2 = hi-z1;
        elif self.direction == 'sagittal':
            z = x1;
            if inOriginalImage:
                img = self.getOriginalImage(z)
            else:
                img = self.getImage(z)
            (hi,wi) = img.shape
            u1 = y1; v1=hi-z2;
            u2 = y2; v2=hi-z1;
        assert (u2 >= u1) and (v2 >= v1), "cipObjectDetector::getDetectionCoordinatesFromImageCoordinates:: error in output coordinates"
        return np.array([z,u1, v1, u2, v2], dtype=int)


    def loadAttributeFromXML(self, xmldoc, attributeName):
        """Auxiliary function to load an attribute from the XML"""
        toReturn = ""
        try:
            toReturn = xmldoc.getElementsByTagName(attributeName)[0].childNodes[0].data
        except:
            print("cipObjectDetector::Error loading %s" % attributeName)
        return toReturn

    # Loads the detector and the parameters of the detection from the xml file
    def loadCascade(self, cascadePath):
        """Gets the information from the trained XML"""
        if cascadePath == None:
            return

        if self.verbose:
            print("cipObjectDetector:: loading cascade %s" % cascadePath)

        # Loads the cascade using opencv parsing
        self.cascadePath = cascadePath
        self.cascade     = cv2.CascadeClassifier(cascadePath)

        # Added attributes from the XML
        xmldoc = minidom.parse(cascadePath)
        self.limMin        = float(self.loadAttributeFromXML(xmldoc, 'minVal'))
        self.limMax        = float(self.loadAttributeFromXML(xmldoc, 'maxVal'))
        self.scaleFactor   = int(self.loadAttributeFromXML(xmldoc, 'scaleFactor'))
        self.direction     = self.loadAttributeFromXML(xmldoc, 'direction')
        self.minDetectionLimits = np.fromstring(self.loadAttributeFromXML(xmldoc, 'minDetectionLimits').replace("\"",""), dtype=float, sep=',')
        self.maxDetectionLimits = np.fromstring(self.loadAttributeFromXML(xmldoc, 'maxDetectionLimits').replace("\"",""), dtype=float, sep=',')
        self.canonicalSampleWidth  = int(self.loadAttributeFromXML(xmldoc, 'canonicalSampleWidth'))
        self.canonicalSampleHeight = int(self.loadAttributeFromXML(xmldoc, 'canonicalSampleHeight'))
        self.locationConstraintsType  = self.loadAttributeFromXML(xmldoc, 'locationConstraintsType')
        self.alwaysPresent            = int(float(self.loadAttributeFromXML(xmldoc, 'alwaysPresent')))
        self.referenceStandardMargins = np.fromstring(self.loadAttributeFromXML(xmldoc, 'referenceStandardMargins').replace("\"",""), dtype=float, sep=',')

        # If the pixel conversion factor has changed, the volume structures should be recomputed to accomodate for it
        voxelEdgeSize = float(self.loadAttributeFromXML(xmldoc, 'voxelEdgeSize'))
        if (voxelEdgeSize != self.voxelEdgeSize):
            self.voxelEdgeSize = voxelEdgeSize
            if self.sitkImageOriginal is not None:
                self.loadVolumeFromSITKImage(self.sitkImageOriginal)

        # Ensure that min is min and max is max in the constraints
        for i in range(0,6):
            a = np.min( (self.minDetectionLimits[i], self.maxDetectionLimits[i] ) )
            b = np.max( (self.minDetectionLimits[i], self.maxDetectionLimits[i] ) )
            self.minDetectionLimits[i] = a
            self.maxDetectionLimits[i] = b

        # Adds some margins (5% of the image) to the detection limits to be safe of not excluding the structure of interest
        for i in range(0,3):
            self.minDetectionLimits[i] = np.max((0,self.minDetectionLimits[i]-0.05))
            self.maxDetectionLimits[i] = np.min((1,self.maxDetectionLimits[i]+0.05))
            if self.locationConstraintsType == "x1y1z1x2y2z2":
                self.minDetectionLimits[i+3] = np.max((0,self.minDetectionLimits[i+3]-0.05))
                self.maxDetectionLimits[i+3] = np.min((1,self.maxDetectionLimits[i+3]+0.05))
            else:
                self.minDetectionLimits[i+3] = np.max((0,self.minDetectionLimits[i+3]-0.1))
                self.maxDetectionLimits[i+3] = np.min((1,self.maxDetectionLimits[i+3]+0.1))

        if self.verbose:
            print("cipObjectDetector:: cascade loaded")

    # Loads the volume from the file
    def loadVolumeFromPath(self, volumePath):
        if volumePath == None:
            return;
        self.volumePath = volumePath
        sitkImage = sitk.Cast(sitk.ReadImage( volumePath ), sitk.sitkInt16)
        self.loadVolumeFromSITKImage( sitkImage )
        if self.verbose:
            print("cipObjectDetector:: volume loaded")

    def loadVolumeFromSITKImage(self, sitkImageIn):
        self.sitkImageOriginal = sitkImageIn
        self.volumeOriginal    = self.npyArrayFromSITKImage(sitkImageIn)
        (npCTScan, self.sitkImage, self.pixelConversionFactor) = self.ResampleSITKImage(self.sitkImageOriginal, self.voxelEdgeSize)
        self.volume    = npCTScan.astype(np.float32)

    # Turns the image (whatever type) to uint 8 to perform the detection
    def imgToUint8(self, img):
        #assert (self.limMin != None) and (self.limMax != None), 'cipObjectDetector::imgToUint8 error with limMin and limMax"
        img = img.astype(float)
        img[img<self.limMin] = self.limMin
        img[img>self.limMax] = self.limMax
        img = 255*(img-self.limMin)/(self.limMax-self.limMin)
        image = img.astype(np.uint8)
        return image

    # Returns the image of slice i depending on the plane
    def getImage(self, i):
        if self.direction == 'axial':
            img   = self.volume[:,:,i]
        elif self.direction == 'coronal':
            img   = np.squeeze(self.volume[i,:,:])
            img   = np.flipud(np.transpose(img, [1,0]))
        elif self.direction == 'sagittal':
            img   = np.squeeze(self.volume[:,i,:])
            img   = np.transpose(np.fliplr(img), [1,0])
        return img

    def getOriginalImage(self, i):
        if self.direction == 'axial':
            img   = self.volumeOriginal[:,:,i]
        elif self.direction == 'coronal':
            img   = np.squeeze(self.volumeOriginal[i,:,:])
            img   = np.flipud(np.transpose(img, [1,0]))
        elif self.direction == 'sagittal':
            img   = np.squeeze(self.volumeOriginal[:,i,:])
            img   = np.transpose(np.fliplr(img), [1,0])
        return img

    # Given a volume, detect in each slice areas that look like the object to detect
    def findBoxes(self, numberOfNeighbors = 4):
        dets = np.array([])
        nDet = 0

        shpv = self.volume.shape
        if self.direction == 'axial':
            axis2 = [0,1,2]
        elif self.direction == 'coronal':
            axis2 = [0,0,1]
        elif self.direction == 'sagittal':
            axis2 = [0,0,0]

        # Analyzing all images on the volume has been proven to provide better detections
        # than doing it in sets of scaleFactor
        for i in range(1,self.volume.shape[axis2[2]], 1):

            # Not too up, nor too down
            if self.useLocationConstraints and \
                ((float(i)/shpv[axis2[2]] < self.minDetectionLimits[axis2[2]]) or \
                 (float(i)/shpv[axis2[2]] > self.maxDetectionLimits[axis2[2]])) :
                continue;

            img = self.getImage(i)
            shp = img.shape

            # Resize the image for fast detection
            img = cv2.resize(img, (shp[1]/self.scaleFactor, shp[0]/self.scaleFactor))

            # If the volume is not uint8, cast it using the clipping limits
            if self.volume.dtype != np.dtype(np.uint8):
                img = self.imgToUint8(img)

            # Detect and add them to the poll of detections
            rects = self.cascade.detectMultiScale(img, 1.05, numberOfNeighbors)
            if len(rects) == 0:
                continue

            # Analyze each detection - see notes of 20150317
            # Detections are given in u,v, ulength, vlength coordinates
            for rect in rects:
                (u,v,ul,vl) = rect
                un =u *self.scaleFactor; vn = v*self.scaleFactor;
                uln=ul*self.scaleFactor; vln=vl*self.scaleFactor;

                # Do not store images that are outside the location limits
                if self.useLocationConstraints:

                    imageCoords = self.getImageCoordinatesFromDetectionCoordinates(np.array([i,un,vn,uln,vln]), False)
                    # The location constraints are defined as object size, without the margins
                    (x1,y1,z1,x2,y2,z2) = self.getObjectCoordinatesFromImageCoordinates(imageCoords)

                    # Check that the first corner of the object is within range
                    if self.direction == 'axial':
                        if ( x1 < self.minDetectionLimits[0]*shpv[1] ) or \
                           ( x1 > self.maxDetectionLimits[0]*shpv[1] ) or \
                           ( y1 < self.minDetectionLimits[1]*shpv[0] ) or \
                           ( y1 > self.maxDetectionLimits[1]*shpv[0] ):
                                continue
                    if self.direction == "coronal":
                        if ( x1 < self.minDetectionLimits[0]*shpv[1] ) or \
                           ( x1 > self.maxDetectionLimits[0]*shpv[1] ) or \
                           ( z1 < self.minDetectionLimits[2]*shpv[2] ) or \
                           ( z1 > self.maxDetectionLimits[2]*shpv[2] ):
                                continue
                    if self.direction == 'sagittal':
                        if ( y1 < self.minDetectionLimits[1]*shpv[0] ) or \
                           ( y1 > self.maxDetectionLimits[1]*shpv[0] ) or \
                           ( z1 < self.minDetectionLimits[2]*shpv[2] ) or \
                           ( z1 > self.maxDetectionLimits[2]*shpv[2] ):
                                continue

                    # The location costraints can be stored in different manners
                    if self.locationConstraintsType == "x1y1z1x2y2z2":
                         if self.direction == 'axial':
                             if ( x2 < self.minDetectionLimits[3]*shpv[1] ) or \
                                ( x2 > self.maxDetectionLimits[3]*shpv[1] ) or \
     	                        ( y2 < self.minDetectionLimits[4]*shpv[0] ) or \
                                ( y2 > self.maxDetectionLimits[4]*shpv[0] ):
                                 continue
                         if self.direction == "coronal":
                             if ( x2 < self.minDetectionLimits[3]*shpv[1] ) or \
        	                ( x2 > self.maxDetectionLimits[3]*shpv[1] ) or \
                	        ( z2 < self.minDetectionLimits[5]*shpv[2] ) or \
                                ( z2 > self.maxDetectionLimits[5]*shpv[2] ):
                                 continue
                         if self.direction == 'sagittal':
                             if ( y2 < self.minDetectionLimits[4]*shpv[0] ) or \
	                        ( y2 > self.maxDetectionLimits[4]*shpv[0] ) or \
        	                ( z2 < self.minDetectionLimits[5]*shpv[2] ) or \
                                ( z2 > self.maxDetectionLimits[5]*shpv[2] ):
                                 continue
                    else: # We assume the else is x1y1z1sxsysz
                         sx = ( np.max((x1,x2)) - np.min((x1,x2)) )
                         sy = ( np.max((y1,y2)) - np.min((y1,y2)) )
                         sz = ( np.max((z1,z2)) - np.min((z1,z2)) )
                         if self.direction == 'axial':
                             if ( sx  < self.minDetectionLimits[3]*shpv[1] ) or \
                                ( sx  > self.maxDetectionLimits[3]*shpv[1] ) or \
                                ( sy  < self.minDetectionLimits[4]*shpv[0] ) or \
                                ( sy  > self.maxDetectionLimits[4]*shpv[0] ):
                                 continue
                         if self.direction == 'coronal':
                             if ( sx  < self.minDetectionLimits[3]*shpv[1] ) or \
                                ( sx  > self.maxDetectionLimits[3]*shpv[1] ) or \
                                ( sz  < self.minDetectionLimits[5]*shpv[2] ) or \
                                ( sz  > self.maxDetectionLimits[5]*shpv[2] ):
                                 continue
                         if self.direction == 'sagittal':
                             if ( sy  < self.minDetectionLimits[4]*shpv[0] ) or \
                                ( sy  > self.maxDetectionLimits[4]*shpv[0] ) or \
                                ( sz  < self.minDetectionLimits[5]*shpv[2] ) or \
                                ( sz  > self.maxDetectionLimits[5]*shpv[2] ):
                                 continue

                # Saves the surviving detections
                if nDet==0:
                    dets = np.array([i, un, vn, uln, vln])
                else:
                    dets= np.vstack( (dets, [i, un, vn, uln, vln] ))
                nDet = nDet + 1
        return dets

    # Finds the boxes, performs clustering
    def detect(self, cascadePath = None,
               useLocationConstraints = None):

        if self.verbose:
            print("cipObjectDetector:: performing the detection")

        tstart = time.time()
        if self.volume is None:
            print("cipObjectDetector::detect called without volume data")
            return([])

        if not cascadePath is None:
            self.loadCascade(cascadePath)

        if self.cascade is None:
            print("cipObjectDetector::detect: No cascade has been defined")
            return ([])

        if not useLocationConstraints is None:
            self.useLocationConstraints = useLocationConstraints

        self.dets = None

        # Is there any box detected
        # Normally starts at 4 - we want to get all the boxes. We will decide later.
        for ineig in range(4,-1,-1):
            self.dets = self.findBoxes(ineig) # detectionCandidates
            if self.dets.size != 0:
                break

        # If no box - make the same without location constraints - disabled for now
        # if (self.dets.size == 0) and (self.alwaysPresent == 1):
        if False:
            print("Could not find the object with location constraints, trying without")
            self.useLocationConstraints = False
            for ineig in range(4,-1,-1):
                self.dets = self.findBoxes(ineig) # detectionCandidates
                if self.dets.size != 0:
                    break
        tend = time.time()

        # If the detection fails, return a zero image and an empty detection
        if(self.dets is None):
            if self.verbose:
                print("Could not find the object")
            img = self.volume[:,:,0].astype(np.uint8)
            img[:,:] = 0
            self.detection = None
            self.detectionCluster = None
            self.detectionCoordinates = None
            self.objectCoordinates = None
        else:
            # If many detections - cluster them and find the average
            if self.dets.ndim > 1:
                db = DBSCAN(eps=40, min_samples=1).fit(self.dets)
                labels = db.labels_
                lbl = set(labels)
                smmx = 0
                lblmx = []
                for i in lbl:
                    sm = np.sum(labels==i)
                    if sm > smmx:
                        smmx = sm
                        lblmx = i
                idx = (labels == lblmx)
                dmt = self.dets[idx,:]
                self.detectionCluster = self.dets[idx,:]
                self.detection = np.mean(dmt,axis=0).round().astype(int)
            else:
                self.detection = self.dets
                # This is a terrible hack to prevent 1D self.dets!!
                self.dets = np.vstack( (self.dets, self.dets) )
                self.detectionCluster = self.dets

        if self.verbose:
            print("Processing time %f with nn=%i" % (tend - tstart, ineig))
        if not self.detection == None:
            self.detectionCoordinates = self.getImageCoordinatesFromDetectionCoordinates(
                            self.detection, True)
            self.objectCoordinates = self.getObjectCoordinatesFromImageCoordinates(self.detectionCoordinates )
        else:
            self.detectionCoordinates = None
            self.objectCoordinates = None
        return self.objectCoordinates


    def getDetectionAsNpyArray(self):
        """Returns an npy array with the result of the detection"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        (z, u1, v1, u2, v2) = self.getDetectionCoordinatesFromImageCoordinates(self.detectionCoordinates)
        img = self.getOriginalImage(z)
        imgCrop = img[ v1:v2, u1:u2 ]
        return imgCrop

    def getDetectionSliceAsNpyArray(self):
        """Returns an npy array with the slice where the detection happened"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        (z, u1, v1, u2, v2) = self.getDetectionCoordinatesFromImageCoordinates(self.detectionCoordinates)
        img = self.getOriginalImage(z)
        return img

    def getDetectionAsSITKImage(self,outputObjectWithMargins):
        """Returns an SITKImage with the result of the detection"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        if outputObjectWithMargins:
            dc = self.detectionCoordinates
        else:
            dc = self.objectCoordinates
        # Hack - for some reasons sometimes I was getting images 1 pixel too large
        #  I reduced the origin by one
        x1 = int(np.max( (np.min( (dc[0], dc[3]) )-1,0) ) )
        y1 = int(np.max( (np.min( (dc[1], dc[4]) )-1,0) ) )
        z1 = int(np.max( (np.min( (dc[2], dc[5]) )-1,0) ) )
        xs = int(np.max((dc[0], dc[3])))-int(np.min((dc[0], dc[3])))+1
        ys = int(np.max((dc[1], dc[4])))-int(np.min((dc[1], dc[4])))+1
        zs = int(np.max((dc[2], dc[5])))-int(np.min((dc[2], dc[5])))+1
        #print([x1,y1,z1,x1+xs,y1+ys,z1+zs])
        #print self.sitkImageOriginal.GetSize()
        sitkImageCrop = sitk.RegionOfInterest(self.sitkImageOriginal, [xs, ys ,zs ], [x1, y1, z1])
        return sitkImageCrop

    def getDetectionSliceAsSITKImage(self, includeMargins=True):
        """Returns an SITKImage with the slice where the detection happened"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        if includeMargins:
            dc = self.detectionCoordinates
        else:
            dc = self.objectCoordinates

        sz = self.sitkImageOriginal.GetSize()
        xs = sz[0]; ys = sz[1]; zs = sz[2]
        x1 = 0; y1 = 0; z1 = 0;
        if self.direction == 'axial':
            zs  = 1;
            z1  = int(dc[2])
        if self.direction == 'coronal':
            ys = 1
            y1 = int(dc[1])
        if self.direction == 'sagittal':
            xs = 1
            x1 = int(dc[0])

        sitkImageCrop = sitk.RegionOfInterest(self.sitkImageOriginal, [xs, ys, zs], [x1, y1, z1])
        return sitkImageCrop

    def writeDetectionSliceAsPng(self, outputName):
        """outputName: where to store it, outputCanonicalImage: resample to the predefined size"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        (z, u1, v1, u2, v2) = self.getDetectionCoordinatesFromImageCoordinates(self.detectionCoordinates,True)
        img = self.getOriginalImage(z)
        imgu = self.imgToUint8(img)
        cimg = cv2.cvtColor(imgu,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(cimg, (u1, v1 ), ( u2,v2 ) , (127, 255, 0), 3)
        cv2.imwrite(outputName, cimg);

    # Outputs a png with the detection
    def writeDetectionAsPng(self, outputName, outputCanonicalImage = False, outputObjectWithMargins = False):
        """outputName: where to store it, outputCanonicalImage: resample to the predefined size"""
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return

        if outputObjectWithMargins:
            (z, u1, v1, u2, v2) = self.getDetectionCoordinatesFromImageCoordinates(self.detectionCoordinates,True)
        else:
            (z, u1, v1, u2, v2) = self.getDetectionCoordinatesFromImageCoordinates(self.objectCoordinates,True)
        img = self.getOriginalImage(z)
        imgu = self.imgToUint8(img)
        cimg = cv2.cvtColor(imgu,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(cimg, (u1, v1 ), ( u2,v2 ) , (127, 255, 0), 3)
        cimgout  = cimg[ int(np.min([v1,v2])):int(np.max([v1,v2])), int(np.min([u1,u2])):int(np.max([u1,u2])) ,:]
        if outputCanonicalImage:
            cimgout = cv2.resize(cimgout,(self.canonicalSampleWidth, self.canonicalSampleHeight))
        if self.verbose:
            print outputName
        cv2.imwrite(outputName, cimgout);

    # Saves all the detections produced by the cascade - useful for debug purposes
    def writeAllDetectionsAsPng(self, outputName):
        if self.detection == None:
            print("Object detection: called outputDetection, but nothing has been detected");
            return
        if self.volume == None:
            print("Object detection: called outputDetection, but there is no volume");
            return
        nDet = 0
        shpv = self.volume.shape
        for det in self.dets:
            (i,un,vn,uln,vln) = det
            nmout = outputName % (nDet)
            nDet = nDet+1
            imgf  = self.getImage(i)
            imgfu = self.imgToUint8(imgf)
            cimgfu = cv2.cvtColor(imgfu,cv2.COLOR_GRAY2BGR)

            # Draws the rectangle of the detection
            cv2.rectangle(cimgfu, (un,vn), \
                          (un+uln,vn+vln), (127, 255, 0), 1)
            # Draws the object within the rectangle of the detection
            imageCoordinates  = self.getImageCoordinatesFromDetectionCoordinates(det, False)
            objectCoordinates = self.getObjectCoordinatesFromImageCoordinates   (imageCoordinates)
            [w,u1,v1,u2,v2]   = self.getDetectionCoordinatesFromImageCoordinates(objectCoordinates, False)
            cv2.rectangle(cimgfu, (u1,v1),(u2,v2), (127, 255, 0), 3)

            # Draws the detection constraints - where the corner should be
            mdl = self.minDetectionLimits; Mdl = self.maxDetectionLimits
            [w,u1m,v1m,u2m,v2m] = self.getDetectionCoordinatesFromImageCoordinates(
                    [mdl[0]*shpv[1], mdl[1]*shpv[0], mdl[2]*shpv[2],
                     Mdl[0]*shpv[1], Mdl[1]*shpv[0], Mdl[2]*shpv[2]], False)
            cv2.rectangle(cimgfu, (u1m,v1m), (u2m, v2m), (255,127,0),3)

            cv2.imwrite(nmout , cimgfu);

    def writeDetectionAsNRRD(self, outputName, outputCanonicalImage = False, outputObjectWithMargins = False):
        """outputName: where to store it, outputCanonicalImage: resample to the predefined size"""
        sitkImageCrop = self.getDetectionAsSITKImage(outputObjectWithMargins)
        if outputCanonicalImage == False:
            # sitk.WriteImage(sitk.Cast(sitkImageCrop, sitk.sitkInt16),
                            # outputName, useCompression=True)
            sitk.WriteImage(sitk.Cast(sitkImageCrop, sitk.sitkInt16),
                            outputName)
        else:
            #ovSize       = outputIsotropicVoxelSize
            imageSpacing = np.array(sitkImageCrop.GetSpacing()).astype(float)
            imageSize    = np.array(sitkImageCrop.GetSize()).astype(float)

            if self.direction == 'axial':
                outputSize    = np.array([self.canonicalSampleWidth, self.canonicalSampleHeight, 1]).astype(int)
            if self.direction == 'coronal':
                outputSize    = np.array([self.canonicalSampleWidth, 1, self.canonicalSampleHeight]).astype(int)
            if self.direction == 'sagittal':
                outputSize    = np.array([1, self.canonicalSampleWidth, self.canonicalSampleHeight]).astype(int)

            outputSpacing = np.array(imageSpacing*imageSize/outputSize)

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(sitkImageCrop.GetDirection())
            resampler.SetOutputOrigin   (sitkImageCrop.GetOrigin())
            resampler.SetSize( outputSize )
            resampler.SetOutputSpacing(outputSpacing)
            resampler.SetOutputPixelType(sitk.sitkFloat32)

            #pixelConversionFactor = imageSpacing / ovSize
            stkimageOutput = resampler.Execute(sitkImageCrop)
            # sitk.WriteImage(sitk.Cast(stkimageOutput, sitk.sitkInt16),
                            # outputName, useCompression=True)
            sitk.WriteImage(sitk.Cast(stkimageOutput, sitk.sitkInt16),
                            outputName)


    def writeDetectionSliceAsNRRD(self, outputName):
        sitkImageCrop = self.getDetectionSliceAsSITKImage()
        # sitk.WriteImage(sitk.Cast(sitkImageCrop, sitk.sitkInt16),
                        # outputName, useCompression=True)
        sitk.WriteImage(sitk.Cast(sitkImageCrop, sitk.sitkInt16),
                        outputName)


    def writeDetectionCoordinatesAsCSV(self, outputName):
        if self.detectionCoordinates is None:
            return
        f = open(outputName,'w')
        dc = self.detectionCoordinates
        f.write('%i,%i,%i,%i,%i,%i' % (dc[0], dc[1], dc[2],
                dc[3], dc[4], dc[5]))
        f.close()

    def writeAllDetectionsAsCSV(self, outputName):
        if self.dets is None:
            return
        f = open(outputName,'w')
        for dc in self.dets:
            f.write('%i,%i,%i,%i,%i\n' % (dc[0], dc[1], dc[2],
                    dc[3], dc[4]))
        f.close()

    def writeAllDetectionsClusterAsCSV(self, outputName):
        if self.detectionCluster is None:
            return
        f = open(outputName,'w')
        for dc in self.detectionCluster:
            f.write('%i,%i,%i,%i,%i\n' % (dc[0], dc[1], dc[2],
                    dc[3], dc[4]))
        f.close()

    def writeAllDetectionsCoordinatesAsCSV(self, outputName):
        if self.dets is None:
            return
        f = open(outputName,'w')
        for dc in self.dets:
            cd = self.getImageCoordinatesFromDetectionCoordinates(
                        dc, True)
            f.write('%i,%i,%i,%i,%i,%i\n' % (cd[0], cd[1], cd[2],
                    cd[3], cd[4], cd[5]))
        f.close()

    def writeAllDetectionsClusterCoordinatesAsCSV(self, outputName):
        if self.detectionCluster is None:
            return
        f = open(outputName,'w')
        for dc in self.detectionCluster:
            cd = self.getImageCoordinatesFromDetectionCoordinates(
                        dc, True)
            f.write('%i,%i,%i,%i,%i,%i\n' % (cd[0], cd[1], cd[2],
                    cd[3], cd[4], cd[5]))
        f.close()



if __name__ == "__main__":
    dataDir = '/Users/predout/Data/COPDGene/CTs/'
    #dataDir = '/Volumes/PE/COPDGene/CTs/'
    outputDir = '/Users/predout/Data/detections2'

    detectors = [\
                '/Users/predout/opt.source/acil/acil_python/ObjectDetection/Detectors/ChestCTPulmonaryArteryAxial.xml'
                ]

    oprefix = ['ChestCTPulmonaryArtery']

    volumeNames = [\
        #'10002K_INSP_STD_BWH_COPD',\
        '10004O_INSP_STD_BWH_COPD',\
    ]

    for volumeName in volumeNames:
        volumePath = os.path.join(dataDir, volumeName + '.nhdr')
        nVl = volumeName[0:6]

        # Constructs the detector with a given volume
        detector = cipObjectDetector(verbose = True )

        # Load the data from the volumePath or from the sitkImage
        #detector.loadVolumeFromPath(volumePath)
        sitkImage = sitk.Cast(sitk.ReadImage( volumePath ), sitk.sitkInt16)
        detector.loadVolumeFromSITKImage( sitkImage )


        for (prefix, detectorName) in zip(oprefix, detectors):
            # Detect whatever it is

            start = time.time()
            detector.loadCascade(detectorName)
            (coords) = detector.detect(useLocationConstraints=True )
            end = time.time()
            print("Detection time %f" % (end - start))

            start = time.time()
            oName           = outputDir  + '/' + prefix + '_' + nVl + '.png'
            oNameSlicePng   = outputDir  + '/' + prefix + '_' + nVl + 'slice.png'
            oNameCrop       = outputDir  + '/' + prefix + '_' + nVl + '_crop.png'
            oNameAll        = outputDir  + '/all/' + prefix + '_' + nVl + '_%04i.png'
            oNameNrrd       = outputDir  + '/' + prefix + '_' + nVl + '.nrrd'
            oNameNrrdCrop   = outputDir  + '/' + prefix + '_' + nVl + '_crop.nrrd'
            oNameNrrdCropCan = outputDir  + '/' + prefix + '_' + nVl + '_cropCan.nrrd'
            oNameNrrdCropNM   = outputDir  + '/' + prefix + '_' + nVl + '_cropNM.nrrd'
            oNameNrrdCropCanNM = outputDir  + '/' + prefix + '_' + nVl + '_cropCanNM.nrrd'
            oNameCSV         = outputDir  + '/' + prefix + '_' + nVl + '.csv'
            oNameCSVDets     = outputDir  + '/' + prefix + '_' + nVl + '_dets.csv'
            oNameCSVDetsClus = outputDir  + '/' + prefix + '_' + nVl + '_detsClus.csv'
            oNameCSVDetsCoords     = outputDir  + '/' + prefix + '_' + nVl + '_detsCoords.csv'
            oNameCSVDetsClusCoords = outputDir  + '/' + prefix + '_' + nVl + '_detsClusCoords.csv'

            # Saves beautiful images in your preferred format

            detector.writeDetectionAsPng      (oName)
            detector.writeDetectionAsPng      (oNameCrop, True)
            detector.writeAllDetectionsAsPng  (oNameAll)
            detector.writeDetectionSliceAsPng (oNameSlicePng)
            detector.writeDetectionAsNRRD     (oNameNrrdCrop, False, True)
            detector.writeDetectionAsNRRD     (oNameNrrdCropCan, True, True)
            detector.writeDetectionAsNRRD     (oNameNrrdCropNM,False, False )
            detector.writeDetectionAsNRRD     (oNameNrrdCropCanNM, True, False)
            detector.writeDetectionSliceAsNRRD(oNameNrrd)
            detector.writeDetectionCoordinatesAsCSV (oNameCSV)
            detector.writeAllDetectionsAsCSV      (oNameCSVDets)
            detector.writeAllDetectionsClusterAsCSV (oNameCSVDetsClus)
            detector.writeAllDetectionsCoordinatesAsCSV        (oNameCSVDetsCoords)
            detector.writeAllDetectionsClusterCoordinatesAsCSV (oNameCSVDetsClusCoords)
            end = time.time()
            print("Saving time %f" % (end - start))

#            # Inteface for further post-processing
#            img = detector.getDetectionAsNpyArray()
#            plt.imshow(img); plt.show(); plt.pause(5)
