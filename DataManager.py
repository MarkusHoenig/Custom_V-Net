import numpy as np
import SimpleITK as sitk
from os import listdir, makedirs
from os.path import isfile, join, splitext, isdir

class DataManager(object):
    params=None
    srcFolder=None
    resultsDir=None

    fileList=None
    gtList=None
    dmapList=None
    pmapList=None

    sitkImages=None
    sitkGT=None
    sitkDmap=None
    sitkPMap=None
    meanIntensityTrain = None

    def __init__(self,srcFolder,resultsDir,parameters):
        self.params=parameters
        self.srcFolder=srcFolder
        self.resultsDir=resultsDir

    def createImageFileList(self):
        self.fileList = [f for f in listdir(self.srcFolder) if isfile(join(self.srcFolder, f)) and 'segmented' not in f and 'raw' not in f and 'dmap' not in f and 'pmap' not in f and 'txt' not in f]
        print 'FILE LIST: ' + str(self.fileList)


    def createGTFileList(self):
        self.gtList=list()
        self.dmapList=list()
        self.pmapList=list()
        for f in self.fileList:
            filename, ext = splitext(f)
            self.gtList.append(join(filename + '_segmented' + ext))
            if self.params['dmap']:
                self.dmapList.append(join(filename + '_dmap' + ext))
            if self.params['ProbabilityMap']:
                self.pmapList.append(join(filename + '_pmap' + ext))
            #else:
             #   self.gtList.append(join(filename + '_segmented' + ext))
        #print 'DMap LIST: ' + str(self.gtList)

    def createPmapFileList(self):
        self.pmapList=list()
        for f in self.fileList:
            filename, ext = splitext(f)
            self.pmapList.append(join(filename + '_pmap' + ext))


    def loadImages(self):
        self.sitkImages=dict()
        rescalFilt=sitk.RescaleIntensityImageFilter()
        rescalFilt.SetOutputMaximum(1)
        rescalFilt.SetOutputMinimum(0)

        stats = sitk.StatisticsImageFilter()
        m = 0.
        for f in self.fileList:
            self.sitkImages[f]=rescalFilt.Execute(sitk.Cast(sitk.ReadImage(join(self.srcFolder, f)),sitk.sitkFloat32))
            stats.Execute(self.sitkImages[f])
            m += stats.GetMean()

        self.meanIntensityTrain=m/len(self.sitkImages)

        #if (self.params['ProbabilityMap']) :
        #    self.sitkPMap=sitk.Cast(sitk.ReadImage("/media/markus/Daten/Data/All_Labels/Atlas/prob_Atlas.mhd"),sitk.sitkFloat32)

    def loadGT(self):
        self.sitkGT=dict()

        for f in self.gtList:
            self.sitkGT[f.replace('_segmented','')]=sitk.Cast(sitk.ReadImage(join(self.srcFolder, f)),sitk.sitkFloat32)  #>0.5


    def loadDmap(self):
        self.sitkDmap=dict()

        for f in self.dmapList:
            self.sitkDmap[f.replace('_dmap','')] = sitk.Cast(sitk.ReadImage(join(self.srcFolder, f)), sitk.sitkFloat32)  # >0.5

    def loadPmap(self):
        self.sitkPmap=dict()

        for f in self.pmapList:
            self.sitkPmap[f.replace('_pmap','')] = sitk.Cast(sitk.ReadImage(join(self.srcFolder, f)), sitk.sitkFloat32)  # >0.5




    def loadTrainingData(self):
        self.createImageFileList()
        self.createGTFileList()
        self.loadImages()
        self.loadGT()
        if self.params['dmap']:
            self.loadDmap()
        if self.params['ProbabilityMap']:
            self.loadPmap()


    def loadTestData(self):
        self.createImageFileList()
        self.createPmapFileList()
        self.loadImages()
        if self.params['ProbabilityMap']:
            self.loadPmap()


    def getNumpyImages(self):
        dat = self.getNumpyData(self.sitkImages,sitk.sitkLinear)
        return dat


    def getNumpyGT(self):
        dat = self.getNumpyData(self.sitkGT,sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key]).astype(dtype=np.float32)      #>0.5

        return dat

    def getNumpyDmap(self):
        dat = self.getNumpyData(self.sitkDmap, sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key]).astype(dtype=np.float32)  # >0.5

        return dat

    #def getPMap(self):
    #    PMap = np.transpose(sitk.GetArrayFromImage(self.sitkPMap).astype(dtype=np.float32), [2, 1, 0])
    #    return PMap

    def getNumpyPMap(self):
        dat = self.getNumpyData(self.sitkPmap, sitk.sitkLinear)

        for key in dat:
            dat[key] = (dat[key]).astype(dtype=np.float32)  # >0.5

        return dat


    def getNumpyData(self,dat,method):
        ret=dict()
        for key in dat:
            ret[key] = np.zeros([self.params['VolSize'][0], self.params['VolSize'][1], self.params['VolSize'][2]], dtype=np.float32)

            img=dat[key]

            #we rotate the image according to its transformation using the direction and according to the final spacing we want
            factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

            factorSize = np.asarray(img.GetSize() * factor, dtype=float)

            newSize = np.max([factorSize, self.params['VolSize']], axis=0)

            newSize = newSize.astype(dtype=int)

            T=sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img)
            resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(method)
            if self.params['normDir']:
                resampler.SetTransform(T.GetInverse())

            imgResampled = resampler.Execute(img)


            imgCentroid = np.asarray(newSize, dtype=float) / 2.0

            imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize(list(self.params['VolSize'].astype(dtype=int)))
            regionExtractor.SetIndex(list(imgStartPx))

            imgResampledCropped = regionExtractor.Execute(imgResampled)

            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [2, 1, 0])

        return ret


    def writeResults(self, result, key, version=0, binary=True):
        if binary:
            self.writeResultsFromNumpyLabel(result, key, version)
        else:
            self.writeResultsDirectly(result, key, version)



    def writeResultsFromNumpyLabel(self, result, key, version):
        img = self.sitkImages[key]

        toWrite=sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                     self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]), range(imgStartPx[0],int(imgStartPx[0]+self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]), range(imgStartPx[1], int(imgStartPx[1]+self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]), range(imgStartPx[2], int(imgStartPx[2]+self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX),int(srcY),int(srcZ),float(result[dstX,dstY,dstZ]))
                    except:
                        pass


        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())

        if self.params['normDir']:
            resampler.SetTransform(T)

        toWrite = resampler.Execute(toWrite)

        thfilter=sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(self.params['probThreshold'])
        toWrite = thfilter.Execute(toWrite)

        #connected component analysis (better safe than sorry)

        cc = sitk.ConnectedComponentImageFilter()
        toWritecc = cc.Execute(sitk.Cast(toWrite,sitk.sitkUInt8))

        arrCC=np.transpose(sitk.GetArrayFromImage(toWritecc).astype(dtype=float), [2, 1, 0])

        lab=np.zeros(int(np.max(arrCC)+1),dtype=float)

        for i in range(1,int(np.max(arrCC)+1)):
            lab[i]=np.sum(arrCC==i)

        activeLab=np.argmax(lab)

        toWrite = (toWritecc==activeLab)

        toWrite = sitk.Cast(toWrite,sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        #print join(self.resultsDir, filename + '_result' + ext)
        if version==0 :
            writer.SetFileName(join(self.resultsDir, filename + '_result' + ext))
        else:
            if not isdir(join(self.resultsDir,str(version))):
                makedirs(join(self.resultsDir,str(version)))
            writer.SetFileName(join(self.resultsDir + str(version), filename + '_result' + str(version) + ext))
        writer.Execute(toWrite)



    def writeResultsDirectly(self, result, key, version=0):

        img = self.sitkImages[key]

        toWrite = sitk.Image(img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], sitk.sitkFloat32)

        factor = np.asarray(img.GetSpacing()) / [self.params['dstRes'][0], self.params['dstRes'][1],
                                                 self.params['dstRes'][2]]

        factorSize = np.asarray(img.GetSize() * factor, dtype=float)

        newSize = np.max([factorSize, self.params['VolSize']], axis=0)

        newSize = newSize.astype(dtype=int)

        T = sitk.AffineTransform(3)
        T.SetMatrix(img.GetDirection())

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing([self.params['dstRes'][0], self.params['dstRes'][1], self.params['dstRes'][2]])
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

        if self.params['normDir']:
            resampler.SetTransform(T.GetInverse())

        toWrite = resampler.Execute(toWrite)

        imgCentroid = np.asarray(newSize, dtype=float) / 2.0

        imgStartPx = (imgCentroid - self.params['VolSize'] / 2.0).astype(dtype=int)

        for dstX, srcX in zip(range(0, result.shape[0]),
                              range(imgStartPx[0], int(imgStartPx[0] + self.params['VolSize'][0]))):
            for dstY, srcY in zip(range(0, result.shape[1]),
                                  range(imgStartPx[1], int(imgStartPx[1] + self.params['VolSize'][1]))):
                for dstZ, srcZ in zip(range(0, result.shape[2]),
                                      range(imgStartPx[2], int(imgStartPx[2] + self.params['VolSize'][2]))):
                    try:
                        toWrite.SetPixel(int(srcX), int(srcY), int(srcZ), float(result[dstX, dstY, dstZ]))
                    except:
                        pass

        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())

        if self.params['normDir']:
            resampler.SetTransform(T)

        toWrite = resampler.Execute(toWrite)

        # thfilter = sitk.BinaryThresholdImageFilter()
        # thfilter.SetInsideValue(1)
        # thfilter.SetOutsideValue(0)
        # thfilter.SetLowerThreshold(self.params['probThreshold'])
        # toWrite = thfilter.Execute(toWrite)


        # for X in range(0, 128):
        #     for Y in range(0, 128):
        #         for Z in range(0, 64):
        #             try:
        #                 toWrite.SetPixel(int(X),int(Y),int(Z),float(result[X,Y,Z]))
        #             except:
        #                 pass

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        # print join(self.resultsDir, filename + '_result' + ext)
        if version==0 :
            writer.SetFileName(join(self.resultsDir, filename + '_result' + ext))
        else:
            if not isdir(join(self.resultsDir,str(version))):
                makedirs(join(self.resultsDir,str(version)))
            writer.SetFileName(join(self.resultsDir + str(version), filename + '_result' + str(version) + ext))
        writer.Execute(toWrite)