#!/usr/bin/python

import sys
import os
import numpy as np
import glob
import VNet as VN
os.environ['GLOG_minloglevel'] = '0'

if "-ssh" in sys.argv:
    HomePath='/home/'
else:
    HomePath='/media/markus/Daten/'

#sys.path.append(HomePath + "SVN/3D-Caffe-Markus/python/")

basePath = HomePath + 'SVN/V-Net/VNet-Markus/' #os.getcwd()
MyPath= HomePath + 'Data/'

params = dict()
params['DataManagerParams']=dict()
params['ModelParams']=dict()

#params of the algorithm
params['ModelParams']['numcontrolpoints']=8
params['ModelParams']['sigma']=10
params['ModelParams']['device']=0
params['ModelParams']['prototxtTrain']=os.path.join(basePath,'Prototxt/train_final-model_research_Pmap.prototxt')
params['ModelParams']['prototxtTest']=os.path.join(basePath,'Prototxt/test_final-model_research_Pmap.prototxt')
params['ModelParams']['snapshot']=20000
params['ModelParams']['dirTrain']=os.path.join(MyPath,'Train/')
params['ModelParams']['dirTest']=os.path.join(MyPath,'Test/')
params['ModelParams']['dirImages']=os.path.join(MyPath,'AllImages-Precise/')
params['ModelParams']['dirResult']=os.path.join(HomePath,'Masterarbeit/V-Net/Final-research-version-Pmap-New/')  #where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots']=os.path.join(HomePath,'Masterarbeit/V-Net/Final-research-version-Pmap-New/Models/')  #where we need to save the results (relative to the base path)
params['ModelParams']['batchsize'] = 1  #the batchsize
params['ModelParams']['numIterations'] = 20000
params['ModelParams']['baseLR'] = 0.0001  #the learning rate, initial one: 0.0001
params['ModelParams']['stepSize'] = 10000  #the stepsize, initial one: 20000
params['ModelParams']['stepSnapshot'] = 100000
params['ModelParams']['momentum'] = 0.9  #the momentum 0.9
params['ModelParams']['momentum2'] = 0.999  #the momentum2 of Adam 0.999
params['ModelParams']['weightDecay'] = 0.0005  #the weight_decay, initial one: 0.0005
params['ModelParams']['delta'] = 0.0000001 #epsilon used to not divide by 0 (can improve stability if higher)
params['ModelParams']['nProc'] = 4  #the number of threads to do data augmentation
params['ModelParams']['Solver'] = 1  #0=SGDMomentum/1=Adam

params['ModelParams']['ValInter'] = 50   #Interval between Validation Phases (0 for no validation)
params['ModelParams']['ValNum'] = 8 #Number of Images to Validate
params['ModelParams']['CrossVal'] = 5   #number of Cross validations (images taken from dirImages when Cross val > 0)

params['ModelParams']['bestEpoch'] = True       #save the snapshot of the best validation loss
params['ModelParams']['restore'] = False        #restore pretraned weights
params['ModelParams']['histmatching'] = True       #enables histogram matching for training
params['ModelParams']['whitening'] = True        #enables global contrast normalization for training
params['ModelParams']['RandomDeform'] = 1       #percent of images that get randomly deformded

params['ModelParams']['dirCoeffs'] = HomePath + 'SVN/calc_coeffs/calc_coeffs.py'

#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1.5,1.5,1.5],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([128,96,96],dtype=int)
params['DataManagerParams']['normDir'] = False #if rotates the volume according to its transformation in the mhd file. Not reccommended.

params['DataManagerParams']['ProbabilityMap'] = True
params['ModelParams']['WNet'] = False
params['DataManagerParams']['AutoIter'] = 1  # Number of iterations for the auto-kontext-model
params['DataManagerParams']['labelOut'] = True # writes label or probabilitymap to .mhd file
params['DataManagerParams']['probThreshold'] = 0.7


if '-train' in sys.argv:
    if params['ModelParams']['CrossVal'] == 0:
        model = VN.VNet(params)
        model.train()
    else:
        VN.crossval(params)

if '-test' in sys.argv:
    model = VN.VNet(params)
    model.test()


