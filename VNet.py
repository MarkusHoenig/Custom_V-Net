
import numpy as np
import os
import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe
import glob
from random import shuffle
import DataManager as DM
import utilities
import shutil
from os.path import splitext, isdir
from multiprocessing import Process, Queue
import calc_coeffs


import threading
import subprocess

class VNet(object):
    params=None
    dataManagerTrain=None
    dataManagerTest=None
    inputdepth=None

    train_loss = None
    val_loss = list()
    test_loss = None

    numpy_val=None
    test_set=None
    val_set=None



    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()
        self.val_loss = []

        if (self.params['DataManagerParams']['ProbabilityMap']):
            self.inputdepth=2
        else: self.inputdepth=1

        if not isdir(self.params['ModelParams']['dirSnapshots']):
            os.makedirs(self.params['ModelParams']['dirSnapshots'])


    def prepareDataThread(self, dataQueue, numpyImages, numpyGT, numpyDmap=None, numpyPmap=None):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        keys = numpyImages.keys()

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()
        whichDataList = np.random.randint(len(keys), size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichDataForMatchingList = np.random.randint(len(keys), size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))

        for whichData,whichDataForMatching in zip(whichDataList,whichDataForMatchingList):
            filename, ext = splitext(keys[whichData])
            currKey=keys[whichData]
            #currImgKey = filename + ext
            #currGtKey = filename + '_segmented' + ext

            # data augmentation through hist matching across different examples...
            ImgKeyMatching = keys[whichDataForMatching]

            defImg = numpyImages[currKey]
            defLab = numpyGT[currKey]

            if self.params['DataManagerParams']['dmap'] :
                defDmap = numpyDmap[currKey]
            else: defDmap=None

            if self.params['DataManagerParams']['ProbabilityMap'] :
                defPmap = numpyPmap[currKey]
            else: defPmap=None

            if self.params['ModelParams']['histmatching'] :
                defImg = utilities.hist_match(defImg, numpyImages[ImgKeyMatching])

            if(np.random.rand(1)[0]<self.params['ModelParams']['RandomDeform']): #do not apply deformations always, just sometimes
                defImg, defLab, defDmap = utilities.produceRandomlyDeformedImage(defImg, defLab, defDmap,
                                    self.params['ModelParams']['numcontrolpoints'],
                                               self.params['ModelParams']['sigma'], self.params['DataManagerParams']['dmap'])

            weightData = np.zeros_like(defLab,dtype=float)
            weightData[defLab == 1] = np.prod(defLab.shape) / np.sum((defLab == 1).astype(dtype=np.float32))
            weightData[defLab == 0] = np.prod(defLab.shape) / np.sum((defLab == 0).astype(dtype=np.float32))

            dataQueue.put(tuple((defImg, defLab, defDmap, defPmap, currKey)))



    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        batchData = np.zeros((batchsize, self.inputdepth, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        if self.params['DataManagerParams']['dmap']:
            batchDmap = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)


        train_loss = np.zeros(nr_iter)
        if self.params['ModelParams']['WNet']:
            train_loss_2 = np.zeros(nr_iter)
        if self.params['DataManagerParams']['dmap']:
            train_loss_dist = np.zeros(nr_iter)
        min_loss_it = None

        for it in range(nr_iter):
            for i in range(batchsize):

                [defImg, defLab, defDmap, defPmap, key] = dataQueue.get()

                batchData[i, 0, :, :, :] = defImg.astype(dtype=np.float32)
                batchLabel[i, 0, :, :, :] = defLab.astype(dtype=np.float32)         #>0.5
                if self.params['DataManagerParams']['dmap']:
                    batchDmap[i, 0, :, :, :] = defDmap.astype(dtype=np.float32)
                    #except: batchLabel[i, 0, :, :, :] = defDmap.astype(dtype=np.float32)

                if self.params['DataManagerParams']['ProbabilityMap']:
                    batchData[i, 1, :, :, :] = defPmap.astype(dtype=np.float32)

            solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            if 'dmap' in solver.net.blobs:
                solver.net.blobs['dmap'].data[...] = batchDmap.astype(dtype=np.float32)
            elif self.params['DataManagerParams']['dmap']:
                solver.net.blobs['label'].data[...] = batchDmap.astype(dtype=np.float32)



                #solver.net.blobs['dmap'].data[...] = batchDmap.astype(dtype=np.float32)

            #if (self.params['DataManagerParams']['ProbabilityMap']) :
            #    solver.net.blobs['pmap'].data[...] = batchPMap.astype(dtype=np.float32)
            #solver.net.blobs['labelWeight'].data[...] = batchWeight.astype(dtype=np.float32)
            #use only if you do softmax with loss


            solver.step(1)  # this does the training
            train_loss[it] = solver.net.blobs['loss'].data / self.params['ModelParams']['batchsize']
            if self.params['ModelParams']['WNet']:
                train_loss_2[it] = solver.net.blobs['loss_2'].data / self.params['ModelParams']['batchsize']
            if self.params['DataManagerParams']['dmap'] and 'dist_loss' in solver.net.blobs:
                train_loss_dist[it] = solver.net.blobs['dist_loss'].data / self.params['ModelParams']['batchsize']


            if self.params['ModelParams']['ValInter']!=0:
                val_loss=None
                if it % self.params['ModelParams']['ValInter']==0:
                    val_loss = self.valThread(solver, self.params['ModelParams']['ValNum'])
                    print "Validation Loss: " + str(val_loss)
                    self.val_loss.append(val_loss)
                if self.params['ModelParams']['bestEpoch']:
                    if (it == 0):
                        min_loss = val_loss
                        min_loss_it = 0
                        solver.snapshot()
                    elif (it >= nr_iter * 0.1 and (it) % (self.params['ModelParams']['ValInter']) == 0):  # *0.7   /10
                        if val_loss==None:
                            val_loss = self.valThread(solver, self.params['ModelParams']['ValNum'])
                        if val_loss < min_loss:
                            for f in glob.glob(self.params['ModelParams']['dirSnapshots'] + '_iter_' + str(
                                            min_loss_it + 1) + '.*'):
                                os.remove(f)
                            solver.snapshot()
                            min_loss = val_loss
                            min_loss_it = it



            if (np.mod(it, 10) == 0): # and self.params['ModelParams']['SSH']==False):
                plt.clf()
                plt.plot(range(0, it), train_loss[0:it], label='Trainings Loss')
                if self.params['ModelParams']['ValInter'] != 0:
                    plt.plot(range(0, it + 1, self.params['ModelParams']['ValInter']), self.val_loss, label='Validation Loss')
                if self.params['DataManagerParams']['dmap'] and 'dist_loss' in solver.net.blobs:
                    plt.plot(range(0, it), train_loss_dist[0:it], label='Distance Loss')
                if self.params['ModelParams']['WNet']:
                    plt.plot(range(0, it), train_loss_2[0:it], label='W-Net Loss 2')
                plt.xlabel('Iterations')
                plt.legend()
                plt.pause(0.00000001)
                plt.show()





        self.train_loss=train_loss
        solver.snapshot()

        #if validation is on:
        if self.params['ModelParams']['ValInter']!=0 or self.params['ModelParams']['CrossVal'] != 0:
            val_loss = self.valThread(solver, self.params['ModelParams']['ValNum'])     #last validation at the last iteration
            print "Validation Loss: " + str(val_loss)
            self.val_loss.append(val_loss)
            if self.params['ModelParams']['bestEpoch']:                                 #if best epoch is on: solver restore best epoch
                if val_loss < min_loss:
                    min_loss = val_loss
                    min_loss_it = it
                print "\nBest loss: " + str(min_loss) + "\nat Iteration: " + str(min_loss_it + 1)
                solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(min_loss_it + 1) + ".solverstate")
                test_it=min_loss_it
            else:
                test_it=nr_iter
            self.test_loss = self.valThread(solver)                                     #Test Loss is computed
            print "Test Loss: " + str(self.test_loss)

        plt.clf()
        plt.plot(range(nr_iter), train_loss, label='Trainings Loss')
        if self.params['ModelParams']['ValInter'] != 0:
            plt.plot(range(0, nr_iter+1, self.params['ModelParams']['ValInter']), np.asarray(self.val_loss), label='Validation Loss')
        if self.params['DataManagerParams']['dmap'] and 'dist_loss' in solver.net.blobs:
            plt.plot(range(nr_iter), train_loss_dist, color='green', label='Distance Loss')
        if self.params['ModelParams']['ValInter'] != 0:
            plt.plot(test_it, self.test_loss, 'rx', markersize=12, label='Test Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iterations')
        plt.legend()
        plt.savefig(os.path.join(str(self.params['ModelParams']['dirResult']), 'learning-curve.png'))





    def valThread(self, solver, NumImages=None):

        keylist = self.numpy_val['Images'].keys()
        keylist.sort()

        batchsize = self.params['ModelParams']['batchsize']
        if NumImages:
            nr_iter = int(NumImages/batchsize)
        else:
            nr_iter=int((len(self.numpy_val['Images'])-self.params['ModelParams']['ValNum'])/batchsize)
            for i in range(self.params['ModelParams']['ValNum']):
                keylist.pop(0)

        batchData = np.zeros((batchsize, self.inputdepth, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        batchLabel = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)
        if self.params['DataManagerParams']['dmap']:
            batchDmap = np.zeros((batchsize, 1, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=float)

        loss=np.zeros(nr_iter)
        if self.params['ModelParams']['WNet']:
            loss_2 = np.zeros(nr_iter)

        for it in range(nr_iter):
            for i in range(batchsize):

                currkey = keylist.pop(0)

                Img = self.numpy_val['Images'][currkey]
                Lab = self.numpy_val['GT'][currkey]

                batchData[i, 0, :, :, :] = Img.astype(dtype=np.float32)
                batchLabel[i, 0, :, :, :] = Lab.astype(dtype=np.float32)         #>0.5
                if self.params['DataManagerParams']['dmap']:
                    Dmap=self.numpy_val['Dmap'][currkey]
                    batchDmap[i, 0, :, :, :] = Dmap.astype(dtype=np.float32)

                if self.params['DataManagerParams']['ProbabilityMap']:
                    Pmap=self.numpy_val['Pmap'][currkey]
                    batchData[i, 1, :, :, :] = Pmap.astype(dtype=np.float32)

            solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            if 'dmap' in solver.net.blobs:
                solver.net.blobs['dmap'].data[...] = batchDmap.astype(dtype=np.float32)
            elif self.params['DataManagerParams']['dmap']:
                solver.net.blobs['label'].data[...] = batchDmap.astype(dtype=np.float32)

                #solver.net.blobs['dmap'].data[...] = batchDmap.astype(dtype=np.float32)

            out = solver.net.forward()
            loss[it] = out["loss"]/batchsize
            if self.params['ModelParams']['WNet']:
                loss_2[it] = out["loss_2"]/batchsize
            if 'val_loss' in out:
                loss[it]=out["val_loss"]/batchsize


########################################################################################################################
            # if not NumImages:
            #     l = out["labelmap"]
            #     labelmap = np.squeeze(l[0, 0, :, :, :])
            #
            #     # results[key] = np.squeeze(labelmap)
            #     self.dataManagerTrain.writeResults(np.squeeze(labelmap), currkey, binary=self.params['DataManagerParams']['labelOut'])
########################################################################################################################

        return np.mean(loss)




    def train(self, valcycle=None, keylist=None):

        if self.params['ModelParams']['CrossVal'] == 0:
            print self.params['ModelParams']['dirTrain']
            #we define here a data manage object
            self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])
            self.dataManagerTrain.loadTrainingData()  # loads in sitk format
        else:
            print self.params['ModelParams']['dirImages']
            #we check if we have set a data manage object
            if self.dataManagerTrain == None :
                exit()


        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)

        assert howManyGT == howManyImages

        print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        # Write a temporary solver text file because pycaffe is stupid
        if (self.params['ModelParams']['Solver'] == 0):     #SDG-Solver
            with open("solver.prototxt", 'w') as f:
                f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
                f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
                f.write("momentum: " + str(self.params['ModelParams']['momentum']) + " \n")
                f.write("weight_decay: " + str(self.params['ModelParams']['weightDecay']) + " \n")
                f.write("lr_policy: \"step\" \n")
                f.write("stepsize: " + str(self.params['ModelParams']['stepSize']) + " \n")
                f.write("gamma: 0.1 \n")
                f.write("display: 1 \n")
                f.write("snapshot: " + str(self.params['ModelParams']['stepSnapshot']) + " \n")
                f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
                #f.write("test_iter: 3 \n")
                #f.write("test_interval: " + str(test_interval) + "\n")

            f.close()
            solver = caffe.SGDSolver("solver.prototxt")
            os.remove("solver.prototxt")

        if (self.params['ModelParams']['Solver'] == 1):     #Adam-Solver
            with open("solver.prototxt", 'w') as f:
                f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
                f.write("solver_type: ADAM \n")
                f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
                f.write("momentum: " + str(self.params['ModelParams']['momentum']) + " \n")
                f.write("momentum2: " + str(self.params['ModelParams']['momentum2']) + " \n")
                f.write("weight_decay: " + str(self.params['ModelParams']['weightDecay']) + " \n")
                f.write("lr_policy: \"fixed\" \n")
                f.write("delta: " + str(self.params['ModelParams']['delta']) + " \n")
                f.write("display: 1 \n")
                f.write("snapshot: " + str(self.params['ModelParams']['stepSnapshot']) + " \n")
                f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
            f.close()
            solver = caffe.get_solver("solver.prototxt")
            os.remove("solver.prototxt")

        if (self.params['ModelParams']['restore'] and self.params['ModelParams']['snapshot'] > 0):
            solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        numpyImages = self.dataManagerTrain.getNumpyImages()
        numpyGT = self.dataManagerTrain.getNumpyGT()
        if self.params['DataManagerParams']['dmap']:
            numpyDmap = self.dataManagerTrain.getNumpyDmap()
        else: numpyDmap = None
        if self.params['DataManagerParams']['ProbabilityMap']:
            numpyPmap = self.dataManagerTrain.getNumpyPMap()
        else: numpyPmap = None


        #numpyImages['Case00.mhd']
        #numpy images is a dictionary that you index in this way (with filenames)

        if self.params['ModelParams']['whitening']:
            for key in numpyImages:
                mean = np.mean(numpyImages[key][numpyImages[key]>0])
                std = np.std(numpyImages[key][numpyImages[key]>0])

                numpyImages[key]-=mean
                numpyImages[key]/=std

        if self.params['ModelParams']['CrossVal'] != 0:        #loading images for validation and testing for cross validation

            howmanyVal = int(howManyImages/self.params['ModelParams']['CrossVal'])

            start=0+howmanyVal*valcycle
            end=howmanyVal+howmanyVal*valcycle
            keylist_val=keylist[start:end]

            self.numpy_val=dict()
            self.numpy_val['Images'] = dict()
            self.numpy_val['GT'] = dict()
            self.numpy_val['Dmap'] = dict()
            self.numpy_val['Pmap'] = dict()
            for key in keylist_val:
                self.numpy_val['Images'][key]=numpyImages[key]
                del numpyImages[key]
                self.numpy_val['GT'][key] = numpyGT[key]
                del numpyGT[key]
                if self.params['DataManagerParams']['dmap']:
                    self.numpy_val['Dmap'][key]=numpyDmap[key]
                    del numpyDmap[key]
                if self.params['DataManagerParams']['ProbabilityMap']:
                    self.numpy_val['Pmap'][key] = numpyPmap[key]
                    del numpyPmap[key]

            # write set keys to file:
            train_set=numpyImages.keys()
            train_set.sort()
            test_set=self.numpy_val['Images'].keys()
            test_set.sort()
            val_set=list()
            for i in range(self.params['ModelParams']['ValNum']):
                val_set.extend([test_set.pop(0)])
            Out = open(self.params['ModelParams']['dirSnapshots'] + "Test-Val-sets.txt", 'w')
            Out.write("Trainings-set:\n")
            Out.write(" ".join(map(str,train_set)))
            Out.write("\nTest-set:\n")
            Out.write(" ".join(map(str,test_set)))
            Out.write("\nValidation-set:\n")
            Out.write(" ".join(map(str,val_set)))
            Out.close()





        elif self.params['ModelParams']['ValInter'] != 0:       #loading images for regular testing or validation
            self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'],
                                                  self.params['ModelParams']['dirResult'],
                                                  self.params['DataManagerParams'])
            self.dataManagerTest.loadTrainingData()

            numpyImages_val = self.dataManagerTest.getNumpyImages()
            numpyGT_val = self.dataManagerTest.getNumpyGT()
            if self.params['DataManagerParams']['dmap']:
                numpyDmap_val = self.dataManagerTest.getNumpyDmap()
            else:
                numpyDmap_val = None
            if self.params['DataManagerParams']['ProbabilityMap']:
                numpyPmap_val = self.dataManagerTest.getNumpyPMap()
            else:
                numpyPmap_val = None


            if self.params['ModelParams']['whitening']:
                for key in numpyImages_val:
                    mean = np.mean(numpyImages_val[key][numpyImages_val[key] > 0])
                    std = np.std(numpyImages_val[key][numpyImages_val[key] > 0])

                    numpyImages_val[key] -= mean
                    numpyImages_val[key] /= std

            self.numpy_val = dict()
            self.numpy_val['Images'] = dict()
            self.numpy_val['GT'] = dict()
            self.numpy_val['Dmap'] = dict()
            self.numpy_val['Pmap'] = dict()
            for key in numpyImages_val:
                self.numpy_val['Images'][key] = numpyImages_val[key]
                self.numpy_val['GT'][key] = numpyGT_val[key]
                if self.params['DataManagerParams']['dmap']:
                    self.numpy_val['Dmap'][key] = numpyDmap_val[key]
                if self.params['DataManagerParams']['ProbabilityMap']:
                    self.numpy_val['Pmap'][key] = numpyPmap_val[key]


        dataQueue = Queue(30) #max 50 images in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue, numpyImages, numpyGT, numpyDmap, numpyPmap))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver)


        Out = open(self.params['ModelParams']['dirResult'] + "Loss.txt", 'w')
        Out.write("Trainings Loss:\n")
        Out.write(" ".join(map(str, self.train_loss)))
        Out.write("\nValidation Loss:\n")
        Out.write(" ".join(map(str, self.val_loss)))
        Out.write("\nTest Loss:\n")
        Out.write(" ".join(map(str, [self.test_loss])))
        Out.close()

        return [self.train_loss, self.val_loss, self.test_loss]



    def test(self, NumImages=0):

        if self.params['ModelParams']['CrossVal']==0:
            self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        else:
            self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirImages'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])

        self.dataManagerTest.loadTestData()

        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()

        if self.params['ModelParams']['whitening']:
            for key in numpyImages:
                mean = np.mean(numpyImages[key][numpyImages[key]>0])
                std = np.std(numpyImages[key][numpyImages[key]>0])

                numpyImages[key] -= mean
                numpyImages[key] /= std

        results = dict()

        batch = np.zeros((1, self.inputdepth, self.params['DataManagerParams']['VolSize'][0], self.params['DataManagerParams']['VolSize'][1], self.params['DataManagerParams']['VolSize'][2]), dtype=np.float32)



        if self.test_set != None:
            print "\nTest set size: "+str(len(self.test_set))
            key_list = numpyImages.keys()
            for key in key_list:
                if key not in self.test_set:
                    del numpyImages[key]

        if NumImages==0:
            NumImages=len(numpyImages)

        print "\nTesting with "+str(NumImages)+" Images\n"

        for key in sorted(numpyImages.keys())[:NumImages]:

            batch[0,0,:,:,:] = numpyImages[key].astype(dtype=np.float32)
            #btch = np.reshape(numpyImages[key],[1,1,numpyImages[key].shape[0],numpyImages[key].shape[1],numpyImages[key].shape[2]])

            if (self.params['DataManagerParams']['ProbabilityMap']):
                PMap = self.dataManagerTest.getNumpyPMap()
                filename, ext = splitext(key)
                keyn=filename + "_pmap" + ext
                batch[0, 1, :, :, :] = PMap[key].astype(dtype=np.float32)

            net.blobs['data'].data[...] = batch

            out = net.forward()

            if (self.params['DataManagerParams']['ProbabilityMap']):
                for i in range(self.params['DataManagerParams']['AutoIter'] - 1) :
                    l=out["labelmap"][0,0,:,:,:]
                    self.dataManagerTest.writeResults(np.squeeze(l), key, version=i+1, binary=self.params['DataManagerParams']['labelOut'])
                    batch[0,1,:,:,:] = l
                    net.blobs['data'].data[...] = batch
                    out = net.forward()

            l = out["labelmap"]
            labelmap = np.squeeze(l[0,int(self.params['ModelParams']['labelmap']),:,:,:])

            #results[key] = np.squeeze(labelmap)
            self.dataManagerTest.writeResults(np.squeeze(labelmap), key, binary=self.params['DataManagerParams']['labelOut'])

            if (out.has_key('3_labelmap')):
                l3 = out["3_labelmap"]
                labelmap3 = np.squeeze(l3[0, 0, :, :, :])
                #results3[key] = np.squeeze(labelmap3)

                self.dataManagerTest.writeResults(np.squeeze(labelmap3), key, version=3, binary=self.params['DataManagerParams']['labelOut'])

            if (out.has_key('2_labelmap')):
                l2 = out["2_labelmap"]
                labelmap2 = np.squeeze(l2[0, 0, :, :, :])
                #results2[key] = np.squeeze(labelmap3)

                self.dataManagerTest.writeResults(np.squeeze(labelmap2), key, version=2, binary=self.params['DataManagerParams']['labelOut'])

        if self.params['ModelParams']['CrossVal'] == 0:
            dir_Test = self.params['ModelParams']['dirTest']
        else:
            dir_Test = self.params['ModelParams']['dirImages']

        self.loss=calc_coeffs.calc_coeffs(self.params['ModelParams']['dirResult'], dir_Test)
        for i in range(self.params['DataManagerParams']['AutoIter'] - 1):
            calc_coeffs.calc_coeffs(self.params['ModelParams']['dirResult'] + str(i + 1), dir_Test)


        return np.mean(self.loss)




def crossval(params):
    test_loss=np.zeros(params['ModelParams']['CrossVal'])
    DataManagerCross = DM.DataManager(params['ModelParams']['dirImages'], params['ModelParams']['dirResult'], params['DataManagerParams'])
    DataManagerCross.loadTrainingData()
    # if params['DataManagerParams']['ProbabilityMap']:
    #     key_in=open(params['ModelParams']['dirResult'] + "Key_list.txt", 'r')
    #     key_list=key_in.readline().split(" ")
    # else:
    key_list = DataManagerCross.getNumpyImages().keys()
    shuffle(key_list)
    key_out = open(params['ModelParams']['dirResult'] + "Key_list.txt", 'w')
    key_out.write(" ".join(key_list))
    key_out.close()
    for i in range(params['ModelParams']['CrossVal']):
        model = VNet(params)
        model.dataManagerTrain=DataManagerCross
        test_loss[i] = model.train(i, key_list)[-1]
        del model
        for f in ['learning-curve.png', 'Loss.txt', 'Models']:
            filename, ext = splitext(f)
#            shutil.rmtree(params['ModelParams']['dirResult'] + filename + str(i) + ext)
            try: os.rename(params['ModelParams']['dirResult'] + f, params['ModelParams']['dirResult'] + filename + str(i) + ext)
            except: shutil.rmtree(params['ModelParams']['dirResult'] + filename + str(i) + ext); os.rename(params['ModelParams']['dirResult'] + f,params['ModelParams']['dirResult'] + filename + str(i) + ext)
    Out = open(params['ModelParams']['dirResult'] + "Test_Loss.txt", 'w')
    Out.write("K-Fold-Cross-Validation\n\nTrainings Loss:\n")
    for i in range(params['ModelParams']['CrossVal']):
        Out.write(str(i) + " : " + str(test_loss[i]) + "\t")
    Out.write("\n\nMean Loss: " + str(np.mean(test_loss)))
    Out.write("\nStd deviation: " + str(np.std(test_loss)))
    Out.close()

