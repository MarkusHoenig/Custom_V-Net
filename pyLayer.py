import caffe
import numpy as np

class DiceLoss(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None
    #Dice = [0,0]

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the dice. the result of the softmax and the ground truth.")



    def reshape(self, bottom, top):
        # check input dimensions match
        # if bottom[0].count != 2*bottom[1].count:
        #     print bottom[0].data.shape
        #     print bottom[1].data.shape
        #     raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, bottom, top):

        dice = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.union = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.intersection = np.zeros(bottom[0].data.shape[0],dtype=np.float32)

        #self.result = np.reshape(np.squeeze(np.argmax(bottom[0].data[...],axis=1)),[bottom[0].data.shape[0],bottom[0].data.shape[2]])
                                                        #    V
        self.result = np.reshape(np.squeeze(bottom[0].data[:,0,:]),[bottom[0].data.shape[0],bottom[0].data.shape[2]])
        self.gt = np.reshape(np.squeeze(bottom[1].data[...]),[bottom[1].data.shape[0],bottom[1].data.shape[2]])

        self.gt = (self.gt > 0.5).astype(dtype=np.float32)
        self.result = self.result.astype(dtype=np.float32)
        self.result[self.result[:]>=0.5]=1
        self.result[self.result[:]<0.5]=0

        for i in range(0,bottom[0].data.shape[0]):
            # compute dice
            CurrResult = (self.result[i,:]).astype(dtype=np.float32)
            CurrGT = (self.gt[i,:]).astype(dtype=np.float32)

            self.union[i]=(np.sum(CurrResult) + np.sum(CurrGT))
            self.intersection[i]=(np.sum(CurrResult * CurrGT))

            dice[i] = 1- (2 * self.intersection[i] / (self.union[i]+0.00001))
            print "Dice Loss: " + str(dice[i])
            #self.Dice[i]=dice[i]

        top[0].data[0]=np.sum(dice)

    def backward(self, top, propagate_down, bottom):
        for btm in [0]:
            prob = bottom[0].data[...]
            bottom[btm].diff[...] = np.zeros(bottom[btm].diff.shape, dtype=np.float32)
            for i in range(0, bottom[btm].diff.shape[0]):

                #self.union[i] = np.sum(prob[i,1,:]) + np.sum(self.gt[i,:])
                #self.intersection[i] = np.sum(prob[i,1,:] * self.gt[i,:])

                #if self.Dice[0] == 1 & self.Dice[1] == 1:

                #    bottom[btm].diff[i, 0, :] += 0.5
                #    bottom[btm].diff[i, 1, :] -= 0.5
                #else:                  #   V                                      #  |
                bottom[btm].diff[i, 0, :] -= 2.0 * (                               #  V
                (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0*prob[i,0,:]*(self.intersection[i]) / (self.union[i]) ** 2)
                if bottom[btm].diff.shape[1]==2:
                    bottom[btm].diff[i, 1, :] += 2.0 * (
                    (self.gt[i, :] * self.union[i]) / ((self.union[i]) ** 2) - 2.0*prob[i,0,:]*(self.intersection[i]) / (self.union[i]) ** 2)




class DistLoss(caffe.Layer):
    """
    Compute loss based on Distance-Maps.
    """

    alpha = 0
    beta = 0

    dmap_o = None
    dmap_i = None
    result_o = None
    result_i = None

    union = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute the loss. the result of the softmax and the distance map.")



    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != 2*bottom[1].count:
            print bottom[0].data.shape
            print bottom[1].data.shape
            raise Exception("the dimension of inputs should match")

        # loss output is two scalars (mean and std)
        top[0].reshape(1)

    def forward(self, bottom, top):

        dist = np.zeros(bottom[0].data.shape[0],dtype=np.float32)
        self.union = np.zeros(bottom[0].data.shape[0],dtype=np.float32)

        self.result_i = np.reshape(np.squeeze(bottom[0].data[:,0,:]), [bottom[0].data.shape[0], bottom[0].data.shape[2]])   #inside labelmap
        self.result_o = np.reshape(np.squeeze(bottom[0].data[:,1,:]), [bottom[0].data.shape[0], bottom[0].data.shape[2]])   #outside labelmap
        self.result_i[self.result_i < 0.5] = 0; self.result_i[self.result_i >= 0.5] = 1
        self.result_o[self.result_o < 0.5] = 0; self.result_o[self.result_o >= 0.5] = 1

        gt = np.reshape(np.squeeze(bottom[1].data[...]), [bottom[1].data.shape[0], bottom[1].data.shape[2]])

        self.dmap_i = gt.astype(dtype=np.float32)
        self.dmap_o = gt.astype(dtype=np.float32)
        self.result_i = self.result_i.astype(dtype=np.float32)
        self.result_o = self.result_o.astype(dtype=np.float32)

        self.dmap_o[self.dmap_o < 0] = -self.alpha
        self.dmap_i[self.dmap_i > 0] = self.beta

        for i in range(0,bottom[0].data.shape[0]):
            # compute distance loss

            self.union[i] = (np.sum(self.result_i[i]) + np.sum(self.result_o[i]))           #bottom[0].data.shape[0]*bottom[0].data.shape[1]*bottom[0].data.shape[2]
            dist[i] = (self.dmap_o[i].dot(self.result_i[i]) - self.dmap_i[i].dot(self.result_o[i])) / (self.union[i])

            print "Distance Loss: " + str(dist[i])

        top[0].data[0]=np.sum(dist)


    def backward(self, top, propagate_down, bottom):
        btm = 0
        prob_i = np.squeeze(bottom[0].data[:, 0, :])
        prob_o = np.squeeze(bottom[0].data[:, 1, :])

        bottom[btm].diff[...] = np.zeros(bottom[btm].diff.shape, dtype=np.float32)
        for i in range(0, bottom[btm].diff.shape[0]):
            bottom[btm].diff[i, 0, :] = (self.dmap_o[i] * self.union[i] - 2 * prob_i[i] *(self.dmap_o[i].dot(self.result_i[i]) - self.dmap_i[i].dot(self.result_o[i]))) / (self.union[i] ** 2)

            bottom[btm].diff[i, 1, :] = (-self.dmap_i[i] * self.union[i] - 2 * prob_o[i] * (self.dmap_o[i].dot(self.result_i[i]) - self.dmap_i[i].dot(self.result_o[i]))) / (self.union[i] ** 2)


