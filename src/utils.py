import numpy as np
import pandas as pd
import sys
import os
import mindspore
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle as pkl
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.metrics import mask_evaluation_np

class Scaler_NYC:
    def __init__(self, train):
        """ NYC Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W = data.shape
        data = np.transpose(data,(0,2,1)).reshape((-1,D))
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33:40] = (data[:,33:40] - self.min[33:40]) / (self.max[33:40] - self.min[33:40])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        data[:,46] = (data[:,46] - self.min[46]) / (self.max[46] - self.min[46])
        data[:,47] = (data[:,47] - self.min[47]) / (self.max[47] - self.min[47])
        return np.transpose(data.reshape((T,W,-1)),(0,2,1))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape (T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


class Scaler_Chi:
    def __init__(self, train):
        """Chicago Max-Min

        Arguments:
            train {np.ndarray} -- shape(T, D, W, H)
        """
        train_temp = np.transpose(train,(0,2,1)).reshape((-1,train.shape[1]))
        self.max = np.max(train_temp,axis=0)
        self.min = np.min(train_temp,axis=0)
    def transform(self, data):
        """norm train，valid，test

        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} -- shape(T, D, W, H)
        """
        T,D,W = data.shape
        data = np.transpose(data,(0,2,1)).reshape((-1,D))#(T*W*H,D)
        data[:,0] = (data[:,0] - self.min[0]) / (self.max[0] - self.min[0])
        data[:,33] = (data[:,33] - self.min[33]) / (self.max[33] - self.min[33])
        data[:,39] = (data[:,39] - self.min[39]) / (self.max[39] - self.min[39])
        data[:,40] = (data[:,40] - self.min[40]) / (self.max[40] - self.min[40])
        return np.transpose(data.reshape((T,W,-1)),(0,2,1))

    def inverse_transform(self,data):
        """
        Arguments:
            data {np.ndarray} --  shape(T, D, W, H)

        Returns:
            {np.ndarray} --  shape(T, D, W, H)
        """
        return data*(self.max[0]-self.min[0])+self.min[0]


def mask_loss(predicts,labels,region_mask,data_type="nyc"):
    """

    Arguments:
        predicts {Tensor} -- predict，(batch_size,pre_len,W,H)
        labels {Tensor} -- label，(batch_size,pre_len,W,H)
        region_mask {np.array} -- mask matrix，(W,H)
        data_type {str} -- nyc/chicago

    Returns:
        {Tensor} -- MSELoss,(1,)
    """
    _,_,w,h=predicts.shape
    if w==20:
        batch_size,pre_len,w,h = predicts.shape
        predicts=predicts.reshape(batch_size,w*h)
        predicts=mindspore.ops.MatMul(predicts,trans1)

    else:
        batch_size,pre_len,w,h = predicts.shape
        predicts=predicts.reshape(batch_size,w*h)
        predicts=mindspore.ops.MatMul(predicts,trans2)


    region_mask = mindspore.Tensor.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    predicts=predicts.reshape(batch_size,1,-1)
    loss = ((labels-predicts))**2

    return (mindspore.ops.ReduceMean(loss))


def mask_loss2(predicts,labels,region_mask):

    region_mask=np.dot(region_mask,trans3.transpose(1,0))
    region_mask = mindspore.Tensor.from_numpy(region_mask).to(predicts.device)
    region_mask /= region_mask.mean()
    region_mask=region_mask.reshape(10,10)
    region_mask=region_mask
    loss = ((labels-predicts))**2
    return mindspore.ops.ReduceMean(loss)

def setseed(seed):
    mindspore.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
