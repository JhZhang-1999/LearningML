import tensorflow as tf
import numpy as np
import pickle
import random

n=10000
d=3072
h=10



def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_4') # validation
tdata=np.array(batch_1[b'data'],dtype=int)
tlabel=np.array(batch_1[b'labels'])
vdata=np.array(batch_4[b'data'],dtype=int)
vlabel=np.array(batch_4[b'labels'])



