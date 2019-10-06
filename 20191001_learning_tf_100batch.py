import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import math
import random
import matplotlib.pyplot as plt

def convert(label,size):
    output=np.zeros([size,n_classes])
    for i in range(size):
        output[i,label[i]]=1 # 第二维度只有正确标签是1其余是0
    return output

n=500 # sample size
n_input=3072 # features
n_classes=10 # classes
lr=0.5 # learning rate
max_epoch=10000
batch_size=100
n_hidden=10
seed=0

def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_4') # validation
tdata=np.array(batch_1[b'data'][:500],dtype=int)
tlabel=np.array(batch_1[b'labels'][:500])
tlabel=convert(tlabel,len(tlabel))
vdata=np.array(batch_4[b'data'][:500],dtype=int)
vlabel=np.array(batch_4[b'labels'][:500])
vlabel=convert(vlabel,len(vlabel))

'''
W=tf.Variable(tf.random_normal(shape=[p,n],mean=0,stddev=1),name='Weights')
b=tf.Variable(tf.zeros([p,1]),name='bias')
'''

def sigmaprime(x):
    return tf.multiply(tf.sigmoid(x),tf.subtract(tf.constant(1.0),tf.sigmoid(x)))

x_in=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])

def multilayer_perceptron(x,weights,biases):
    h_layer_1=tf.add(tf.matmul(x,weights['h1']),biases['h1'])
    out_layer_1=tf.sigmoid(h_layer_1)
    h_out=tf.matmul(out_layer_1,weights['out'])+biases['out']
    return tf.sigmoid(h_out),h_out,out_layer_1,h_layer_1

weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden],seed=seed)),
    'out':tf.Variable(tf.random_normal([n_hidden,n_classes],seed=seed))
}

biases={
    'h1':tf.Variable(tf.random_normal([1,n_hidden],seed=seed)),
    'out':tf.Variable(tf.random_normal([1,n_classes],seed=seed))
}

# forward pass
y_hat,h_2,o_1,h_1=multilayer_perceptron(x_in,weights,biases)

# error
err=y_hat-y

# backward pass
delta_2=tf.multiply(err,sigmaprime(h_2))
delta_w_2=tf.matmul(tf.transpose(o_1),delta_2)
wtd_error=tf.matmul(delta_2,tf.transpose(weights['out']))
delta_1=tf.multiply(wtd_error,sigmaprime(h_1))
delta_w_1=tf.matmul(tf.transpose(x_in),delta_1)
eta=tf.constant(lr)

# update weights
step=[
    tf.assign(weights['h1'],tf.subtract(weights['h1'], tf.multiply(eta,delta_w_1))),tf.assign(biases['h1'],tf.subtract(biases['h1'],tf.multiply(eta,tf.reduce_mean(delta_1,axis=[0])))),tf.assign(weights['out'],tf.subtract(weights['out'],tf.multiply(eta,delta_w_2))),tf.assign(biases['out'],tf.subtract(biases['out'],tf.multiply(eta,tf.reduce_mean(delta_2,axis=[0]))))
]

acct_mat=tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))
accuracy=tf.reduce_sum(tf.cast(acct_mat,tf.float32))

init=tf.global_variables_initializer()

def my_next_batch(size):
    random_index=random.sample([i for i in range(n)],size)
    x_batch=tdata[random_index] # 取出size*n_input这么大
    y_batch=tlabel[random_index] # 取出size这么大
    return x_batch,y_batch

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(max_epoch):
        batch_xs,batch_ys=my_next_batch(batch_size)
        sess.run(step,feed_dict={x_in:batch_xs,y:batch_ys})
        acc_test=sess.run(accuracy,feed_dict={x_in:vdata,y:vlabel})
        acc_train=sess.run(accuracy,feed_dict={x_in:tdata,y:tlabel})
        if epoch%10!=0:
            continue
        print(epoch,",",acc_train,",",acc_test)