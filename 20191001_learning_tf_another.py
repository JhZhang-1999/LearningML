import tensorflow as tf
import numpy as np
import pickle
import random

##### 这个版本的tf代码是抄的https://www.cnblogs.com/lizheng114/p/7439556.html，能够在9990代跑到45%和38%的准确率。

def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

feature=3072
n=10000
classes=10
batch_size=100
max_epoch=10000

def convert(label,size):
    output=np.zeros([size,classes])
    for i in range(size):
        output[i,label[i]]=1 # 第二维度只有正确标签是1其余是0
    return output

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_4') # validation
tdata=np.array(batch_1[b'data'],dtype=int)
tlabel=np.array(batch_1[b'labels'])
tlabel=convert(tlabel,len(tlabel))
vdata=np.array(batch_4[b'data'],dtype=int)
vlabel=np.array(batch_4[b'labels'])
vlabel=convert(vlabel,len(vlabel))

# x是特征值
x = tf.placeholder(tf.float32, [None, feature])
# w表示每一个特征值（像素点）会影响结果的权重
W = tf.Variable(tf.zeros([feature, classes]))
b = tf.Variable(tf.zeros([classes]))
y = tf.matmul(x, W) + b
epoch= tf.placeholder(tf.float32)
# y_是图片实际对应的值
y_ = tf.placeholder(tf.float32, [None, classes])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
lr=1e-7
lr/=(10**(epoch//1000))

train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 训练数据

def my_next_batch(size):
    random_index=random.sample([i for i in range(n)],size)
    x_batch=tdata[random_index] # 取出size*n_input这么大
    y_batch=tlabel[random_index] # 取出size这么大
    return x_batch,y_batch

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

for epoch in range(max_epoch):
    batch_xs, batch_ys = my_next_batch(batch_size)
    epoch_arr=np.array([epoch])
    epoch_= tf.convert_to_tensor(epoch_arr, dtype = tf.float32)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,epoch:epoch_})
    if epoch%10!=0:
        continue
    print(epoch,",",end="")
    print(sess.run(accuracy, feed_dict={x: tdata, y_: tlabel}),",",end="")
    print(sess.run(accuracy, feed_dict={x: vdata, y_: vlabel}))
# 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确