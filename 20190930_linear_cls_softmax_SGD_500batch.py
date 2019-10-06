import numpy as np
import pickle
import math
import random

max_epoch=1000
learning_rate=1e-06

### 变量说明
# W：10*3072维的权重矩阵
# b：10维的偏置向量
# x：10000*3072维的数据矩阵，x[i]是图片i的代表向量（3072维）
# y：10000维的标签向量，记录每张图的真标签，index是图的index，值是标签值（0-9）
# s：10000*10维的分数矩阵，s[i]是图i对每个标签的算法习得分数（10维）
# 线性分类器原理：s=Wx+b，即s[i]=W*x[i]+b
# s_mini: 256*10维分数矩阵，由10000图片中每次随机抽样256个得来

W=0.0001*np.random.randn(10,3072)
b=np.zeros((1,10))
# 参数初始化，参考：https://blog.csdn.net/raby_gyl/article/details/77879030

def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_4') # validation
x=np.array(batch_1[b'data'][:500],dtype=int)
y=np.array(batch_1[b'labels'][:500])
x_val=np.array(batch_4[b'data'][:500],dtype=int)
y_val=np.array(batch_4[b'labels'][:500])

def svm_loss(s):
    loss=0
    for i in range(500): # 外层循环遍历所有图片
        for j in range(10): # 内层循环按svm公式遍历每个标签
            if j!=y[i]:
                loss+=max(0,s[i,j]-s[i,y[i]]+1) # svm公式
    loss/=10000 # 求所有图片的loss平均
    return loss

def soft_loss(s,batch_index):
    loss=0
    grads=np.full((500,10),0,dtype=float) # 初始化s的梯度矩阵
    for i in range(500):
        m=max(s[i])
        deno=0
        for j in range(10):
            deno+=math.exp(s[i,j]-m) # 此处使用了防止overflow的公式：https://blog.csdn.net/csuzhaoqinghui/article/details/79742685
        loss-=(s[i,y[i]]-m)
        loss+=math.log(deno) # 此处使用了防止underflow的公式：https://blog.csdn.net/csuzhaoqinghui/article/details/79742685
    loss/=500 # loss计算完毕
    for i in batch_index:
        m=max(s[i])
        deno=0
        for j in range(10):
            deno+=math.exp(s[i,j]-m)
        for j in range(10):
            if j==y[i]:
                grads[i,j]=math.exp(s[i,j]-m)/deno-1
            else:
                # grads[i,j]=1/(math.exp(s[i,y[i]]-m)*deno) # 原始grads公式
                grads[i,j]=math.exp(s[i,j]-m)/deno
    return loss,grads

def linear_cls(x,x_val,W,b):
    s=np.full((500,10),0,dtype=float) # 初始化分数矩阵
    s_val=np.full((500,10),0,dtype=float)
    gradb=np.zeros((1,10)) # 初始化b的梯度
    gradW=np.zeros((10,3072)) # 初始化W的梯度
    tacc=0
    vacc=0
    for i in range(500):
        s[i]=np.dot(W,x[i])+b # 计算分数矩阵
        if np.argmax(s[i])==y[i]: # 对图片i分类正确
            tacc+=1
        s_val[i]=np.dot(W,x_val[i])+b
        if np.argmax(s_val[i])==y_val[i]:
            vacc+=1
    random_index=random.sample([i for i in range(500)],256) # 抽出256个index
    loss,grads=soft_loss(s,random_index) # 分数矩阵设置完成，计算当前分数矩阵的loss和s带来的梯度
    for i in range(500):
        for j in range(10):
            gradb[0,j]+=grads[i,j]/500
            for k in range(3072):
                gradW[j,k]+=grads[i,j]*x[i,k]/500
    return loss,tacc,vacc,gradW,gradb

print("epoch, loss, train acc, val acc")
for t in range(max_epoch):
    print(t,end="")
    print(",",end="")
    loss,tacc,vacc,gradW,gradb=linear_cls(x,x_val,W,b)
    print(loss,end="")
    print(",",end="")
    print(tacc,end="")
    print(",",end="")
    print(vacc)
    W-=gradW*learning_rate
    b-=gradb*learning_rate