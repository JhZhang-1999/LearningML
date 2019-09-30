import numpy as np
import pickle
import math

max_epoch=1000
learning_rate=5e-07

W=0.0001*np.random.randn(10,3072)
b=np.zeros((1,10))
# 参数初始化，参考：https://blog.csdn.net/raby_gyl/article/details/77879030

def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_4') # validation
x=np.array(batch_1[b'data'][:100],dtype=int)
y=np.array(batch_1[b'labels'][:100])
x_val=np.array(batch_4[b'data'][:100],dtype=int)
y_val=np.array(batch_4[b'labels'][:100])

def svm_loss(s):
    loss=0
    for i in range(100): # 外层循环遍历所有图片
        for j in range(10): # 内层循环按svm公式遍历每个标签
            if j!=y[i]:
                loss+=max(0,s[i,j]-s[i,y[i]]+1) # svm公式
    loss/=20 # 求所有图片的loss平均
    return loss

def soft_loss(s):
    loss=0
    grads=np.full((100,10),0,dtype=float) # 初始化s的梯度矩阵
    for i in range(100):
        m=max(s[i])
        deno=0
        for j in range(10):
            deno+=math.exp(s[i,j]-m) # 此处使用了防止overflow的公式：https://blog.csdn.net/csuzhaoqinghui/article/details/79742685
        for j in range(10):
            if j==y[i]:
                grads[i,j]=math.exp(s[i,j]-m)/deno-1
            else:
                # grads[i,j]=1/(math.exp(s[i,y[i]]-m)*deno) # 原始grads公式
                grads[i,j]=math.exp(s[i,j]-m)/deno
        loss-=(s[i,y[i]]-m)
        loss+=math.log(deno) # 此处使用了防止underflow的公式：https://blog.csdn.net/csuzhaoqinghui/article/details/79742685
    loss/=100
    return loss,grads

def linear_cls(x,x_val,W,b):
    s=np.full((100,10),0,dtype=float) # 初始化分数矩阵
    s_val=np.full((100,10),0,dtype=float)
    gradb=np.zeros((1,10)) # 初始化b的梯度
    gradW=np.zeros((10,3072)) # 初始化W的梯度
    tacc=0
    vacc=0
    for i in range(100):
        s[i]=np.dot(W,x[i])+b # 计算分数矩阵
        if np.argmax(s[i])==y[i]: # 对图片i分类正确
            tacc+=1
        s_val[i]=np.dot(W,x_val[i])+b
        if np.argmax(s_val[i])==y_val[i]:
            vacc+=1
    loss,grads=soft_loss(s) # 分数矩阵设置完成，计算当前分数矩阵的loss和s带来的梯度
    for i in range(100):
        for j in range(10):
            gradb[0,j]+=grads[i,j]/100
            for k in range(3072):
                gradW[j,k]+=grads[i,j]*x[i,k]/100
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