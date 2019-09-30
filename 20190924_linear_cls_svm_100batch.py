import numpy as np
import pickle

max_epoch=200
learning_rate=1e-06

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
    loss/=100 # 求所有图片的loss平均
    return loss

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
        for j in range(10):
            if s[i,j]-s[i,y[i]]+1>0: # s[i][j]和s[i][y[i]]]贡献了loss（分别为+1和-1）
                if j==y[i]:
                    gradb[0,j]-=(1/100)
                    for k in range(3072):
                        gradW[j,k]-=(x[i,k]/100)
                else:
                    gradb[0,j]+=(1/100) # note：必须写上第一维的0作为第一个index，否则报错
                    for k in range(3072):
                        gradW[j,k]+=(x[i,k]/100)
    loss=svm_loss(s) # 分数矩阵设置完成，计算当前分数矩阵的loss
    return loss,tacc,vacc,gradW,gradb

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