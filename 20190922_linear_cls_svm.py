import numpy as np
import pickle

max_epoch=200
learning_rate=1e-07

### 变量说明
# W：10*3072维的权重矩阵
# b：10维的偏置向量
# x：10000*3072维的数据矩阵，x[i]是图片i的代表向量（3072维）
# y：10000维的标签向量，记录每张图的真标签，index是图的index，值是标签值（0-9）
# s：10000*10维的分数矩阵，s[i]是图i对每个标签的算法习得分数（10维）
# 线性分类器原理：s=Wx+b，即s[i]=W*x[i]+b

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
x=np.array(batch_1[b'data'],dtype=int)
y=np.array(batch_1[b'labels'])
x_val=np.array(batch_4[b'data'],dtype=int)
y_val=np.array(batch_4[b'labels'])

def svm_loss(s):
    loss=0
    for i in range(10000): # 外层循环遍历所有图片
        for j in range(10): # 内层循环按svm公式遍历每个标签
            if j!=y[i]:
                loss+=max(0,s[i,j]-s[i,y[i]]+1) # svm公式
    loss/=10000 # 求所有图片的loss平均
    return loss

def linear_cls(x,x_val,W,b):
    s=np.full((10000,10),0,dtype=float) # 初始化分数矩阵
    s_val=np.full((10000,10),0,dtype=float)
    gradb=np.zeros((1,10)) # 初始化b的梯度
    gradW=np.zeros((10,3072)) # 初始化W的梯度
    tacc=0
    vacc=0
    for i in range(10000):
        s[i]=np.dot(W,x[i])+b # 计算分数矩阵
        if np.argmax(s[i])==y[i]: # 对图片i分类正确
            tacc+=1
        s_val[i]=np.dot(W,x_val[i])+b
        if np.argmax(s_val[i])==y_val[i]:
            vacc+=1
        for j in range(10):
            if s[i,j]-s[i,y[i]]+1>0: # s[i][j]和s[i][y[i]]]贡献了loss（分别为+1和-1）
                if j==y[i]:
                    gradb[0,j]-=(1/10000)
                    for k in range(3072):
                        gradW[j,k]-=(x[i,k]/10000)
                else:
                    gradb[0,j]+=(1/10000) # note：必须写上第一维的0作为第一个index，否则报错
                    for k in range(3072):
                        gradW[j,k]+=(x[i,k]/10000)
    loss=svm_loss(s) # 分数矩阵设置完成，计算当前分数矩阵的loss
    return loss,tacc,vacc,gradW,gradb

for t in range(max_epoch):
    print("epoch",t,"starts:")
    loss,tacc,vacc,gradW,gradb=linear_cls(x,x_val,W,b)
    print("loss=",loss,"train acc=",tacc,"val acc=",vacc)
    print("—"*20)
    W-=gradW*learning_rate
    b-=gradb*learning_rate