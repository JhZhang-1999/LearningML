import tensorflow as tf
from tensorflow import keras
import numpy as np
np.random.seed(1337)
import pickle

# 此版本抄的https://www.jianshu.com/p/e9c1e68a615e

feature_dim=1024

'''
# 一个kernel矩阵使用正则化因子为0.01的L1正则项的全连接层
layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# 一个偏差向量使用正则化因子为0.01的L2正则项的全连接层
layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))
# 一个使用随机正交矩阵初始化Kernel的全连接层
layers.Dense(64, kernel_initializer='orthogonal')
# 一个偏差初始化时全为2的全连接层
layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))
'''

def unpickle(file): # import the dataset
    import pickle
    with open(file,'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_1') # train
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\LearningML\cifar-10-batches-py\data_batch_4') # validation
X_train=np.array(batch_1[b'data'],dtype=int)
x_r=X_train[:,:1024]
x_g=X_train[:,1024:2048]
x_b=X_train[:,2048:]
X_train=(x_r+x_g+x_b)/3
y_train=np.array(batch_1[b'labels'])
X_test=np.array(batch_4[b'data'],dtype=int)
x_r=X_test[:,:1024]
x_g=X_test[:,1024:2048]
x_b=X_test[:,2048:]
X_test=(x_r+x_g+x_b)/3
y_test=np.array(batch_4[b'labels'])

# 数据预处理
X_train = X_train.reshape(X_train.shape[0],-1)/255. # 归一化
X_test = X_test.reshape(X_test.shape[0],-1)/255. 
y_train =keras.utils.to_categorical(y_train, num_classes=10) # 将类别标签化为0/1类别标签，即(数据量*类别数)维的矩阵
y_test =keras.utils.to_categorical(y_test, num_classes=10)

# 不使用model.add()，用以下方式也可以构建网络
model = keras.models.Sequential([
    keras.layers.Dense(500, input_dim=feature_dim,kernel_initializer='he_normal'),
    # kernel_regularizer=keras.regularizers.l2(0.01)
    keras.layers.Activation('relu'),
    #keras.layers.Dense(10, input_dim=feature_dim, kernel_regularizer=keras.regularizers.l2(0.01)),
    #keras.layers.Activation('relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax'),
])

# 定义优化器
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, verbose=0, mode='max', epsilon=0.005, cooldown=0, min_lr=0)
# optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# optimizer=keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy']) # metrics赋值为'accuracy'，会在训练过程中输出正确率

# 这次我们用fit()来训练网路
print('Training ------------')
model.fit(X_train, y_train, epochs=50, batch_size=64,callbacks=[reduce_lr])

print('\nTesting ------------')
# 评价训练出的网络
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)