import pickle
import numpy

k=31

def unpickle(file): # import the dataset
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_1')
#batch_2=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_2')
#batch_3=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_3')
batch_4=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_4')
#batch_5=unpickle(r'D:\LIFE\WORK\4-University\Others\NeuralNetworks\cs231n\LearningML\cifar-10-batches-py\data_batch_5')

def distanceL2(arr1,arr2): # L2 distance
    length=len(arr1)
    sum_dif=0
    for i in range(0,length):
        sum_dif+=pow((int(arr1[i])-int(arr2[i])),2)
    return sum_dif

def distanceL1(arr1,arr2): # L1 distance
    length=len(arr1)
    sum_dif=0
    for i in range(0,length):
        sum_dif+=abs(int(arr1[i])-int(arr2[i]))
    return sum_dif

def mode(arr):
    mydict=dict((elem,arr.count(elem)) for elem in arr) 
    mode=[k for k,v in mydict.items() if max(mydict.values())==v] # a gooooood way to search for the mode!
    if isinstance(mode,int):
        return mode
    else:
        return mode[0]

def knn_cal(data,traindata):
    traindata_size=len(traindata[b'labels'])
    min_distance=[distanceL2(data,traindata[b'data'][0])]*k # use the first distance as the benchmark
    neighbor_label=[traindata[b'labels'][0]]*k
    for i in range(0,traindata_size): # i to traversal all traindatas
        temp_distance=distanceL2(data,traindata[b'data'][i])
        if temp_distance<min_distance[k-1]: # this "if" inserts the new neighbor into the neighbor array
            for j in range(k-1,0,-1):
                if temp_distance<min_distance[j] and temp_distance>=min_distance[j-1]: # set the insert position
                    for t in range(k-1,j,-1):
                        min_distance[t]=min_distance[t-1]
                        neighbor_label[t]=neighbor_label[t-1]
                    min_distance[j]=temp_distance
                    neighbor_label[j]=traindata[b'labels'][i]
                if temp_distance<min_distance[0]:
                    for t in range(k-1,0,-1):
                        min_distance[t]=min_distance[t-1]
                        neighbor_label[t]=neighbor_label[t-1]
                    min_distance[0]=temp_distance
                    neighbor_label[0]=traindata[b'labels'][i]
    label=mode(neighbor_label)
    return label

def knn_main(traindata, valdata):
    result=[]
    #valdata_size=len(valdata[b'labels'])
    valdata_size=100
    for i in range(0,valdata_size): # i to traversal all valdatas
        temp_label=knn_cal(valdata[b'data'][i],traindata)
        result.append(temp_label)
    return result

result=knn_main(batch_1,batch_4)
correct=0
for i in range(len(result)):
    print(result[i],batch_4[b'labels'][i])
    if result[i]==batch_4[b'labels'][i]:
        correct+=1
print("correct data:",correct)

### notes for batches: 
# 4 keys: b'batch_label', b'labels', b'data', b'filenames'
# batch_label is b'training batch 1 of 5'
# labels are an array containing 10000 nums in range [0,10)
# data is a 10000*3072 array, each row represents a pic, 1024R + 1024G + 1024B
# filenames are names like 'category_s_000000.png'

### notes for batches_meta:
# 3 keys: b'num_cases_per_batch', b'label_names', b'num_vis'
# value = 10000, [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'], 3072 respectively