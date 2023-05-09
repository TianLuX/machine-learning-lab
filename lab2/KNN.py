import numpy as np
from collections import Counter

#鸢尾花数据集
from sklearn import datasets
#随机划分训练集和测试集合
from sklearn.model_selection import train_test_split


#选取测试集合比例
rate = 0.20
#处理鸢尾花数据，包括data和target,总共150组数据
def creat_data():
    iris = datasets.load_iris()
    #划分训练集和测试集
    train_data, test_data, train_label, test_label = \
        train_test_split(iris.data, iris.target, test_size=rate)

    return train_data, train_label,test_data,test_label

#距离函数
def Distance(test_once,train_data):
    distance = []
    for train in train_data:
        #分别计算测试数据和每个训练数据的使用欧氏距离
        t = np.linalg.norm(train-test_once)
        distance.append(t)
    return distance

#KNN
def KNN(test_once,train_data,train_lable,k):
    #计算测试数据与训练数据之间的距离
    distance = Distance(test_once,train_data)
    #返回最近距离的索引
    Kindex = np.argpartition(distance, k)[0:k]
    lables = {0:0,1:0,2:0}
    max = 0
    maxIndex = -1
    for i in Kindex:
        lables[train_lable[i]] += 1
        if max< lables[train_lable[i]]:
            max = lables[train_lable[i]]
            maxIndex = train_lable[i]
    return maxIndex


#进行测试,逐个进行测试
train_data, train_label,test_data,test_label = creat_data()
acc = 0
#k值一般取值为1或者3
k = 1
for i in range(len(test_data)):
    predict = KNN(test_data[i],train_data,train_label,k)
    if predict == test_label[i]:
        acc += 1

print("选取鸢尾花测试集比例为" + str(rate) + ",选择的k值为" + str(k) +
      "，预测准确率为{:.5f}".format(acc / test_data.shape[0]))


