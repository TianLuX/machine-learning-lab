import numpy as np
import pandas as pd

#鸢尾花数据集
from sklearn import datasets
#随机划分训练集和测试集合
from sklearn.model_selection import train_test_split

#选取测试集合比例
rate = 0.20
#处理鸢尾花数据，包括data和target,总共150组数据
#data数据包括sepal length、sepal width、petal length、petal width
def creat_data():
    iris = datasets.load_iris()
    #划分训练集和测试集
    train_data, test_data, train_label, test_label = \
        train_test_split(iris.data, iris.target, test_size=rate)
    # print(X_train, X_test, y_train, y_test)
    # print(iris)

    #训练集
    train_data = \
        pd.DataFrame(train_data,columns=['sepal length','sepal width','petal length','petal width'])
    train_data['label'] = train_label

    #测试集
    test_data = \
        pd.DataFrame(test_data,columns=['sepal length','sepal width','petal length','petal width'])
    test_data['label'] = test_label

    # print(train_data)
    return train_data, test_data

#统计mylabel类的先验概率
def classify(mylabel,data):
    label_count = data['label'][data['label']==mylabel].count()
    total = float(data['label'].count())
    return label_count/total

#由于鸢尾花的属性都为连续属性，计算mylabel下attr的均值和方差
def means_and_var(mylabel,attr,data):
    data_means = data[attr][data['label'] == mylabel].mean()
    data_variance = data[attr][data['label'] == mylabel].var()
    return data_means, data_variance

#计算P(x|y)
def p_x_when_y(x,mean_y,variance_y):
    p = 1/(np.sqrt((2*np.pi*variance_y))) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p


#开始分类
train_data, test_data = creat_data()

#三种分类的先验概率
labels = [0.0, 1.0, 2.0]
priori = []
for label in labels:
    priori.append(classify(label,train_data))

# label0 = classify(0.0,train_data)
# label1 = classify(1.0,train_data)
# label2 = classify(2.0,train_data)

#计算四个属性的均值和方差
mean = []
var = []

for label in labels:
    mean_tmp = []
    var_tmp = []
    for item in train_data.columns[:-1]:
        tmp_mean, tmp_var = means_and_var(label, item ,train_data)
        mean_tmp.append(tmp_mean)
        var_tmp.append(tmp_var)
    mean.append(mean_tmp)
    var.append(var_tmp)
# print(np.array(mean).shape)
# print(var)
#
# sepal_length_mean0,sepal_length_var0 = means_and_var(0.0,'sepal length',train_data)
# sepal_width_mean0,sepal_width_var0 = means_and_var(0.0,'sepal width',train_data)
# petal_length_mean0,petal_length_var0 = means_and_var(0.0,'petal length',train_data)
# petal_width_mean0,petal_width_var0 = means_and_var(0.0,'petal width',train_data)
#
# sepal_length_mean1,sepal_length_var1 = means_and_var(1.0,'sepal length',train_data)
# sepal_width_mean1,sepal_width_var1 = means_and_var(1.0,'sepal width',train_data)
# petal_length_mean1,petal_length_var1 = means_and_var(1.0,'petal length',train_data)
# petal_width_mean1,petal_width_var1 = means_and_var(1.0,'petal width',train_data)
#
# sepal_length_mean2,sepal_length_var2 = means_and_var(2.0,'sepal length',train_data)
# sepal_width_mean2,sepal_width_var2 = means_and_var(2.0,'sepal width',train_data)
# petal_length_mean2,petal_length_var2 = means_and_var(2.0,'petal length',train_data)
# petal_width_mean2,petal_width_var2 = means_and_var(2.0,'petal width',train_data)

#在测试集上进行预测
accurancy = 0
for i in range(test_data.shape[0]):
    actual = test_data['label'][i]
    predict = []
    for inx, label in enumerate(labels):
        predict_tmp = priori[inx]
        for j in range(len(test_data.columns) - 1):
            cur_mean = mean[inx][j]
            cur_var = var[inx][j]
            attr = str(test_data.columns[j])
            predict_tmp *= p_x_when_y(test_data[attr][i],cur_mean, cur_var)
        predict.append(predict_tmp)

    if np.argmax(np.array(predict)) == actual:
        accurancy += 1

print("选取鸢尾花测试集比例为" + str(rate) + "，预测准确率为{:.5f}".format(accurancy / test_data.shape[0]))
    # predict0 = label0*p_x_when_y(test_data['sepal length'][i],sepal_length_mean0,sepal_length_var0)\
    #            *p_x_when_y(test_data['sepal width'][i],sepal_width_mean0,sepal_width_var0)\
    #            *p_x_when_y(test_data['petal length'][i],petal_length_mean0,petal_length_var0)\
    #            *p_x_when_y(test_data['petal width'][i],petal_width_mean0,petal_width_var0)
    # predict1 = label1*p_x_when_y(test_data['sepal length'][i],sepal_length_mean1,sepal_length_var1)\
    #            *p_x_when_y(test_data['sepal width'][i],sepal_width_mean1,sepal_width_var1)\
    #            *p_x_when_y(test_data['petal length'][i],petal_length_mean1,petal_length_var1)\
    #            *p_x_when_y(test_data['petal width'][i],petal_width_mean1,petal_width_var1)
    # predict2 = label2*p_x_when_y(test_data['sepal length'][i],sepal_length_mean2,sepal_length_var2)\
    #            *p_x_when_y(test_data['sepal width'][i],sepal_width_mean2,sepal_width_var2)\
    #            *p_x_when_y(test_data['petal length'][i],petal_length_mean2,petal_length_var2)\
    #            *p_x_when_y(test_data['petal width'][i],petal_width_mean2,petal_width_var2)

    # if(max(predict0,predict1,predict2) == predict0):
    #     predict = 0
    # elif(max(predict0,predict1,predict2) == predict1):
    #     predict = 1
    # elif(max(predict0,predict1,predict2) == predict2):
    #     predict = 2
    # if(predict == label):
    #     accurancy += 1



