import pandas as pd
from math import log

# 鸢尾花数据集
from sklearn import datasets
# 随机划分训练集和测试集合
from sklearn.model_selection import train_test_split

#选取测试集合比例
rate = 0.15
#处理鸢尾花数据，包括data和target,总共150组数据
def creat_data():
    iris = datasets.load_iris()
    # 划分训练集和测试集
    train_data, test_data, train_label, test_label = \
        train_test_split(iris.data, iris.target, test_size=rate)

    # 训练集
    train_data = \
        pd.DataFrame(train_data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    train_data['label'] = train_label

    # 测试集
    test_data = \
        pd.DataFrame(test_data, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    test_data['label'] = test_label

    return train_data, test_data

# 由于鸢尾花的属性都为连续属性，计算attr的均值,作为分类边界
def means(attr, data):
    data_means = data[attr].mean()
    return data_means

#计算熵
def entropy(dataSet):
    num_data = len(dataSet)
    labelCount = {}
    currentLabels = dataSet['label']

    for label in currentLabels:
        if label not in labelCount.keys():
            labelCount[label] = 1
        else:
            labelCount[label] += 1

    e = 0.0
    for key in labelCount:
        prob = float(labelCount[key]) / num_data  # 计算单个类的熵值
        e -= prob * log(prob, 2)  # 累加每个类的熵值
    return e

#对数据按照某个特征进行划分结果
def splitData(attr, data):
    mean = means(attr, data)
    return data[data[attr] >= mean], data[data[attr] < mean], mean

# 选择最优分类,计算信息增益
def chooseBestFeatureToSplit(data,attrs):
    # 此处的attrs属性是数组
    e = entropy(data)
    max_gain = 0.0
    max_attr = ''
    for attr in attrs:
        data1, data2, mean = splitData(attr, data)
        e1 = entropy(data1)
        e2 = entropy(data2)
        gain = e - len(data1)/(0.0+len(data)) * e1 - len(data2)/(0.0+len(data)) * e2
        if gain > max_gain:
            max_gain = gain
            max_attr = attr
    return max_attr

#构建决策树
def create_decision_tree(data, attrs):
    if entropy(data) < 0.1 or len(attrs) == 1:# 如果数据集值剩下了一类，直接返回 # 如果所有特征都已经切分完了，也直接返回
        labelCount = {}
        currentLabels = data['label']
        max = 0
        max_label = ''
        for label in currentLabels:
            if label not in labelCount.keys():
                labelCount[label] = 1
            else:
                labelCount[label] += 1
            if labelCount[label] > max:
                max = labelCount[label]
                max_label = label

        return max_label
    # 寻找最佳切分的特征
    attr = chooseBestFeatureToSplit(data, attrs)
    node = {attr: {}}
    attrs.remove(attr)
    # 递归调用，对每一个切分出来的取值递归建树
    data1, data2, mean = splitData(attr, data)
    node[attr]["大于等于"+str(mean)] = create_decision_tree(data1,attrs)
    node[attr]["小于"+str(mean)] = create_decision_tree(data2, attrs)
    return node


def classify(node, attrs, data):
    attr = list(node.keys())[0]# 获取当前节点判断的特征
    node = node[attr]
    pred = None
    for key in node:# 根据特征进行递归
        if '大于等于' in key:
            k = key
            mean = float(str(k).strip('大于等于'))
            if data[attr] >= mean:# 如果再往下依然还有子树，那么则递归，否则返回结果
                if isinstance(node[key], dict):
                    pred = classify(node[key], attrs, data)
                else:
                    pred = node[key]
        elif '小于' in key:
            k = key
            mean = float(str(k).strip('小于'))
            if data[attr] < mean:
                if isinstance(node[key], dict):
                    pred = classify(node[key], attrs, data)
                else:
                    pred = node[key]
    # 如果没有对应的分叉，则找到一个分叉返回
    if pred is None:
        for key in node:
            if not isinstance(node[key], dict):
                pred = node[key]
                break
    return pred


train_data, test_data= creat_data()
attrs = ['sepal length', 'sepal width', 'petal length', 'petal width']

tree = create_decision_tree(train_data, attrs)
print(tree)

acc = 0.0
for i in range(len(test_data)):
    predict = classify(tree, attrs, test_data.loc[i])
    if predict == test_data['label'][i]:
        acc += 1

print("选取鸢尾花测试集比例为" + str(rate) + "，预测准确率为{:.5f}".format(acc / len(test_data)))



