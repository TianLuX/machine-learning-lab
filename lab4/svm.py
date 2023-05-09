# coding:UTF-8
import numpy as np

# 鸢尾花数据集
from sklearn import datasets
# 随机划分训练集和测试集合
from sklearn.model_selection import train_test_split


# SVM类用于保存需要的参数
class SVM:
    def __init__(self, train_data, train_label, C, toler, kernel_option):
        """
        初始化部分
        :param train_data: 训练样本特征
        :param train_label: 训练样本标签
        :param C: 惩罚参数
        :param toler: 迭代终止条件之一
        :param kernel_option: 选用的核函数
        """
        self.train_x = train_data
        self.train_y = train_label
        self.C = C
        self.toler = toler

        self.sample_num = np.shape(train_data)[0]  # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.sample_num, 1)))  # 拉格朗日乘子，矩阵类型
        self.b = 0  # 偏置
        self.error_tmp = np.mat(np.zeros((self.sample_num, 2)))  # 保存E的缓存
        self.kernel_opt = kernel_option  # 选用的核函数及其参数
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt)  # 核函数的输出


def cal_kernel_value(train_data, train_data_i, kernel_option):
    """
    计算样本之间的核函数的值
    :param train_data: 训练样本
    :param train_data_i: 第i个训练样本
    :param kernel_option: 核函数的类型以及参数
    :return: 样本之间的核函数的值
    """
    kernel_type = kernel_option[0]  # 核函数的类型，分为rbf（高斯核函数）和其他
    sample_num = np.shape(train_data)[0]  # 样本的个数
    kernel_value = np.mat(np.zeros((sample_num, 1)))

    if kernel_type == 'rbf':  # rbf核函数
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(sample_num):  # 从0到样本数目循环
            diff = train_data[i, :] - train_data_i
            kernel_value[i] = np.exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:  # 不使用核函数
        kernel_value = train_data * train_data_i.T
    return kernel_value


def calc_kernel(train_x, kernel_option):
    """
    计算核函数矩阵
    :param train_x: 训练样本的特征值
    :param kernel_option: 核函数的类型以及参数
    :return: 样本的核函数矩阵
    """
    sample_num = np.shape(train_x)[0]  # 样本的个数
    kernel_matrix = np.mat(np.zeros((sample_num, sample_num)))  # 初始化样本之间的核函数矩阵
    for i in range(sample_num):
        kernel_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernel_option)  # 计算核函数的值
    return kernel_matrix


def cal_error(svm, alpha_k):
    """
    计算误差值
    :param svm: 模型
    :param alpha_k: 选择出的变量，对应alpha
    :return: 误差值
    """

    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernel_mat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


def update_error(svm, alpha_k):
    """
    更新误差值
    :param svm: 模型
    :param alpha_k: 选择出的变量，对应alpha
    :return: 误差值
    """

    error = cal_error(svm, alpha_k)  # 选择样本更新误差值
    svm.error_tmp[alpha_k] = [1, error]


def select_second_sample_j(svm, alpha_i, error_i):
    """
    选择第二个样本
    :param svm: 模型
    :param alpha_i: 选择出的第一个变量
    :param error_i: E_i
    :return: 选择出的第二个变量和E_j
    """

    # 标记为已被优化
    svm.error_tmp[alpha_i] = [1, error_i]
    candidateAlphaList = np.nonzero(svm.error_tmp[:, 0].A)[0]

    max_step = 0
    alpha_j = 0
    error_j = 0

    if len(candidateAlphaList) > 1:
        for alpha_t in candidateAlphaList:
            if alpha_t == alpha_i:
                continue
            error_t = cal_error(svm, alpha_t)
            if abs(error_t - error_i) > max_step:
                max_step = abs(error_t - error_i)
                alpha_j = alpha_t
                error_j = error_t
    else:
        alpha_j = alpha_i# 随机选择
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.sample_num))
        error_j = cal_error(svm, alpha_j)

    return alpha_j, error_j


def choose_and_update(svm, alpha_i):
    """
    选择两个alpha值进行更新
    :param svm: 模型
    :param alpha_i: 选择出的第一个变量
    :return:
    """

    error_i = cal_error(svm, alpha_i)  # 计算第一个样本的E_i

    # 判断选择出的第一个变量是否违反了KKT条件
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
            (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # 1、选择第二个变量
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 2、计算上下界
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            l = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            h = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            l = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            h = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if l == h:
            return 0

        # 3、计算eta
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] \
              - svm.kernel_mat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # 4、更新alpha_j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 5、确定最终的alpha_j
        if svm.alphas[alpha_j] > h:
            svm.alphas[alpha_j] = h
        if svm.alphas[alpha_j] < l:
            svm.alphas[alpha_j] = l

        # 6、判断是否结束
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error(svm, alpha_j)
            return 0

        # 7、更新alpha_i
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])

        # 8、更新b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
             * svm.kernel_mat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
             * svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
             * svm.kernel_mat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
             * svm.kernel_mat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # 9、更新error
        update_error(svm, alpha_j)
        update_error(svm, alpha_i)

        return 1
    else:
        return 0


def SVM_training(train_data, train_label, C, toler, max_iter, kernel_option=('rbf', 0.431029)):
    """
    训练SVM
    :param train_data: 训练数据
    :param train_label: 训练数据的标签
    :param C: 惩罚因子
    :param toler: 迭代的终止条件之一
    :param max_iter: 最大迭代次数
    :param kernel_option: 核函数的类型及其参数
    :return: 训练完成的模型
    """

    # 1、初始化
    svm = SVM(train_data, train_label, C, toler, kernel_option)

    # 2、开始训练
    entireSet = True #整个样本集合
    alpha_pairs_changed = 0
    iteration = 0  # 迭代次数

    while (iteration < max_iter) and ((alpha_pairs_changed > 0) or entireSet):
        print("第 {} 迭代 ".format(iteration))
        alpha_pairs_changed = 0

        if entireSet:
            # 对所有的样本
            for x in range(svm.sample_num):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            # 非边界样本
            bound_samples = []
            for i in range(svm.sample_num):
                if svm.alphas[i, 0] > 0 and svm.alphas[i, 0] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1

        # 在所有样本和非边界样本之间交替
        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True

    return svm


def predict_one(svm, test_data_i):
    """
    依次预测各个样本
    :param svm: 模型
    :param test_data_i: 测试用例
    :return: 预测结果
    """
    # 预测步骤
    # 计算核函数矩阵
    kernel_value = cal_kernel_value(svm.train_x, test_data_i, svm.kernel_opt)
    # 计算预测值
    predict_label = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict_label


def cal_accuracy(svm, test_data, test_label):
    """
    计算预测准确度
    :param svm: 模型
    :param test_data: 测试样本
    :param test_label: 测试标签
    :return: 准确度
    """
    
    sample_num = np.shape(test_data)[0]  # 样本的个数
    correct = 0.0
    for i in range(sample_num):
        # 对每一个样本进行预测值
        predict = predict_one(svm, test_data[i, :])
        if np.sign(predict) == np.sign(test_label[i]):  # 判断正确则加一
            correct += 1
    accuracy = correct / sample_num
    return accuracy

#选取测试集合比例
rate = 0.20
#处理鸢尾花数据，包括data和target,总共150组数据
def creat_data():
    iris = datasets.load_iris()
    #划分训练集和测试集
    train_data, test_data, train_label, test_label = \
        train_test_split(iris.data, iris.target, test_size=rate)

    train_label_matrix = np.mat(np.zeros((len(train_label), 1)))  # 初始化样本之间的核函数矩阵
    for i in range(len(train_label)):
        train_label_matrix[i][0] = int(train_label[i] + 1)

    test_label_matrix = np.mat(np.zeros((len(test_label), 1)))  # 初始化样本之间的核函数矩阵
    for i in range(len(test_label)):
        test_label_matrix[i][0] = int(test_label[i] + 1)

    return np.mat(train_data), train_label_matrix, np.mat(test_data), test_label_matrix


if __name__ == "__main__":
    print("1、加载训练数据和测试数据")
    train_data, train_label, test_data, test_label = creat_data()
    print("2、设置相关参数，训练SVM模型")
    C = 1 #惩罚因子
    toler = 0.001 #迭代终止的边界值
    maxIter = 5 #最大迭代次数
    svm_model = SVM_training(train_data, train_label, C, toler, maxIter)
    print("3、使用训练完成的SVM模型计算准确率")
    accuracy = cal_accuracy(svm_model, test_data, test_label)
    print("准确率是: %.3f%%" % (accuracy * 100))




