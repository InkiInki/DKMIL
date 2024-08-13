# coding: utf-8
"""
作者: 因吉
邮箱: inki.yinji@qq.com
创建: 2021 0805
修改: 2021 0806
"""


import numpy as np
from scipy.io import loadmat
from numba import jit


def get_iter(tr, tr_lab, te, te_lab):
    """
    获取单词迭代器
    :param  tr:     训练集
    :param  tr_lab: 训练集标签
    :param  te:     测试集
    :param  te_lab: 测试集标签
    :return 相应迭代器
    """
    yield tr, tr_lab, te, te_lab


def get_k_cv_idx(num_x, k=5, seed=None):
    """
    获取k次交叉验证的索引
    :param num_x:       数据集的大小
    :param k:           决定使用多少折的交叉验证
    :param seed:        随机种子
    :return:            训练集索引，测试集索引
    """
    # 随机初始化索引
    if seed is not None:
        np.random.seed(seed)
    rand_idx = np.random.permutation(num_x)
    # 每一折的大小
    fold = int(np.floor(num_x / k))
    ret_tr_idx = []
    ret_te_idx = []
    for i in range(k):
        # 获取当前折的训练集索引
        tr_idx = rand_idx[0: i * fold].tolist()
        tr_idx.extend(rand_idx[(i + 1) * fold:])
        ret_tr_idx.append(tr_idx)
        # 添加当前折的测试集索引
        ret_te_idx.append(rand_idx[i * fold: (i + 1) * fold].tolist())
    return ret_tr_idx, ret_te_idx


def get_performance(type_performance):
    """
    获取分类性能度量
    :param type_performance: 分类性能度量指标
    :return: 分类性能度量函数
    """
    ret_per = {}
    for type_per in type_performance:
        if type_per == "acc":
            from sklearn.metrics import accuracy_score
            metric = accuracy_score
        else:
            from sklearn.metrics import f1_score
            metric = f1_score
        ret_per[type_per] = metric

    return ret_per


def print_progress_bar(idx, size):
    """
    打印进度条
    :param idx:    当前位置
    :param size:   总进度
    """
    print('\r' + '▇' * int(idx // (size / 50)) + str(np.ceil((idx + 1) * 100 / size)) + '%', end='')


def load_file(data_path):
    """
    载入.mat类型的多示例数据集
    :param data_path:  数据集的存储路径
    """
    return loadmat(data_path)['data']


# @jit(nopython=True)
def dis_euclidean(arr1, arr2):
    """"""
    return np.sqrt(np.sum((arr1 - arr2)**2))


# @jit(nopython=True)
def kernel_rbf(arr1, arr2=None, gamma=1):
    """"""
    if arr2 is None:
        return np.sqrt(np.sum(arr1**2))
    return np.exp(-gamma * dis_euclidean(arr1, arr2))


def project_perturbation(data_point, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def get_bag_label(data_loader):
    bags = []
    labels = []
    for bag, label in data_loader:
        bags.append(bag)
        labels.append(label)

    return bags, labels


def get_iter(tr, tr_lab, te, te_lab):
    yield tr, tr_lab, te, te_lab


def print_acc_and_recall(acc_list, f_acc_list, recall_list, f_recall_list, f_new_acc_list=None):
    print("Acc: %.3lf, %.3lf" % (np.average(acc_list), np.std(acc_list, ddof=1)))
    print("Declined Acc: %.3lf, %.3lf" % (np.average(acc_list) - np.average(f_acc_list), np.std(f_acc_list, ddof=1)))
    print("Recall: %.3lf, %.3lf" % (np.average(recall_list), np.std(recall_list, ddof=1)))
    print("Declined Recall: %.3lf, %.3lf" % (
    np.average(recall_list) - np.average(f_recall_list), np.std(f_recall_list, ddof=1)))
    if f_new_acc_list is not None:
        print("Bias: %.3lf, %.3lf" % (np.average(f_new_acc_list), np.std(f_new_acc_list, ddof=1)))


def write2file(file_path, context, mode="a+"):
    with open(file_path, mode) as log_txt:
        print(context, end="")
        log_txt.write(context)
    log_txt.close()


if __name__ == '__main__':
    path = r"D:\Data\VAD\Avenue\Avenue Dataset\testing_label_mask\1_label.mat"
    data = loadmat(path)["volLabel"][0]
    for i in range(len(data)):
        print(i, data[i].sum())
