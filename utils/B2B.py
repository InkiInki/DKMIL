import os
import numpy as np
from utils.func_basic import print_progress_bar, kernel_rbf, dis_euclidean

np.set_printoptions(precision=6)


class B2B:
    """
    用于初始化数据集相关的包距离矩阵
    :param    data_name：      数据集名称，用于存储文件的命名
    :param    bags：           整个包空间，格式详见musk1+等数据集
    :param    b2b_type：       包之间距离度量的方式，已有的包括：平均Hausdorff ("ave")距离和
    :param    gamma:           rbf核的gamma值
    :param    mi_matrix：      mi距离的关联矩阵
    :param    b2b_save_home：  默认距离矩阵的存储主目录
    """

    def __init__(self, data_name, bags, b2b_type="ave", gamma=1, mi_matrix=None, mean_cov=None,
                 min_max_vector=None, b2b_save_home="../Data/Distance/"):
        """
        构造函数
        """
        # 传递的参数
        self._data_name = data_name
        self._bags = bags
        self._b2b_type = b2b_type
        self._gamma = gamma
        self._mi_matrix = mi_matrix
        self._mean_cov = mean_cov
        self._min_max_vector = min_max_vector
        self._b2b_save_home = b2b_save_home
        self.__initialize__b2b()

    def __initialize__b2b(self):
        """
        初始化函数
        """
        # 存储计算的距离矩阵
        self._dis = []
        # 获取距离矩阵的存储路径
        self._save_b2b_path = self._b2b_save_home + "b2b_" + self._data_name + '_' + self._b2b_type + ".npz"
        self._b2b_name = {"ave": "ave_hausdorff",
                          "csd": "cauchy_schwarz_divergence",
                          "max": "max_hausdorff",
                          "min": "min_hausdorff",
                          "mig": "mi_graph",
                          "mik": "mi_kernel",
                          "mad": "mahalanobis",
                          "sim": "simpler"}
        self.__compute_dis()

    def __compute_dis(self):
        """
        计算距离
        """
        if not os.path.exists(self._save_b2b_path):
            # 包的大小
            N = len(self._bags)
            dis = np.zeros((N, N))
            print("使用%s计算距离矩阵..." % self._b2b_name[self._b2b_type])

            for i in range(N):
                # 打印进度条
                print_progress_bar(i, N)
                # 包i和j的距离即j和i的距离
                for j in range(i, N):
                    if self._b2b_type == "ave":
                        dis[i, j] = dis[j, i] = ave_hausdorff(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
                    elif self._b2b_type == "max":
                        dis[i, j] = dis[j, i] = max_hausdorff(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
                    elif self._b2b_type == "min":
                        dis[i, j] = dis[j, i] = min_hausdorff(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
                    elif self._b2b_type == "mig":
                        assert self._mi_matrix is not None
                        dis[i, j] = dis[j, i] = mi_graph(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1],
                                                         self._mi_matrix[i], self._mi_matrix[j], self._gamma)
                    elif self._b2b_type == "mik":
                        assert self._min_max_vector is not None
                        dis[i, j] = dis[j, i] = mi_kernel(self._min_max_vector[i], self._min_max_vector[j], self._gamma)
                    elif self._b2b_type == "mad":
                        assert self._mean_cov is not None
                        dis[i, j] = dis[j, i] = mahalanobis(self._mean_cov[0][i], self._mean_cov[0][j],
                                                            self._mean_cov[1][i], self._mean_cov[1][j])
                    elif self._b2b_type == "sim":
                        dis[i, j] = dis[j, i] = simpler(self._bags[i][0][:, : -1], self._bags[j][0][:, : -1])
                    elif self._b2b_type == "csd":
                        dis[i, j] = dis[j, i] = cauchy_schwarz_divergence(self._bags[i][0][:, : -1],
                                                                          self._bags[j][0][:, : -1], self._gamma)
            print()
            # 排除nan值
            dis[dis == -np.pi] = np.max(dis)

            # 结束的时候需要换行一下
            np.savez(self._save_b2b_path, dis=dis)
        self._dis = np.load(self._save_b2b_path)['dis']

    def get_dis(self):
        """
        获取距离矩阵
        """
        return self._dis


class IsolationForest:
    """
        The designed isolation tree.
        :param
            mat:                                The given data matrix.
            feature_idx:                        The index of selected attributes.
            attribute_choose_mechanism:         The feature choose mechanism.
                Including "random", "cycle", and the default setting is "cycle".
            attribute_value_choose_mechanism:   The feature value choose mechanism.
                Including "random", "average", and the default setting is "random".
        """

    def __init__(self, mat, feature_idx=None, attribute_choose_mechanism="random",
                 attribute_value_choose_mechanism="random"):
        self.__mat = mat
        self.__feature_idx = feature_idx
        self.__attribute_choose_mechanism = attribute_choose_mechanism
        self.__attribute_value_choose_mechanism = attribute_value_choose_mechanism
        self.__m = 0
        self.__n = 0
        self.__idx_count = 0
        self.tree_ = None
        self.__init_isolation_tree()

    def __init_isolation_tree(self):
        """
        The initialize of IsolationTree
        """
        self.__m = len(self.__mat)
        self.__n = len(self.__mat[0])
        if self.__feature_idx is None:
            self.__feature_idx = np.arange(self.__n)
        self.tree_ = self.__get_tree(np.arange(self.__m), -1, "Root")

    def __get_tree(self, idx, height, flag):
        """
        Getting tree.
        """
        if len(idx) == 0:
            return
        elif len(idx) == 1:
            return Tree((idx[0], height + 1, flag), None, None)
        else:
            attribute_idx = self.__get_attribute_idx()
            attribute_arr = self.__mat[idx, attribute_idx]
            attribute_value = self.__get_attribute_value(attribute_arr)
            attribute_arr = list(set(attribute_arr))

            if len(attribute_arr) == 1:
                left_idx, right_idx = [idx[0]], idx[1:]
            else:
                left_idx, right_idx = self.__filter(idx, attribute_arr, attribute_value)

            return Tree((len(idx), height + 1, attribute_idx, attribute_value, flag),
                        self.__get_tree(left_idx, height + 1, "Left"),
                        self.__get_tree(right_idx, height + 1, "Right"))

    def __get_attribute_idx(self):
        """
        Getting attribute index.
        """
        if self.__attribute_choose_mechanism == "random":
            return np.random.choice(self.__feature_idx)
        elif self.__attribute_choose_mechanism == "cycle":
            if self.__idx_count == len(self.__feature_idx):
                self.__idx_count = 0

            ret_feature_idx = self.__feature_idx[self.__idx_count]
            self.__idx_count += 1

            return ret_feature_idx

    def __get_attribute_value(self, attribute_arr):
        """
        Taking a value from the specified attribute.
        """
        if self.__attribute_value_choose_mechanism == "random":
            return np.random.choice(attribute_arr)
        elif self.__attribute_value_choose_mechanism == "average":
            return np.average(attribute_arr)

    def __filter(self, idx, attribute_arr, attribute_value):
        """
        Filtering data.
        """
        ret_left, ret_right = [], []
        for idx_i, att_i in zip(idx, attribute_arr):
            if att_i < attribute_value:
                ret_left.append(idx_i)
            else:
                ret_right.append(idx_i)

        return ret_left, ret_right

    def show_tree(self):
        """
        Showing tree.
        """
        if self.tree_ is None:
            return
        tree = [self.tree_]

        while len(tree) > 0:
            node = tree.pop(0)
            value = node.value
            if len(value) == 5:
                # The non-leaf node.
                print("Len: %d; layer: %d; attribute idx: %d; attribute value: %.2f; flag: %s" % value)
            else:
                print("Data idx: %d; layer: %d; flag: %s" % value)

            if node.left is not None:
                tree.append(node.left)

            if node.right is not None:
                tree.append(node.right)


class Tree:
    """
    数据结构中的树
    """

    def __init__(self, value, left, right):
        self.value = value
        self.left = left
        self.right = right


def ave_hausdorff(bag1, bag2):
    """
    平均Hausdorff距离，相关文献可以参考：
        "Multi-instance clustering with applications to multi-instance prediction."
    :param
        bag1:   数据包1，需要使用numpy格式，形状为$n1 \times d$，其中$n1$为包的大小，$d$为实例的维度
        bag2：   类似于包1
    :return
        两个包的距离度量
    """
    # 统计总距离值
    sum_dis = 0
    for ins1 in bag1:
        # 计算当前实例与最近实例的距离
        temp_min = np.inf
        for ins2 in bag2:
            temp_min = min(dis_euclidean(ins1, ins2), temp_min)
        sum_dis += temp_min

    for ins2 in bag2:
        temp_min = np.inf
        for ins1 in bag1:
            temp_min = min(dis_euclidean(ins2, ins1), temp_min)
        sum_dis += temp_min

    return sum_dis / (len(bag1) + len(bag2))


def cardinality_potential_kernel(bag1, bag2, cardinality_svm, gamma=1):
    """"""
    ret_kernel = 1
    bag = np.vstack((bag1, bag2))
    for ins in bag:
        ret_kernel *= cardinality_svm.predict_proba(np.reshape(ins, (1, -1)))[0][0] * kernel_rbf(ins, gamma=gamma)

    return ret_kernel


def cauchy_schwarz_divergence(bag1, bag2, gamma=1):
    """"""
    ret_dis = 0
    for ins1 in bag1:
        for ins2 in bag2:
            ret_dis += kernel_rbf(ins1, ins2, gamma)

    return ret_dis * gamma / np.pi


def comformal_kernel(bag1, bag2, centers):
    """"""
    ret_sim = 0
    for center in centers:
        for ins1 in bag1:
            for ins2 in bag2:
                ret_sim += kernel_rbf(ins1, center) * kernel_rbf(ins2, center) * kernel_rbf(ins1, ins2)

    return ret_sim / len(bag1) / len(bag2)


def max_hausdorff(bag1, bag2):
    """
    最大Hausdorff距离
    """
    len_bag1 = len(bag1)
    len_bag2 = len(bag2)

    temp_max1 = -1
    for i in range(len_bag1):
        temp_min = np.inf
        for j in range(len_bag2):
            temp_dis = dis_euclidean(bag1[i], bag2[j])
            temp_min = min(temp_dis, temp_min)
        temp_max1 = max(temp_min, temp_max1)

    temp_max2 = -1
    for j in range(len_bag2):
        temp_min = np.inf
        for i in range(len_bag1):
            temp_dis = dis_euclidean(bag2[j], bag1[i])
            temp_min = min(temp_dis, temp_min)
        temp_max2 = max(temp_min, temp_max2)

    return max(temp_max1, temp_max2)


def min_hausdorff(bag1, bag2):
    """
    最小Hausdorff距离
    """
    len_bag1 = len(bag1)
    len_bag2 = len(bag2)

    ret_min = np.inf
    for i in range(len_bag1):
        for j in range(len_bag2):
            ret_min = min(ret_min, dis_euclidean(bag1[i], bag2[j]))

    return ret_min


def marginalized_kernel(bag1, bag2, h1, h2, gamma=1):
    """"""
    C1, C2 = [], []
    for ins1 in bag1:
        c = h1 * kernel_rbf(ins1, h1)
        C1.append(c)
    for ins1 in bag1:
        c = h2 * kernel_rbf(ins1, h2)
        C2.append(c)
    C1, C2 = np.array(C1), np.array(C2)

    ret_kernel = 0
    for ins1, c1 in zip(bag1, C1):
        for ins2, c2 in zip(bag2, C2):
            # ret_kernel += kernel_rbf(ins1, ins2, gamma) * kernel_rbf(C1, C2, gamma) * h1 * h2
            ret_kernel += kernel_rbf(ins1, ins2, gamma) * h1 * h2

    return ret_kernel


def simpler(bag1, bag2):
    """
    使用包的均值向量来代表包
    """
    return dis_euclidean(np.average(bag1, axis=0), np.average(bag2, axis=0))


def mi_graph(bag1, bag2, mi_matrix1, mi_matrix2, gamma=1):
    """
    mi图核
    """
    number_row_i, num_row_j = len(mi_matrix1), len(mi_matrix2)
    numerator = 0
    for a in range(number_row_i):
        for b in range(num_row_j):
            numerator += (1 / (np.sum(mi_matrix1[a]) * np.sum(mi_matrix2[b]))) * \
                         kernel_rbf(bag1[a], bag2[b], gamma=gamma)
    denominator_1 = 0
    for a in range(number_row_i):
        denominator_1 += 1 / np.sum(mi_matrix1[a])
    denominator_2 = 0
    for b in range(num_row_j):
        denominator_2 += 1 / np.sum(mi_matrix2[b])
    denominator = denominator_1 + denominator_2
    return numerator / denominator


def mi_kernel(min_max_vector1, min_max_vector2, gamma=1, p=1):
    """
    MI核
    """
    return np.power(kernel_rbf(min_max_vector1, min_max_vector2, gamma) + 1, p)


def mahalanobis(mean1, mean2, cov1, cov2):
    """
    多示例马氏距离
    """
    mean = mean1 - mean2
    cov = np.matrix(cov1 / 2 + cov2 / 2 + np.identity(len(cov1)) * 1e-6).I
    dis = np.dot(np.dot(mean, cov), mean)
    return -np.pi if np.isnan(dis) or dis == np.inf else dis


def isk(bag1, bag2, forest):
    """"""
    dis = 0
    for i in range(len(bag1)):
        temp_dis = 0
        for j in range(len(bag2)):
            temp_dis += isk_i2i(bag1[i], bag2[j], forest)
        dis += temp_dis / len(bag2)

    return dis / len(bag1)


def isk_i2i(arr1, arr2, forest):
    """
    Compute the similarity between two arrays by using the isolation kernel.
    """
    temp_num_forest = len(forest)
    temp_count = 0
    for i in range(temp_num_forest):
        if isk_tree(arr1, arr2, forest[i]):
            temp_count += 1

    return temp_count / temp_num_forest


def isk_tree(arr1, arr2, tree):
    """
    The flag function \mathbb{I}.
    """
    if tree is not None:
        temp_value = tree.value

        if len(temp_value) == 3:
            return True
        temp_attribute_idx, temp_threshold = temp_value[2:4]
        if arr1[temp_attribute_idx] < temp_threshold <= arr2[temp_attribute_idx]:
            return False
        if arr1[temp_attribute_idx] >= temp_threshold > arr2[temp_attribute_idx]:
            return False
        if tree.left is not None and \
                arr1[temp_attribute_idx] < temp_threshold and arr2[temp_attribute_idx] < temp_threshold:
            return isk_tree(arr1, arr2, tree.left)
        if tree.right is not None and \
                arr1[temp_attribute_idx] >= temp_threshold and arr2[temp_attribute_idx] >= temp_threshold:
            return isk_tree(arr1, arr2, tree.right)
    return True
