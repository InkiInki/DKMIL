import warnings
import numpy as np
import os as os
from utils.func_basic import (
    load_file,
    kernel_rbf,
)
warnings.filterwarnings("ignore")


class MIL:

    """
    多示例学习的原型类
    :param
        data_path：   数据集的存储路径
        save_home：   中间数据，例如距离矩阵的存储主目录
        bag_space：   格式与.mat文件一致
                      需要注意的是，当bag_space为None时，将读取给定目录下的文件
                      否则，将使用bag_space中的数据，但是依然需要传递文件名，以获取相对应的距离矩阵
    :attributes
        data_name：   数据集的名称
        bag_space：   包空间，详细格式请查看../Data/Benchmark/musk1+.mat
        ins_space：   实例空间
        bag_size：    记录每个包大小的向量，长度为N
        bag_lab：     包标签向量
        ins_lab：     实例标签
        bag_idx：     包索引向量
        ins_idx：     实例空间中 包所对应的实例的范围
        ins_bag_idx： 实例空间中 实例对应的包的序号
        zero_ratio：  数据集含零比率
        N：           包空间的大小
        n：           实例数量
        d：           实例的维度
        C：           数据集的类别树
    """
    def __init__(self, data_path, save_home="D:/Data/Distance/", bag_space=None):
        self.data_path = data_path
        self.save_home = save_home
        self.bag_space = bag_space
        self.__init_mil()

    def __init_mil(self):
        """
        初始化函数
        """
        if self.bag_space is None:
            self.bag_space = load_file(self.data_path)
        # elif self.bag_space == "test":
        #     from gendata_gulfport import merge
        #     self.bag_space = merge()
        self.N = len(self.bag_space)

        self.bag_size = np.zeros(self.N, dtype=int)
        self.bag_lab = np.zeros_like(self.bag_size, dtype=int)

        self.bag_idx = np.arange(self.N)
        for i in range(self.N):
            self.bag_size[i] = len(self.bag_space[i][0])
            self.bag_lab[i] = self.bag_space[i][1]
        # 将所有包的标签调整到 [0, C - 1]的范围，C为数据集的类别数
        self.__bag_lab_map()

        self.n = sum(self.bag_size)
        self.d = len(self.bag_space[0][0][0]) - 1
        self.C = len(list(set(self.bag_lab)))

        self.ins_space = np.zeros((self.n, self.d))
        self.ins_idx = np.zeros(self.N + 1, dtype=int)
        self.ins_lab = np.zeros(self.n, dtype=int)
        self.ins_bag_idx = np.zeros(self.n, dtype=int)
        for i in range(self.N):
            self.ins_idx[i + 1] = self.bag_size[i] + self.ins_idx[i]
            self.ins_space[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i][0][:, :self.d]
            self.ins_lab[self.ins_idx[i]: self.ins_idx[i + 1]] = self.bag_space[i][0][:, -1]
            self.ins_bag_idx[self.ins_idx[i]: self.ins_idx[i + 1]] = np.ones(self.bag_size[i]) * i

        self.data_name = self.data_path.strip().split("/")[-1].split(".")[0]
        self.zero_ratio = len(self.ins_space[self.ins_space == 0]) / (self.n * self.d)
        self.__generate_save_home()

    def __generate_save_home(self):
        """
        Generate the save home.
        """
        if not os.path.exists(self.save_home):
            os.makedirs(self.save_home)

    def __bag_lab_map(self):
        """
        将包的标签统一映射到[0..C - 1]的范围，其中C为数据集类别数
        """
        lab_list = list(set(self.bag_lab))
        lab_dict = {}
        for i, lab in enumerate(lab_list):
            lab_dict[lab] = i
        for i in range(self.N):
            self.bag_lab[i] = lab_dict[self.bag_lab[i]]

    def get_info(self):
        """
        打印输出数据集信息
        """
        temp_idx = 5 if self.N > 5 else self.N
        print("数据集{}的信息如下：".format(self.data_name), "\n"
              "包总数：", self.N, "\n"
              "类别数：", self.C, "\n"
              "包大小", self.bag_size[:temp_idx], "...\n"
              "包标签：", self.bag_lab[:temp_idx], "...\n"
              "最大包大小：", np.max(self.bag_size), "\n"
              "最小包大小：", np.min(self.bag_size), "\n"
              "含零率：", self.zero_ratio, "\n"
              "实例数：", self.n, "\n"
              "实例维度：", self.d, "\n"
              "实例索引：", self.ins_idx[: temp_idx], "...\n"
              "实例标签：", self.ins_lab[: temp_idx], "...\n"
              "实例原始包索引：", self.ins_bag_idx[:temp_idx], "...\n")

    def get_sub_ins_space(self, bag_idx):
        """
        根据索引获取子空间
        """
        n = sum(self.bag_size[bag_idx])
        ret_ins_space = np.zeros((n, self.d))
        ret_ins_label = np.zeros(n)
        ret_ins_bag_idx = np.zeros(n, dtype=int)
        ret_ins_idx = np.zeros(n, dtype=int)
        count = 0
        for i in bag_idx:
            bag_size = self.bag_size[i]
            ret_ins_space[count: count + bag_size] = self.get_bag(i)
            ret_ins_label[count: count + bag_size] = self.bag_lab[i]
            ret_ins_bag_idx[count: count + bag_size] = i
            ret_ins_idx[count: count + bag_size] = self.ins_idx[i] + np.arange(bag_size)
            count += bag_size

        return ret_ins_space, ret_ins_label, ret_ins_bag_idx, ret_ins_idx

    def get_po_na_idx(self, idx):
        """"""
        assert self.C == 2
        idx_po, idx_na = [], []
        for i in idx:
            if self.bag_lab[i] == 1:
                idx_po.append(i)
            else:
                idx_na.append(i)

        return idx_po, idx_na

    def get_ADJ(self, is_torch=False):
        """
        获取每一个包的邻接矩阵
        :param is_torch:  是否转换为torch.Tensor
        """
        import torch
        from sklearn.metrics import euclidean_distances
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        ADJ = []
        for i in range(self.N):
            bag = self.get_bag(i)
            # 获取距离矩阵
            M_adj = euclidean_distances(bag)
            # 归一化
            M_adj = mm.fit_transform(M_adj)
            # 类型转换
            if is_torch:
                M_adj = torch.from_numpy(M_adj)
            # 获取邻接矩阵
            ADJ.append(M_adj)

        return ADJ

    def get_kMeans_center(self):
        """
        获取每个包的聚类中心
        """
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=1)

        CENTER = []
        for bag in self.bag_space:
            bag = bag[0][:, :-1]
            kmeans.fit(bag)
            CENTER.append(kmeans.cluster_centers_.tolist()[0])

        return np.array(CENTER)

    def get_euclidean_center(self):
        """
        获取每个包的欧几里得中心
        """
        from sklearn.metrics import euclidean_distances

        CENTER = []
        for bag in self.bag_space:
            bag = bag[0][:, :-1]
            idx = euclidean_distances(bag).sum(0).argmin()
            CENTER.append(bag[idx].tolist())
        return np.array(CENTER)

    def get_dis_matrix(self, gamma=1):
        """
        计算每个包的关联矩阵
        """

        MATRIX = []
        # print("计算关联矩阵中...")
        for i, bag in enumerate(self.bag_space):
            # print_progress_bar(i, self.N)
            bag = bag[0][:, :-1]
            matrix = np.zeros((self.bag_size[i], self.bag_size[i]))
            # 计算阈值
            for j in range(self.bag_size[i]):
                for k in range(self.bag_size[i]):
                    matrix[j, k] = kernel_rbf(bag[j], bag[k], gamma=gamma)
            delta = np.sum(matrix) / (self.bag_size[i] ** 2)

            # 根据阈值进行调整
            for j in range(self.bag_size[i]):
                for k in range(self.bag_size[i]):
                    matrix[j, k] = 1 if matrix[j, k] < delta or j == k else 0
            MATRIX.append(matrix)

        # print()
        return MATRIX

    def get_mean_cov(self):
        """"""
        print("计算协方差矩阵和均值向量...")
        MEAN, COV = [], []
        for i in range(self.N):
            # print_progress_bar(i, self.N)
            bag = self.get_bag(i)
            MEAN.append(np.average(bag, 0))
            COV.append(np.cov(bag.T))

        # print()
        return MEAN, COV

    def get_bag(self, i):
        """
        根据包的索引获取包
        :param i:   包索引
        """
        return self.bag_space[i][0][:, :-1]

    def get_min_max(self):
        """"""
        # print("计算每个包内的基于最大值最小的向量")
        MIN_MAX_VECTOR = []
        for i in range(self.N):
            # print_progress_bar(i, self.N)
            bag = self.get_bag(i)
            min_max_vector = np.hstack((bag.min(0), bag.max(0)))
            MIN_MAX_VECTOR.append(min_max_vector)
        # print()

        return MIN_MAX_VECTOR


if __name__ == '__main__':
    temp_file_name = "D:/OneDrive/Files/Code/Data/MIL/Drug/musk2.mat"
    mil = MIL(temp_file_name)
    # import time
    # s = time.time()
    # mil.get_min_max()
    # print((time.time() - s) * 1000)
    ins = mil.get_sub_ins_space(mil.bag_idx)[0]
    print(ins.max(), ins.min())
