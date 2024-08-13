import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import euclidean_distances
from clustering.density_peak import DensityPeaks
from clustering.kmeans import KMeans
from utils.MIL import MIL
from utils.func_basic import kernel_rbf


class DataDriven(MIL):
    """"""

    def __init__(self, args, data_path, save_home="D:/Data/Distance/", bag_space=None):
        """"""
        super(DataDriven, self).__init__(data_path, save_home=save_home, bag_space=bag_space)
        self.args = args
        self.__initialize()

    def __initialize(self):
        self.n_bag_center = self.args.n_bag_center
        self.n_bag_density = self.args.n_bag_density

        self.n_ins_center = self.args.n_ins_center
        self.n_ins_density = self.args.n_ins_density

        self.n_sample = self.args.n_sample
        self.n_ins_mask = self.args.n_ins_mask

        self.ins_cluster = MiniBatchKMeans(n_clusters=self.n_ins_center)
        self.ins_density = None
        self.bag_cluster = None
        self.bag_density = None

        self.idx = []

    def __reset(self):
        self.knowledge_ins = np.zeros((1, self.d))
        self.knowledge_bag = np.array([], dtype=int)
        self.knowledge_idx = []

    def fit(self, idx=None):
        if idx is None:
            self.idx = np.arange(self.N)
        else:
            self.idx = idx
        self.sample_size = np.min([50, max(10, int(len(self.idx) * 0.1)), len(self.idx)])

        sample_idx = self.__sample_idx()
        self.__reset()
        for idx in sample_idx:
            # Instance index.
            sub_ins_space = self.get_sub_ins_space(idx)[0]
            ins_idx = np.hstack([self.__ins_center(sub_ins_space),
                                 self.__ins_density(sub_ins_space)])
            ins_idx = list(set(ins_idx))
            self.knowledge_ins = np.vstack([self.knowledge_ins,
                                            sub_ins_space[ins_idx]])
            # Bag index.
            dis = self.__b2b_msk(sub_ins_space, idx)
            self.knowledge_bag = np.hstack([self.knowledge_bag,
                                            self.__bag_center(idx, dis),
                                            self.__bag_density(idx, dis)
                                            ])
        bag_idx = np.array(list(set(self.knowledge_bag)), dtype=int)
        sub_ins_space = self.get_sub_ins_space(bag_idx)[0]

        self.knowledge_ins = self.knowledge_ins[1:]
        self.knowledge_bag = self.__b2b_msk(sub_ins_space, np.arange(self.N), bag_idx)
        self.knowledge_idx = bag_idx

    def __sample_idx(self):
        sample_idx = []
        for i in range(self.n_sample):
            sample_idx.append(np.random.choice(self.idx, size=self.sample_size).tolist())
        return np.array(sample_idx)

    def __ins_center(self, ins_space):
        self.ins_cluster.fit(ins_space)
        centers = self.ins_cluster.cluster_centers_
        idx = self.__ins_neighbor(ins_space, centers)
        return idx

    def __ins_density(self, ins_space):
        dis = euclidean_distances(ins_space)
        self.ins_density = DensityPeaks(dis=dis)
        self.ins_density.fit(n=self.n_ins_density)
        idx = self.ins_density.centers
        return idx

    @staticmethod
    def __ins_neighbor(x, centers):
        dis = euclidean_distances(x, centers)
        return np.argsort(dis, axis=0)[0]

    def __bag_center(self, idx, dis):
        self.bag_cluster = KMeans(dis, k=self.n_bag_center)
        self.bag_cluster.fit()
        centers = self.bag_cluster.centers
        return idx[centers]

    def __bag_density(self, idx, dis):
        self.bag_density = DensityPeaks(dis=dis)
        self.bag_density.fit(n=self.n_bag_density)
        idx = self.bag_density.centers
        return idx

    def __b2b_msk(self, ins_space, idx1, idx2=None):
        """"""
        k_means = MiniBatchKMeans(n_clusters=self.n_ins_mask)
        k_means.fit(ins_space)
        ins1 = k_means.cluster_centers_
        k_means.fit(ins_space)
        ins2 = k_means.cluster_centers_
        embedding1, embedding2 = [], []
        for i in idx1:
            bag = self.get_bag(i)
            embedding1.append([euclidean_distances(bag, ins1).mean(0).reshape(1, self.n_ins_mask),
                               euclidean_distances(bag, ins2).mean(0).reshape(1, self.n_ins_mask)])
        if idx2 is None:
            idx2 = idx1
            embedding2 = embedding1
        else:
            for j in idx2:
                bag = self.get_bag(j)
                embedding2.append([euclidean_distances(bag, ins1).mean(0).reshape(1, self.n_ins_mask),
                                   euclidean_distances(bag, ins2).mean(0).reshape(1, self.n_ins_mask)])

        n1_bag, n2_bag = len(idx1), len(idx2)
        dis = np.zeros((n1_bag, n2_bag))
        for i in range(n1_bag):
            for j in range(n2_bag):
                dis[i][j] = (0.5 * kernel_rbf(embedding1[i][0], embedding2[j][0], gamma=1) +
                             0.5 * kernel_rbf(embedding1[i][1], embedding2[j][1], gamma=1))

        return dis
