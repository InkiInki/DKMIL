import torch
import numpy as np
from clustering.density_peak import DensityPeaks
from clustering.kmeans import KMeans
from utils.image2image import MS_SSIM
from args.args_basic import TORCH_SEED
np.random.seed(TORCH_SEED)


class DataDriven:
    """"""

    def __init__(self, args, data_set, in_channels=1):
        """"""
        super(DataDriven, self).__init__()
        self.args = args
        self.data_set = data_set
        self.in_channels = in_channels
        self.knowledge_bag = None
        self.ms_ssim = MS_SSIM(data_range=1)

        self.__initialize()

    def fit(self):
        self.reset()
        for idx in self.sample_idx:
            bags = []
            for i in idx:
                bag = self.data_set.__getitem__(i)[0]
                for image in bag:
                    bags.append(image.tolist())
            bags = np.array(bags)
            bags_temp = np.resize(bags, [bags.shape[0], 161, 161])
            bags_temp = bags_temp.reshape([bags_temp.shape[0], 1, self.in_channels,
                                           bags_temp.shape[-2], bags_temp.shape[-1]])
            bags_temp = torch.from_numpy(bags_temp).float()
            if self.in_channels == 1:
                bags_temp = torch.tile(bags_temp, (1, 1, 3, 1, 1))
            dis = self.__b2b_image(bags_temp)
            idx = np.arange(len(bags))
            image_idx = np.hstack([self.__bag_center(idx, dis),
                                   self.__bag_density(dis)])
            image_idx = list(set(image_idx))
            if self.knowledge_bag is None:
                self.knowledge_bag = bags[image_idx]
            else:
                self.knowledge_bag = np.vstack([self.knowledge_bag, bags[image_idx]])

        self.knowledge_bag = torch.from_numpy(self.knowledge_bag).float()

    def reset(self):
        """"""
        self.knowledge_bag = None

    def __initialize(self):
        self.size = len(self.data_set)
        self.idx = np.arange(self.size)
        self.n_sample = self.args.n_sample
        self.sample_size = 3
        self.n_ins_mask = self.args.n_ins_mask
        self.n_bag_center = self.args.n_bag_center
        self.n_bag_density = self.args.n_bag_density

        self.sample_idx = self.__sample_idx()

    def __sample_idx(self):
        sample_idx = []
        for i in range(self.n_sample):
            sample_idx.append(np.random.choice(self.idx, size=self.sample_size).tolist())
        return np.array(sample_idx)

    def __b2b_image(self, bags):
        n_bag = len(bags)
        dis = np.zeros((n_bag, n_bag))
        for i in range(n_bag):
            for j in range(i, n_bag):
                dis[i][j] = dis[j][i] = self.ms_ssim(bags[i], bags[j])

        return dis

    def __bag_center(self, idx, dis):
        self.bag_cluster = KMeans(dis, k=self.n_bag_center)
        self.bag_cluster.fit()
        centers = self.bag_cluster.centers
        return idx[centers]

    def __bag_density(self, dis):
        self.bag_density = DensityPeaks(dis=dis)
        self.bag_density.fit(n=self.n_bag_density)
        idx = self.bag_density.centers
        return idx


class DataDrivenBag:

    def __init__(self, bag, in_channels=1, ratio=0.5):
        super(DataDrivenBag, self).__init__()
        self.bag = bag
        self.in_channels = in_channels
        self.ratio = ratio
        self.ms_ssim = MS_SSIM(data_range=1)

        self.__initialize()

    def fit(self):

        bag = self.bag
        bag = np.resize(bag, [bag.shape[0], 161, 161])
        bag = bag.reshape([bag.shape[0], 1, self.in_channels, bag.shape[-2], bag.shape[-1]])
        bag = torch.from_numpy(bag).float()
        if self.in_channels == 1:
            bag = torch.tile(bag, (1, 1, 3, 1, 1))

        dis = self.__b2b_image(bag)
        idx = np.hstack([self.__bag_center(np.arange(self.size), dis),
                         self.__bag_density(dis)])
        self.idx = list(set(idx))

    def __initialize(self):
        self.size = len(self.bag)
        self.idx = []
        self.n_ins = max(1, int(self.size * self.ratio))

    def __b2b_image(self, bags):
        n_bag = len(bags)
        dis = np.zeros((n_bag, n_bag))
        for i in range(n_bag):
            for j in range(i, n_bag):
                dis[i][j] = dis[j][i] = self.ms_ssim(bags[i], bags[j])

        return dis

    def __bag_center(self, idx, dis):
        self.bag_cluster = KMeans(dis, k=self.n_ins)
        self.bag_cluster.fit()
        centers = self.bag_cluster.centers
        return idx[centers]

    def __bag_density(self, dis):
        self.bag_density = DensityPeaks(dis=dis)
        self.bag_density.fit(n=self.n_ins)
        idx = self.bag_density.centers
        return idx
