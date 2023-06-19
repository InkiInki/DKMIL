import numpy as np
from clustering.clustering_basic import Clustering


class DensityPeaks(Clustering):

    """
    The density peaks clustering algorithm.
    :param
        dc_r:
            The ratio of cutoff distance.
    @attribute
        centers, lab, blocks:
            Please refer the class named ClusteringOrigin
        sort_max_info_idx:
            The sorted index of max_info.
    """

    def __init__(self, dis, idx=None, dc_r=0.2, para_dc_type='max', para_cutoff_type='gaussian'):
        """
        The constructor.
        """
        super(DensityPeaks, self).__init__(dis, idx)
        self.dc = 0
        self.dc_r = dc_r
        self.dc_type = para_dc_type
        self.cutoff_type = para_cutoff_type
        self.density = []
        self.sort_density_idx = []
        self.max_info = []
        self.sort_max_info_idx = []
        self.master = []
        self.dis2master = []
        self.__initialize_density_peaks()

    def fit(self, n=3):
        """
        clustering to block.
        :param
            n:
                The number of clustering center.
        """
        self.__get_density()
        self.__get_dis2master()
        self.__get_max_info()

        # Get the clustering centers.
        self.lab = np.zeros(self.num_sample, dtype=int)
        self.centers = np.argsort(self.max_info)[-n:][::-1]

        # Get the instances label corresponding the block's index.
        for i in range(n):
            self.lab[self.centers[i]] = -i - 1
        temp_lab = np.copy(self.lab)
        for i in range(self.num_sample):
            if temp_lab[i] < 0:
                continue
            temp_lab[i] = self.__get_lab(self.master[i])

        # Get the clustered block.
        self.lab = -temp_lab
        self.blocks = [[] for _ in range(n)]
        for i in range(self.num_sample):
            self.blocks[self.lab[i] - 1].append(i)
        self.blocks = np.array(self.blocks)

    def __initialize_density_peaks(self):
        """
        The initialize of density peaks.
        """

        self.dc = self.max_dis * self.dc_r

    def __compute_density(self, idx):
        """
        Compute density for the given sample.
        """

        val = self.dis[idx][self.idx]
        return np.sum(np.exp(-(val / self.dc)**2))

    def __get_lab(self, i):
        """
        Get the label of each sample.
        :param
            para_i:
                The given index.
        :return
            The index of given sample.
        """
        if self.lab[i] < 0:
            return self.lab[i]
        else:
            return self.__get_lab(self.master[i])

    def __get_density(self):
        """
        Compute density for all sample.
        """
        self.density = np.zeros(self.num_sample)

        for i in range(self.num_sample):
            self.density[i] = self.__compute_density(self.idx[i])

    def __get_dis2master(self):
        """
        Compute distance to master.
        """
        self.sort_density_idx = np.argsort(self.density)[::-1]
        self.dis2master = np.zeros(self.num_sample)
        self.dis2master[self.sort_density_idx[0]] = np.inf
        self.master = np.zeros(self.num_sample, dtype=int)
        self.master[self.sort_density_idx[0]] = -1
        for i in range(1, self.num_sample):
            idx = self.sort_density_idx[i]
            self.master[idx] = self.sort_density_idx[np.argmin(self.dis[idx][self.sort_density_idx[: i]])]
            self.dis2master[idx] = np.min(self.dis[idx][self.sort_density_idx[: i]])

    def __get_max_info(self):
        """
        Compute
        """
        self.max_info = np.multiply(self.density, self.dis2master)
        self.sort_max_info_idx = np.argsort(self.max_info)[::-1]
