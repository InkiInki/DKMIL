import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Clustering:
    """
    The origin class of all clustering algorithms.
    :param
        para_dis:
            The distances matrix.
        para_idx:
            The index of clustering data.
    @attribute
        centers:
            The clustering centers.
        blocks:
            The clustered blocks.
        lab:
            The instances label, and the instance lab will be same as the index of block.
    """

    def __init__(self, dis, idx=None):
        """
        The constructor.
        """
        self.dis = dis
        self.idx = idx
        self.centers = []
        self.blocks = []
        self.lab = []
        self.max_dis = 0
        self.num_sample = 0
        self.__initialize()

    def __initialize(self):
        """
        The initialize of clustering.
        """

        if self.idx is None:
            self.idx = np.arange(len(self.dis))

        self.max_dis = np.max(self.dis[self.idx, :][:, self.idx])
        self.num_sample = len(self.idx)
        self.ave_dis = np.sum(np.triu(self.dis[self.idx, :][:, self.idx])) / (self.num_sample * (
                self.num_sample - 1) / 2) if self.num_sample > 1 else np.sum(self.dis)
