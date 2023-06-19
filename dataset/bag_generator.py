import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.func_basic import print_progress_bar


class BagLoader:

    def __init__(self, po_label=0, bag_size=(10, 50), po_range=(2, 8), bag_num=(100, 100), seed=None,
                 data_type="mnist", data_path=None):
        """
        :param po_label: The label of positive bag, its range you only can enumerate from $[0..9]$
        :param bag_size: The size of bags --> (min, max)
        :param po_range: The number of positive instance in positive bag --> (min, max)
        :param bag_num: The number of positive bags and negative bag --> (num_positive, num_negative)
        :param seed: The seed for sampling.
            Note: For the fairness of experiments, you should formulate this parameter.
        """
        self.po_label = po_label
        self.bag_size = bag_size
        self.po_range = po_range
        self.bag_num = bag_num
        self.seed = seed
        self.data_type = data_type
        self.data_path = data_path
        self.__init_loader()

    def __init_loader(self):
        print("Initializing data...")
        self.data_space = []
        self.label_space = []
        self.bag_space = []
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.data_type == "mnist":
            self.data_space, self.label_space = self.__load_mnist(True)
            data_space, label_space = self.__load_mnist(False)
            self.data_space.extend(data_space)
            self.label_space.extend(label_space)
        elif self.data_type == "fashion_mnist":
            self.data_space, self.label_space = self.__load_fashion_mnist(True)
            data_space, label_space = self.__load_fashion_mnist(False)
            self.data_space.extend(data_space)
            self.label_space.extend(label_space)
        elif self.data_type == "csv" and self.data_path is not None:
            self.data_space, self.label_space = self.__load_csv()

        self.data_space, self.label_space = np.array(self.data_space), np.array(self.label_space)
        self.po_idx = np.where(self.label_space == self.po_label)[0]
        self.ot_idx = np.where(self.label_space != self.po_label)[0]
        if self.data_type == "mnist" or self.data_type == "csv":
            self.__generate_po_bag(self.bag_num[0])
            self.__generate_ot_bag(self.bag_num[1])
        else:
            self.__generate_po_image(self.bag_num[0])
            self.__generate_ot_image(self.bag_num[1])
        self.bag_space = np.array(self.bag_space)

    def __load_mnist(self, train):
        flag = "train" if train else "test"
        print("Loading MNIST %s data..." % flag)

        data_loader = bag_loader(train, self.data_path)
        num_data = len(data_loader)

        ret_data, ret_label = [], []
        for i, (data, label) in enumerate(data_loader):
            print_progress_bar(i, num_data)
            data, label = data.reshape(-1).numpy().tolist(), int(label.numpy()[0])
            ret_data.append(data)
            ret_label.append(label)
        print()
        return ret_data, ret_label

    def __load_fashion_mnist(self, train):
        flag = "train" if train else "test"
        print("Loading FashionMNIST %s data..." % flag)

        data_loader = bag_loader(train, self.data_path, data_type="fashion_mnist")
        num_data = len(data_loader)

        ret_data, ret_label = [], []
        for i, (data, label) in enumerate(data_loader):
            print_progress_bar(i, num_data)
            data = data[0][0]
            data, label = (data.numpy(), int(label.numpy()[0]))
            ret_data.append(data)
            ret_label.append(label)
        print()
        return ret_data, ret_label

    def __load_csv(self):
        print("Loading CSV data...")
        data = pd.read_csv(self.data_path, encoding="gbk").values
        return data[:, :-1], data[:, -1].reshape(-1)

    def __generate_po_bag(self, bag_num):
        print("Generating positive bag...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            bag = []
            bag_size = np.random.randint(self.po_range[0], self.po_range[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.po_idx)].tolist() + [1]
                bag.append(ins)
            bag_size = np.random.randint(self.bag_size[0] - self.po_range[0], self.bag_size[1] - self.po_range[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.ot_idx)].tolist() + [0]
                bag.append(ins)
            bag = np.array(bag)
            bag = np.array([bag, np.array([[1]])])
            self.bag_space.append(bag)
        print()

    def __generate_ot_bag(self, bag_num):
        print("Generate other class bag...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            bag = []
            bag_size = np.random.randint(self.bag_size[0], self.bag_size[1] + 1)
            for j in range(bag_size):
                ins = self.data_space[np.random.choice(self.ot_idx)].tolist() + [0]
                bag.append(ins)
            bag = np.array(bag)
            bag = np.array([bag, np.array([[0]])])
            self.bag_space.append(bag)
        print()

    def __generate_po_image(self, bag_num):
        print("Generating positive image...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            idx = np.random.choice(self.po_idx)
            bag = self.data_space[idx]
            bag = np.hstack([bag, np.ones((len(bag), 1))])
            self.bag_space.append([bag, np.array([[1]])])
        print()

    def __generate_ot_image(self, bag_num):
        print("Generate other class image...")
        for i in range(bag_num):
            print_progress_bar(i, bag_num)
            idx = np.random.choice(self.ot_idx)
            bag = self.data_space[idx]
            bag = np.hstack([bag, np.zeros((len(bag), 1))])
            self.bag_space.append([bag, np.array([[0]])])
        print()


def bag_loader(train, path=None, data_type="mnist"):
    """"""
    if path is None:
        path = "../../Data"
    if data_type == "mnist":
        loader = datasets.MNIST
    elif data_type == "fashion_mnist":
        loader = datasets.FashionMNIST
    else:
        loader = datasets.CIFAR10
    return DataLoader(loader(path, train=train, download=True,
                             transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=False)


def simple_bag_loader(interval=0.05):
    """

    :param interval:    每个样本之间的基准间隔
    :return:            用于集合核评估的测试数据集
    """
    np.set_printoptions(suppress=True)
    BASIC = np.array([[0, 0.1], [0, -0.1], [0.1, 0], [-0.1, 0]])

    def generator(a, b):
        ret_ = BASIC.copy()
        ret_[:, 0] += a
        ret_[:, 1] += b
        return ret_

    BAGS = []
    count = 0
    for i in np.arange(-1, 1.1, interval):
        for j in np.arange(-1, 1.1, interval):
            # label = np.array([[0 if i <= 0 else 1]])
            label = np.array([[0 if count % 2 == 0 else 1]])
            if np.abs(i) < 1e-6 and np.abs(j) < 1e-6:
                print(count)
            count += 1
            bag = np.hstack([generator(i, j), np.zeros((len(BASIC), 1)) + label])
            BAGS.append(np.array([bag, label]))
    return np.array(BAGS)


if __name__ == '__main__':
    print(simple_bag_loader()[:, 0].shape)
