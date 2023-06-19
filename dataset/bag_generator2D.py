import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, data_type="mnist",
                 target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        """
        :param data_type:               数据集的类型
        :param target_number:           目标类别
        :param mean_bag_length:         平均包的大小
        :param var_bag_length:          包大小的变化值
        :param num_bag:                 包的数量
        :param seed:                    随机种子
        :param train:                   是否训练
        """
        self.data_type = data_type
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        # 设置随机种子
        self.r = np.random.RandomState(seed)

        # 训练包的数量
        self.num_in_train = {"mnist": 60000, "cifar10": 50000, "stl10": 5000}[self.data_type]
        # 测试包的数量
        self.num_in_test = {"mnist": 10000, "cifar10": 10000, "stl10": 1000}[self.data_type]

        self.count_list = []

        if self.train:
            # 获取训练集
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            # 获取测试集
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        # MNIST, CIFAR10
        loader = None
        if self.data_type == "mnist":
            if self.train:
                loader = data_utils.DataLoader(datasets.MNIST('D:/Data/Image',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.MNIST('D:/Data/Image',
                                                              train=False,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)
        elif self.data_type == "cifar10":
            if self.train:
                loader = data_utils.DataLoader(datasets.CIFAR10('D:/Data//Image',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.CIFAR10('D:/Data//Image',
                                                                train=False,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)
        elif self.data_type == "stl10":
            if self.train:
                loader = data_utils.DataLoader(datasets.STL10('D:/Data//Image',
                                                              split="train",
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.STL10('D:/Data//Image',
                                                              split="test",
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)

        # 存储下所有的图像及相应标签，这个时候还不是包的状态
        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        # 初始化包及其标签列表
        bags_list = []
        labels_list = []

        # 生成多个包
        for i in range(self.num_bag):
            # 获取包的长度
            bag_length = np.int_(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # 避免包过小
            if bag_length < 1:
                bag_length = 1

            # 获取包中每个图像的索引
            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            # 获取包标签
            labels_in_bag = all_labels[indices]
            # 判断生成包中是否包含目标类
            labels_in_bag = labels_in_bag == self.target_number

            # 添加当前选中图像组成的包
            bags_list.append(all_imgs[indices])
            # 添加相应标签
            labels_list.append(labels_in_bag)
            self.count_list.append(min(1, int(labels_in_bag.float().sum().data.cpu().numpy())))

        return bags_list, labels_list

    def __len__(self):
        """
        获取包的数量
        """
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        """
        :param index:       指定的包的索引
        """
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        return bag, label


def get_loader(data_type):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--data_type', type=str, default="mnist", help='the type of databases')
    parser.add_argument('--target_number', type=int, default=9, metavar='T')
    parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML', help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL', help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                        help='number of bags in training set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    args.data_type = data_type
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_loader = data_utils.DataLoader(MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

    te_loader = data_utils.DataLoader(MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        train=False),
        batch_size=1,
        shuffle=False,
        **loader_kwargs)

    return tr_loader, te_loader
