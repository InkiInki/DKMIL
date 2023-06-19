import argparse
import numpy as np
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch import nn
from dataset.bag_generator2D import MnistBags
from model.dkmil.dkmil2D import DKMIL
from model.dkmil.knowledge2D import DataDriven
from model.train import train
from args.args_dkmil import lr, weight_decay
from utils.func_basic import get_k_cv_idx

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    """"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--data_type', type=str, default="stl10", help='the type of databases')
    parser.add_argument('--target_number', type=int, default=0, metavar='T')
    parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML', help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL', help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                        help='number of bags in training set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_data = MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True)
    te_data = MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        train=False)

    tr_loader = data_utils.DataLoader(
        tr_data,
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

    te_loader = data_utils.DataLoader(
        te_data,
        batch_size=1,
        shuffle=False,
        **loader_kwargs)

    recorder_list = []
    loss_func = nn.BCELoss()
    acc_list = []
    knowledge = DataDriven(tr_data)
    for i in range(5):
        knowledge.fit()
        model = DKMIL(96, knowledge=knowledge, d_reshape=50 * 21 * 21, in_channels=3)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        [_, model, tr_acc, te_acc, auc] = train(tr_loader, te_loader, model, loss_func, optimizer, device,
                                                trained=False)
        # if not os.path.exists(save_path):
        #     torch.save(model, save_path)

        acc_list.append(te_acc)

    acc_cv = np.average(acc_list)
    print(acc_cv)
    return acc_cv


if __name__ == "__main__":
    main()
