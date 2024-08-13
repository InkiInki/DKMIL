import numpy as np
import yaml
from args.args_basic import *
from dataset.bag_generator2D import get_loader
from model.knowledge2D import DataDriven
from model.train import get_model2D
from utils.func_basic import write2file


def get_param(data_type="mnist", target_number=0, params_config=None):
    import argparse
    if params_config is not None:
        with open(params_config, 'r') as f:
            params_config = yaml.load(f, Loader=yaml.FullLoader)
        return argparse.Namespace(**params_config)

    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--data_type', type=str, default=data_type, help='the type of databases')
    parser.add_argument('--target_number', type=int, default=target_number, metavar='T')
    parser.add_argument('--net_type', type=str, default="dkmil")
    parser.add_argument('--lr', default=0.0005, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lambda_l1', default=1e-5, type=float, help='The weight for loss function')
    parser.add_argument('--weight_decay', default=5*1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--drop_out', default=0.1, type=int)
    parser.add_argument('--epochs', default=50, type=int, help='Number of total training epochs')

    """For model structure"""
    parser.add_argument('--n_bag_center', default=3, type=int)
    parser.add_argument('--n_bag_density', default=3, type=int)
    parser.add_argument('--n_ins_center', default=3, type=int)
    parser.add_argument('--n_ins_density', default=3, type=int)
    parser.add_argument('--n_sample', default=10, type=int)
    parser.add_argument('--n_ins_mask', default=3, type=int)
    parser.add_argument('--n_mask', default=10, type=int)
    parser.add_argument('--r_mask', default=0.1, type=int)
    parser.add_argument('--H_k', default=256, type=int)
    parser.add_argument('--H_a', default=128, type=int)
    parser.add_argument('--D_a', default=64, type=int)
    parser.add_argument('--H_c', default=16, type=int)
    parser.add_argument('--H_s', default=64, type=int)
    parser.add_argument('--D_s', default=128, type=int)

    """For data set"""
    parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML', help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL', help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain', help='number of bags in training set')
    parser.add_argument('--num_bags_vad', type=int, default=50, metavar='NVad', help='number of bags in validation set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=TORCH_SEED, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--params_config', type=str, default=params_config, help='Parameters config file (to reproduce results)')

    args = parser.parse_args()
    if args.params_config is None:
        print("Saving the parameters for algorithm")
        with open(os.path.join(r".\args", f"{args.data_type}_{args.net_type}_seed{TORCH_SEED}.yml"), "w") as f:
            yaml.dump(args.__dict__, f)

    return args


def main():
    """"""
    args = get_param(data_type="mnist", target_number=9)
    tr_data, _, _, tr_loader, _, te_loader = get_loader(args=args, data_is_need=True)

    acc_list, f1_list, auc_list = [], [], []
    knowledge = DataDriven(args, tr_data)
    file_path = ""
    for i in np.arange(0, 5):
        knowledge.fit()
        trainer, model_save_path = get_model2D(args, knowledge=knowledge, round_idx=i)
        print("Saving to", model_save_path)
        _, acc, f1, auc, file_path = trainer.training(tr_loader, te_loader,
                                                      model_save_path=model_save_path, round_idx=i)
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
    write2file(file_path + "/target_number{:d}_results.txt".format(args.target_number),
               context="Average acc {:.4f}, std acc {:.4f}; "
                       "Average f1 {:.4f}, std f1 {:.4f}; "
                       "Average auc {:.4f}, std auc {:.4f}\n"
               .format(np.average(acc_list), np.std(acc_list, ddof=1),
                       np.average(f1_list), np.std(f1_list, ddof=1),
                       np.average(auc_list), np.std(auc_list, ddof=1),))


if __name__ == "__main__":
    data_type = "mnist"  # cifar10, stl10
    main()
