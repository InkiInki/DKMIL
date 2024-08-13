import argparse
import numpy as np
from args.args_basic import *
from model.knowledge import DataDriven
from model.dkmil import BagLoader
from model.train import get_model
from utils.func_basic import get_k_cv_idx, write2file


def get_param():

    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--data_type', type=str, default=data_path.split('/')[-1].split('.')[0])
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

    parser.add_argument('--seed', type=int, default=TORCH_SEED, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()

    return args


def k_cv(n_cv=5):
    """"""
    args = get_param()
    knowledge = DataDriven(args, data_path)
    data_name = data_path.split('/')[-1].split('.')[0]
    idxes_tr, idxes_te = get_k_cv_idx(knowledge.N, k=n_cv)
    acc_list, f1_list, auc_list = [], [], []

    file_path = ""
    for i, (idx_tr, idx_te) in enumerate(zip(idxes_tr, idxes_te)):
        print(f"\tThe {i}-th fold CV")
        print(f"\tLoading data set")
        tr_loader = BagLoader(knowledge.bag_space, knowledge.bag_lab, idx_tr)
        te_loader = BagLoader(knowledge.bag_space, knowledge.bag_lab, idx_te)
        print(f"\tInitialize the knowledge")
        knowledge.fit(idx_tr)
        trainer, model_save_path = get_model(args, knowledge=knowledge, loop=loop, fold_idx=i)
        print("\tSaving to", model_save_path)
        print("\tStart training")
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

    return np.average(acc_list), np.average(f1_list), np.average(auc_list)


if __name__ == '__main__':
    data_path = "D:/OneDrive/Files/Code/Data/MIL/Text/alt_atheism.mat"
    print(f"Modeling on {data_path.split('/')[-1].split('.')[0]}...")
    ACC, F1, AUC = [], [], []
    for loop in range(5):
        print(f"The {loop}-th independent experiment")
        acc_, f1_, auc_ = k_cv()
        ACC.append(acc_)
        F1.append(f1_)
        AUC.append(auc_)

    print("$%.3lf\\pm%.3lf$" % (np.average(ACC), np.std(ACC, ddof=1)))
    print("$%.3lf\\pm%.3lf$" % (np.average(F1), np.std(F1, ddof=1)))
    print("$%.3lf\\pm%.3lf$" % (np.average(AUC), np.std(AUC, ddof=1)))
    write2file(f"record/benchmark/0{data_path.split('/')[-1].split('.')[0]}_results.txt",
               context="Average acc {:.4f}, std acc {:.4f}; "
                       "Average f1 {:.4f}, std f1 {:.4f}; "
                       "Average auc {:.4f}, std auc {:.4f}\n"
               .format(np.average(ACC), np.std(ACC, ddof=1),
                       np.average(F1), np.std(F1, ddof=1),
                       np.average(AUC), np.std(AUC, ddof=1), ))
