import torch.optim as optim
import numpy as np
from torch import nn
from args.args_dkmil import *
from model.dkmil.knowledge import DataDrivenBagSpace, DataDriven
from model.dkmil.dkmill import BagLoader, DKMIL
from model.train import train
from utils.func_basic import get_k_cv_idx
from args.args_basic import *
from args.args_dkmil import lr, weight_decay


def k_cv(net_type="dk"):
    """"""
    knowledge = DataDriven(data_path)
    data_name = data_path.split('/')[-1].split('.')[0]
    idxes_tr, idxes_te = get_k_cv_idx(knowledge.N, k=n_cv, seed=seed)
    acc_list = []

    loss_func = nn.BCELoss()
    for i, (idx_tr, idx_te) in enumerate(zip(idxes_tr, idxes_te)):
        print("Training {}-CV".format(i))
        save_path = "D:/Data/Pretrain/" + net_type + "_" + data_name + "_%d.pth" % i
        loader_tr = BagLoader(knowledge.bag_space, knowledge.bag_lab, idx_tr)
        loader_te = BagLoader(knowledge.bag_space, knowledge.bag_lab, idx_te)

        knowledge.fit(idx_tr)
        model = None
        if net_type == "dk":
            model = DKMIL(knowledge.d, knowledge=knowledge, knowledge_use=True).to(device)
        else:
            pass
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

        # if os.path.exists(save_path):
        #     model.load_state_dict(torch.load(save_path))

        [_, model, tr_acc, te_acc, auc] = train(loader_tr, loader_te, model, loss_func, optimizer, device,
                                                trained=False)
        # if not os.path.exists(save_path):
        #     torch.save(model, save_path)

        acc_list.append(te_acc)

    acc_cv = np.average(acc_list)
    print(acc_cv)
    return acc_cv


if __name__ == '__main__':
    data_path = "D:/OneDrive/Files/Code/Data/MIL/Drug/musk1.mat"
    print(data_path)
    ACC = []
    for loop in range(5):
        print(loop)
        ACC.append(k_cv(net_type="dk"))
    print("Acc {:.4f}, std {:.4f}".format(np.average(ACC), np.std(ACC, ddof=1)))
