import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from args.args_basic import device, TORCH_SEED, os
from utils.func_basic import write2file


def get_model2D(args, round_idx=0, knowledge=None, model_save_home=r"D:\Data\Pretrain\Model\OOD"):
    os.makedirs(model_save_home, exist_ok=True)
    save_path = os.path.join(model_save_home,
                             f"{args.data_type}{args.target_number}_{args.net_type}_seed{TORCH_SEED}_round{round_idx}"
                             f".pth")
    if args.net_type == "camil":
        save_path = os.path.join(r"D:\Data\Pretrain\Model\CaMIL",
                                 f"{args.data_type}{args.target_number}_{args.net_type}_"
                                 f"abmil_seed{TORCH_SEED}_round{round_idx}.pth")
    trainer, d, num_channel = None, -1, 3
    extractor = "cnn"
    if args.data_type == "mnist":
        d = 50 * 4 * 4 if args.net_type != "mamil" else 48 * 5 * 5
        num_channel = 1
    elif args.data_type == "cifar10":
        d = 50 * 5 * 5 if args.net_type != "mamil" else 48 * 6 * 6
        # extractor = "resnet"
    elif args.data_type == "stl10":
        d = 50 * 21 * 21 if args.net_type != "mamil" else 48 * 22 * 22
    elif args.data_type == "tumor":
        d = 50 * 125 * 125 if args.net_type != "mamil" else 50 * 126 * 126
    elif args.data_type == "imagenet":
        d = 50 * 72 * 72 if args.net_type != "ma" else 48 * 73 * 73

    trainer = Trainer(args, d=d, num_channel=num_channel, extractor=extractor, knowledge=knowledge)

    return trainer, save_path


def get_model(args, knowledge, loop, fold_idx,
              model_save_home=r"D:\Data\Pretrain\Model\Benchmark"):
    os.makedirs(model_save_home, exist_ok=True)
    save_path = os.path.join(model_save_home,
                             f"{args.data_type}_{args.net_type}_seed{TORCH_SEED}_loop{loop}_fold{fold_idx}.pth")

    trainer = Trainer(args, d=knowledge.d, extractor="none", knowledge=knowledge)

    return trainer, save_path


class Trainer:
    def __init__(self, args, d=50*4*4, num_channel=1, knowledge=None, extractor="cnn"):

        if extractor != 'none':
            from model.dkmil2D import DKMIL
            self.net = DKMIL(args, d, knowledge, in_channels=num_channel, extractor=extractor)
            self.target_number = args.target_number
        else:
            from model.dkmil import DKMIL
            self.net = DKMIL(args, d, knowledge)
        self.args = args
        self.extractor = extractor
        self.net.to(device)
        self.best_net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, 0.000005)

        self.epochs = args.epochs
        self.lambda_l1 = args.lambda_l1

        self.metrics = {"acc": accuracy_score, "f1": f1_score, "auc": roc_auc_score}
        self.record_save_path = os.path.join(r".\record" if extractor != "none" else r".\record\benchmark",
                                             f"{args.data_type}_{args.net_type}_seed{TORCH_SEED}")
        os.makedirs(self.record_save_path, exist_ok=True)

    def training(self, tr_loader, te_loader, model_save_path, round_idx, loop=0):
        if self.extractor != "none":
            file_path = self.record_save_path + \
                        "/target_number{:d}_round{:d}.txt".format(self.args.target_number, round_idx)
        else:
            file_path = self.record_save_path + \
                        "/loop{:d}.txt".format(loop, round_idx)
        best_acc, best_f1, best_auc = -1, -1, -1
        if os.path.exists(model_save_path):
            write2file(file_path, context="Loading pretrained model...\n")
            self.net.load_state_dict(torch.load(model_save_path))
            self.best_net = self.net
            best_acc, best_f1, best_auc = self.testing(te_loader)
            write2file(file_path, context="Best acc {:.4f}, f1 {:.4f}, auc {:.4f}\n"
                       .format(best_acc, best_f1, best_auc))
        else:
            write2file(file_path, context="Training...\n")
            for epoch in range(self.epochs):
                total_loss = 0.0
                self.net.train()
                for batch_idx, (bag, y) in enumerate(tr_loader):
                    # Get each bag with label
                    if isinstance(y, list):
                        y = y[0].float()
                    # y = y[0].type(torch.LongTensor)
                    bag, y = bag.to(device), y.to(device)

                    self.optimizer.zero_grad()

                    # Prediction and computing loss
                    y_score = self.net(bag)[0]

                    # loss = self.criterion(y_score, y.reshape(1, 1))
                    loss = -1. * (y * torch.log(y_score) + (1. - y) * torch.log(1. - y_score))
                    l1 = torch.tensor([0]).float().to(device)
                    for param in self.net.parameters():
                        l1 += torch.norm(param, 1)
                    loss = loss + self.lambda_l1
                    loss.sum().backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    total_loss += loss.cpu().float().mean().item()

                self.net.eval()
                acc1, f11, auc1 = self.testing(tr_loader)
                acc2, f12, auc2 = self.testing(te_loader)

                if epoch > 5 and best_acc < acc2:
                    best_acc = acc2
                    best_f1 = f12
                    best_auc = auc2
                    self.best_net = self.net
                    torch.save(self.best_net.state_dict(), model_save_path)

                write2file(file_path, context="Epoch {:5d}/{:5d}, training loss {:.4f}; "
                                              "training acc {:.4f}, f1 {:.4f}, auc {:.4f}; "
                                              "testing acc {:.4f}, f1 {:.4f}, auc {:.4f}; "
                                              "best acc {:.4f}, f1 {:.4f}, auc {:.4f}\n"
                           .format(epoch, self.epochs, total_loss,
                                   acc1, f11, auc1, acc2, f12, auc2, best_acc, best_f1, best_auc))

                if best_acc == 1.0:
                    break
        return self.best_net, best_acc, best_f1, best_auc, self.record_save_path

    def testing(self, data_loader):
        self.net.eval()
        total_loss = 0.
        y_list, y_hat_list = [], []
        for batch_idx, (bag, y) in enumerate(data_loader):
            if isinstance(y, list):
                y = y[0].float()
            bag, y = bag.to(device), y.to(device)
            y_score, y_hat, _ = self.net(bag)
            y_list.append(y.reshape(1).detach().cpu().numpy().tolist()[0])
            y_hat_list.append(y_hat.reshape(1).detach().cpu().numpy().tolist()[0])

            loss = self.criterion(y_score, y.reshape(1, 1))
            total_loss += loss.cpu().float().mean().item()

        acc = self.metrics["acc"](y_list, y_hat_list)
        f1 = self.metrics["f1"](y_list, y_hat_list)
        auc = self.metrics["auc"](y_list, y_hat_list)

        return acc, f1, auc
