import torch
from args.args_dkmil import lambda_l1, epoch
from sklearn.metrics import roc_auc_score

torch.set_default_tensor_type('torch.FloatTensor')


def get_acc(model, B, y, device):
    score = model(B.to(device))[0]
    y_hat = torch.ge(score, 0.5).float()
    acc = y_hat.eq(y).cpu().float().mean().item()

    return acc, y_hat


def train(loader_tr, loader_te, model, loss_func, optimizer, device, trained=False):
    """"""
    batch_count = 0
    max_recorder = [0, None, 0, 0, -1]

    for i in range(epoch):
        # print_progress_bar(i, epoch)
        tr_loss, tr_acc, te_acc = 0, 0, 0

        model.train()
        for (B, y) in loader_tr:
            y_prob = model(B.to(device))[0]
            if isinstance(y, list):
                y = y[0].float()

            # loss = -1. * (y * torch.log(y_prob) + (1. - y) * torch.log(1. - y_prob - 1e-6))
            loss = loss_func(y_prob, y.reshape(1, 1))
            l1 = torch.tensor([0]).float()
            for param in model.parameters():
                l1 += torch.norm(param, 1)
            loss = loss + lambda_l1 * l1
            # print(loss, y, y_prob)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            tr_loss += loss.cpu().float().mean().item()
            batch_count += 1

        model.eval()
        for (B_ne, y_ne) in loader_tr:
            if isinstance(y_ne, list):
                y_ne = y_ne[0].float()

            acc, _ = get_acc(model, B_ne, y_ne, device)
            tr_acc += acc

        Y, Y_hat = [], []
        for (B_te, y_te) in loader_te:
            if isinstance(y_te, list):
                y_te = y_te[0].float()

            acc, y_hat = get_acc(model, B_te, y_te, device)
            y_te = y_te.reshape(1).detach().cpu().numpy().tolist()[0]
            y_hat = y_hat.reshape(1).detach().cpu().numpy().tolist()[0]
            Y.append(y_te)
            Y_hat.append(y_hat)
            te_acc += acc

        tr_loss = tr_loss / len(loader_tr)
        tr_acc = tr_acc / len(loader_tr)
        te_acc = te_acc / len(loader_te)
        auc = roc_auc_score(Y, Y_hat)
        print("Epoch: %d, train loss: %.4f, train acc: %.4f, test acc: %.4f, auc: %.4f" %
              (i, tr_loss, tr_acc, te_acc, auc))
        if i >= 5 and te_acc > max_recorder[-2]:
            max_recorder = [i, model, tr_acc, te_acc, auc]

    return max_recorder
