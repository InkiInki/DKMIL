"""
Author: Inki
Email: inki.yinji@gmail.com
Created: 2022 08 18
Modified: 2022 10 07
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.utils.data as data_utils
from sklearn.metrics import euclidean_distances
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DKMIL(nn.Module):

    def __init__(self, args, d, knowledge, knowledge_use=True):
        super(DKMIL, self).__init__()
        self.args = args
        self.d = d
        self.knowledge_use = knowledge_use

        self.knowledge_ins = torch.from_numpy(knowledge.knowledge_ins).float().unsqueeze(0)
        self.knowledge_bag = self.__get_knowledge_bag(knowledge)
        self.__init_mika()

    def forward(self, B):
        """"""
        if B.shape[1] == 1:
            B = torch.tile(B, (1, 2, 1))
        if self.knowledge_use is True:
            B_fuse = self.fuse(B)
        else:
            B_fuse = B
        B_fuse_att = self.att(B_fuse)[0]
        b, A = self.self_att(B_fuse, B_fuse_att)
        Y_prob = self.classifier(b)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def __init_mika(self):
        self.n_ins = max(self.args.n_mask, int(self.args.r_mask * self.knowledge_ins.shape[0]))
        self.n_bag = len(self.knowledge_bag[1]) - 1
        self.fuse = KnowledgeFuseBlock(self.args, self.d, self.knowledge_ins, self.knowledge_bag,
                                       self.n_ins, self.n_bag)
        self.att = AttentionBlock(self.args, self.d)
        self.self_att = SelfAttentionBlock(self.args, self.d)
        self.classifier = nn.Sequential(nn.Linear(self.args.H_s, 1), nn.Sigmoid())

        self.apply(self.weight_init)

    def __get_knowledge_bag(self, knowledge):
        knowledge_idx = knowledge.knowledge_idx
        sub_ins_space = knowledge.get_sub_ins_space(knowledge_idx)[0]
        ins_idx = np.zeros(len(knowledge_idx) + 1, dtype=int)
        for i in range(len(knowledge_idx)):
            ins_idx[i + 1] = ins_idx[i] + knowledge.bag_size[knowledge_idx[i]]

        return [torch.from_numpy(sub_ins_space).float().unsqueeze(0), ins_idx]

    @staticmethod
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch_init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)


class KnowledgeFuseBlock(nn.Module):
    
    def __init__(self, args, d, knowledge_ins, knowledge_bag, n_ins, n_bag):
        super(KnowledgeFuseBlock, self).__init__()
        self.ins_feat_fuse = KnowledgeInsFuseBlock(args, d, knowledge_ins, n_ins)
        self.bag_feat_fuse = KnowledgeBagFuseBlock(args, d, knowledge_bag, n_bag)
        self.embedding = nn.Sequential(nn.Linear(d + n_ins + n_bag, d), nn.LeakyReLU())

    def forward(self, B):
        B_ins = self.ins_feat_fuse(B)
        B_bag = self.bag_feat_fuse(B)
        B_fuse = torch.dstack([B, B_ins, B_bag])
        B_fuse = self.embedding(B_fuse)
        return B_fuse


class KnowledgeInsFuseBlock(nn.Module):

    def __init__(self, args, d, knowledge_ins, n_ins):
        super(KnowledgeInsFuseBlock, self).__init__()
        self.knowledge_ins = knowledge_ins.squeeze(0)
        self.skip = SkipConnectBlock(args, d)
        self.skip2 = SkipConnectBlock(args, d)
        self.skip3 = SkipConnectBlock(args, self.knowledge_ins.shape[0])
        self.mask = MaskBlock(args, self.knowledge_ins.shape[0])
        self.n_ins = n_ins

    def forward(self, X):
        # Compute affinity with the instance representations.
        X_skip = self.skip(X)
        ins_skip = self.skip2(self.knowledge_ins.unsqueeze(0))
        affinity = euclidean_distances(X_skip.cpu().squeeze(0).detach().numpy(), ins_skip.cpu().squeeze(0).detach().numpy())
        affinity = torch.from_numpy(affinity).float().unsqueeze(0).to(device)
        affinity_skip = self.skip3(affinity)
        idx = self.mask(affinity_skip)
        idx = torch.argsort(idx, dim=1, descending=True)[0][:self.n_ins]
        affinity_skip = nn.Softmax(dim=2)(affinity_skip[:, :, idx])

        return affinity_skip


class KnowledgeBagFuseBlock(nn.Module):

    def __init__(self, args, d, knowledge_bag, n_bag):
        super(KnowledgeBagFuseBlock, self).__init__()
        self.ins_space = knowledge_bag[0]
        self.ins_idx = knowledge_bag[1]
        self.n_bag = n_bag
        self.skip = SkipConnectBlock(args, d)
        self.skip2 = SkipConnectBlock(args, d)
        self.skip3 = SkipConnectBlock(args, self.ins_space.shape[1])
        self.mask = MaskBlock(args, self.ins_space.shape[1])

    def forward(self, X):
        # X = X.squeeze(0)
        X_skip = self.skip(X)
        ins_skip = self.skip2(self.ins_space)
        affinity = euclidean_distances(X_skip.cpu().squeeze(0).detach().numpy(), ins_skip.cpu().squeeze(0).detach().numpy())
        affinity = torch.from_numpy(affinity).float().unsqueeze(0).to(device)
        affinity_skip = self.skip3(affinity)
        mask = self.__get_mask(affinity_skip)
        affinity_skip = nn.Softmax(dim=2)(affinity_skip[:, :, mask])
        return affinity_skip

    def __get_mask(self, X):
        mask = self.mask(X).squeeze(0)
        mask_update = []
        for i in range(len(self.ins_idx) - 1):
            start, end = self.ins_idx[i], self.ins_idx[i + 1]
            mask_i = torch.argmax(mask[start: end]).cpu().detach().numpy().tolist() + self.ins_idx[i]
            mask_update.append(mask_i)

        return mask_update


class SkipConnectBlock(nn.Module):

    def __init__(self, args, d):
        super(SkipConnectBlock, self).__init__()
        self.args = args
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=self.args.H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.args.H_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=self.args.H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.args.H_c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(d)
        )

    def forward(self, X):
        X = X.to(device)
        X_t = X.permute(0, 2, 1)
        X1 = self.conv1(X_t).permute(0, 2, 1)
        X2 = self.conv2(X_t)
        X3 = self.conv3(X_t).permute(0, 2, 1)

        feat = torch.matmul(X1, X2)
        feat = nn.Softmax(dim=1)(feat)
        return torch.matmul(feat, X3) + X


class MaskBlock(nn.Module):

    def __init__(self, args, d):
        super(MaskBlock, self).__init__()
        self.args = args
        self.fc1 = nn.Sequential(nn.Linear(d, self.args.H_k), nn.Dropout(self.args.drop_out), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(d, self.args.H_k), nn.Dropout(self.args.drop_out), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(self.args.H_k, d), nn.Dropout(self.args.drop_out), nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(self.args.H_k, 1))

    def forward(self, X):
        X1 = self.fc1(X).permute(0, 2, 1)
        X2 = self.fc2(X)
        A = self.fc3(torch.matmul(X1, X2)).permute(0, 2, 1)
        A = self.fc4(A).squeeze(-1)
        A = nn.Softmax(dim=1)(A)
        return A


class AttentionBlock(nn.Module):

    def __init__(self, args, d):
        super(AttentionBlock, self).__init__()
        self.args = args
        self.d = d
        self.feature_extractor_part = nn.Sequential(nn.Linear(d, self.args.H_a),
                                                    nn.Dropout(self.args.drop_out), nn.LeakyReLU())
        self.attention_V = nn.Sequential(nn.Linear(self.args.H_a, self.args.D_a),
                                         nn.Dropout(self.args.drop_out), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.args.H_a, self.args.D_a),
                                         nn.Dropout(self.args.drop_out), nn.LeakyReLU())
        self.attention_weights = nn.Sequential(nn.Linear(self.args.D_a, 1),
                                               nn.Dropout(self.args.drop_out), nn.LeakyReLU())
        self.embedding = nn.Sequential(nn.Linear(self.args.H_a, d), nn.LeakyReLU())

    def forward(self, X):
        X = X.squeeze(0)
        X = self.feature_extractor_part(X)

        A_V = self.attention_V(X)
        A_U = self.attention_U(X)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = nn.Softmax(dim=1)(A)

        M = torch.mm(A, X)
        M = self.embedding(M)

        return M, A


class SelfAttentionBlock(nn.Module):

    def __init__(self, args, d):
        super(SelfAttentionBlock, self).__init__()
        self.args = args
        self.att = AttentionBlock(args, d * 2)
        self.embedding = nn.Sequential(nn.Linear(d * 2, self.args.H_s), nn.Dropout(self.args.drop_out), nn.LeakyReLU())

    def forward(self, X_fuse, X_fuse_att):
        X_fuse_att = torch.tile(X_fuse_att, dims=(X_fuse.shape[1], 1))
        X_fuse = torch.hstack([X_fuse.squeeze(0), X_fuse_att])
        b, A = self.att(X_fuse)
        b = self.embedding(b)

        return b, A


class BagLoader(data_utils.Dataset):

    def __init__(self, bags, bags_label, idx=None):
        """"""
        self.bags = bags
        self.idx = idx
        if self.idx is None:
            self.idx = list(range(len(self.bags)))
        self.num_idx = len(self.idx)
        self.bags_label = bags_label[self.idx]

    def __getitem__(self, idx):
        bag = [self.bags[self.idx[idx], 0][:, :-1].tolist()]
        bag = torch.from_numpy(np.array(bag))

        return bag.float(), torch.tensor([self.bags_label[idx].tolist()]).float()

    def __len__(self):
        """"""
        return self.num_idx
