import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.utils.data as data_utils
from sklearn.metrics import euclidean_distances
from args.args_dkmil import H_k, H_c, H_a, D_a, H_s, drop_out, n_mask, r_mask
from utils.func_basic import kernel_rbf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

        return bag.float(), torch.tensor([self.bags_label[idx].tolist()]).float(), self.idx[idx]

    def __len__(self):
        """"""
        return self.num_idx
    

class KnowledgeFuseBlock(nn.Module):
    
    def __init__(self, d, n_bag):
        super(KnowledgeFuseBlock, self).__init__()
        self.n_sample = max(n_mask, int(r_mask * n_bag))
        self.bag_feat_fuse = KnowledgeBagFuseBlock(d, n_bag, self.n_sample)
        self.embedding = nn.Sequential(nn.Linear(d + self.n_sample, d), nn.LeakyReLU())

    def forward(self, B, knowledge_bag):
        B_bag = self.bag_feat_fuse(B, knowledge_bag)
        B_fuse = torch.dstack([B.unsqueeze(0), B_bag])
        B_fuse = self.embedding(B_fuse)

        return B_fuse


class KnowledgeBagFuseBlock(nn.Module):

    def __init__(self, d, n_bag, n_sample):
        super(KnowledgeBagFuseBlock, self).__init__()
        self.n_bag = n_bag
        self.n_sample = n_sample

        self.skip = SkipConnectBlock(d)
        self.skip2 = SkipConnectBlock(d)
        self.skip3 = SkipConnectBlock(n_bag)
        self.mask = MaskBlock(n_bag)

    def forward(self, X, knowledge_bag):
        X_skip = self.skip(X.unsqueeze(0))
        knowledge_bag_skip = self.skip2(knowledge_bag.unsqueeze(0))

        affinity = euclidean_distances(X_skip.cpu().squeeze(0).detach().numpy(),
                                       knowledge_bag_skip.cpu().squeeze(0).detach().numpy())
        affinity = torch.from_numpy(affinity).float().unsqueeze(0).to(device)
        affinity_skip = self.skip3(affinity)
        mask = self.__get_mask(affinity_skip)[:self.n_sample]
        affinity_skip = nn.Softmax(dim=2)(affinity_skip[:, :, mask])
        return affinity_skip

    def __get_mask(self, X):
        mask = self.mask(X).squeeze(0)
        mask = torch.argsort(mask, descending=True)

        return mask

    def __get_affinity(self, X, knowledge_bag):
        embedding1 = torch.hstack([X.min(1).values, X.max(1).values]).cpu().detach().numpy()
        embedding2 = torch.hstack([knowledge_bag.min(1).values, knowledge_bag.max(1).values]).cpu().detach().numpy()
        n1, n2 = X.shape[0], knowledge_bag.shape[0]
        affinity = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                affinity[i][j] = np.power(kernel_rbf(embedding1[i], embedding2[j], 1) + 1, 1)
        return affinity


class SkipConnectBlock(nn.Module):

    def __init__(self, d):
        super(SkipConnectBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(H_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(H_c)
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

    def __init__(self, d):
        super(MaskBlock, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d, H_k), nn.Dropout(drop_out), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(d, H_k), nn.Dropout(drop_out), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(H_k, d), nn.Dropout(drop_out), nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(H_k, 1))

    def forward(self, X):
        X1 = self.fc1(X).permute(0, 2, 1)
        X2 = self.fc2(X)
        A = self.fc3(torch.matmul(X1, X2)).permute(0, 2, 1)
        A = self.fc4(A).squeeze(-1)
        A = nn.Softmax(dim=1)(A)
        return A


class AttentionBlock(nn.Module):

    def __init__(self, d):
        super(AttentionBlock, self).__init__()
        self.d = d
        self.feature_extractor_part = nn.Sequential(nn.Linear(d, H_a), nn.Dropout(drop_out), nn.LeakyReLU())
        self.attention_V = nn.Sequential(nn.Linear(H_a, D_a), nn.Dropout(drop_out), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(H_a, D_a), nn.Dropout(drop_out), nn.LeakyReLU())
        self.attention_weights = nn.Sequential(nn.Linear(D_a, 1), nn.Dropout(drop_out), nn.LeakyReLU())
        self.embedding = nn.Sequential(nn.Linear(H_a, d), nn.LeakyReLU())

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

    def __init__(self, d):
        super(SelfAttentionBlock, self).__init__()
        self.att = AttentionBlock(d * 2)
        self.embedding = nn.Sequential(nn.Linear(d * 2, H_s), nn.Dropout(drop_out), nn.LeakyReLU())

    def forward(self, X_fuse, X_fuse_att):
        X_fuse_att = torch.tile(X_fuse_att, dims=(X_fuse.shape[1], 1))
        X_fuse = torch.hstack([X_fuse.squeeze(0), X_fuse_att])
        b, A = self.att(X_fuse)
        b = self.embedding(b)

        return b, A


class DataTransformBlock(nn.Module):
    """"""

    def __init__(self, d_reshape, d_embedding, in_channels=1):
        super(DataTransformBlock, self).__init__()
        self.d_reshape = d_reshape
        self.data_transform = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # self.embedding = nn.Sequential(nn.Linear(d_reshape, 100), nn.Dropout(drop_out), nn.LeakyReLU(),
        #                                nn.Linear(100, d_embedding), nn.Dropout(drop_out), nn.LeakyReLU())
        self.embedding = nn.Sequential(nn.Linear(d_reshape, d_embedding), nn.Dropout(drop_out), nn.LeakyReLU())

    def forward(self, X):
        X = self.data_transform(X).reshape(-1, self.d_reshape)

        return self.embedding(X)


class DKMIL(nn.Module):

    def __init__(self, d, knowledge, d_reshape, d_embedding=None, in_channels=1):
        super(DKMIL, self).__init__()
        self.d = d
        self.d_embedding = d if d_embedding is None else d_embedding

        self.d_reshape = d_reshape
        self.in_channels = in_channels
        self.knowledge_bag = knowledge.knowledge_bag
        self.drop_out = drop_out
        self.__init_dkmil()

    def forward(self, B):
        """"""
        B = B.squeeze(0)
        B_dt = self.data_transform(B)
        knowledge_bag_dt = self.data_transform(self.knowledge_bag)
        B_fuse = self.fuse(B_dt, knowledge_bag_dt)
        B_fuse_att = self.att(B_fuse)[0]
        b, A = self.self_att(B_fuse, B_fuse_att)
        # b, A = self.att(B_fuse)
        Y_prob = self.classifier(b)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

        return Y_prob, A

    def __init_dkmil(self):
        self.data_transform = DataTransformBlock(self.d_reshape, self.d_embedding, in_channels=self.in_channels)
        self.fuse = KnowledgeFuseBlock(self.d_embedding, len(self.knowledge_bag))
        self.att = AttentionBlock(self.d_embedding)
        self.self_att = SelfAttentionBlock(self.d_embedding)
        self.classifier = nn.Sequential(nn.Linear(H_s, 1), nn.Sigmoid())

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch_init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
