import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.utils.data as data_utils
import torchvision.models as models
from sklearn.metrics import euclidean_distances
from utils.func_basic import kernel_rbf
from args.args_basic import device


class DKMIL(nn.Module):

    def __init__(self, args, d, knowledge, d_embedding=None, in_channels=1, extractor="cnn"):
        super(DKMIL, self).__init__()

        self.args = args
        self.d = d
        self.d_embedding = 100 if d_embedding is None else d_embedding
        self.in_channels = in_channels
        self.extractor = extractor
        self.knowledge_bag = knowledge.knowledge_bag.to(device)
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
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def __init_dkmil(self):
        self.data_transform = DataTransformBlock(self.d, self.d_embedding,
                                                 in_channels=self.in_channels, extractor=self.extractor)
        self.fuse = KnowledgeFuseBlock(self.args, self.d_embedding, len(self.knowledge_bag))
        self.att = AttentionBlock(self.args, self.d_embedding)
        self.self_att = SelfAttentionBlock(self.args, self.d_embedding)
        self.classifier = nn.Sequential(nn.Linear(self.args.H_s, 1), nn.Sigmoid())

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            torch_init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)


class KnowledgeFuseBlock(nn.Module):
    
    def __init__(self, args, d, n_bag):
        super(KnowledgeFuseBlock, self).__init__()
        self.n_sample = max(args.n_mask, int(args.r_mask * n_bag))
        self.bag_feat_fuse = KnowledgeBagFuseBlock(args, d, n_bag, self.n_sample)
        self.embedding = nn.Sequential(nn.Linear(d + self.n_sample, d), nn.LeakyReLU())

    def forward(self, B, knowledge_bag):
        B_bag = self.bag_feat_fuse(B, knowledge_bag)
        B_fuse = torch.dstack([B.unsqueeze(0), B_bag])
        B_fuse = self.embedding(B_fuse)

        return B_fuse


class KnowledgeBagFuseBlock(nn.Module):

    def __init__(self, args, d, n_bag, n_sample):
        super(KnowledgeBagFuseBlock, self).__init__()
        self.n_bag = n_bag
        self.n_sample = n_sample

        self.skip = SkipConnectBlock(args, d)
        self.skip2 = SkipConnectBlock(args, d)
        self.skip3 = SkipConnectBlock(args, n_bag)
        self.mask = MaskBlock(args, n_bag)

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

    def __init__(self, args, d):
        super(SkipConnectBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=args.H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(args.H_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=args.H_c, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm1d(args.H_c)
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
        self.fc1 = nn.Sequential(nn.Linear(d, args.H_k), nn.Dropout(args.drop_out), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(d, args.H_k), nn.Dropout(args.drop_out), nn.Tanh())
        self.fc3 = nn.Sequential(nn.Linear(args.H_k, d), nn.Dropout(args.drop_out), nn.LeakyReLU())
        self.fc4 = nn.Sequential(nn.Linear(args.H_k, 1))

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
        self.d = d
        self.feature_extractor_part = nn.Sequential(nn.Linear(d, args.H_a), nn.Dropout(args.drop_out), nn.LeakyReLU())
        self.attention_V = nn.Sequential(nn.Linear(args.H_a, args.D_a), nn.Dropout(args.drop_out), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(args.H_a, args.D_a), nn.Dropout(args.drop_out), nn.LeakyReLU())
        self.attention_weights = nn.Sequential(nn.Linear(args.D_a, 1), nn.Dropout(args.drop_out), nn.LeakyReLU())
        self.embedding = nn.Sequential(nn.Linear(args.H_a, d), nn.LeakyReLU())

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
        self.att = AttentionBlock(args, d * 2)
        self.embedding = nn.Sequential(nn.Linear(d * 2, args.H_s), nn.Dropout(args.drop_out), nn.LeakyReLU())

    def forward(self, X_fuse, X_fuse_att):
        X_fuse_att = torch.tile(X_fuse_att, dims=(X_fuse.shape[1], 1))
        X_fuse = torch.hstack([X_fuse.squeeze(0), X_fuse_att])
        b, A = self.att(X_fuse)
        b = self.embedding(b)

        return b, A


class DataTransformBlock(nn.Module):
    """"""

    def __init__(self, d_reshape, d_embedding, in_channels=1, drop_out=0.1, extractor="cnn"):
        super(DataTransformBlock, self).__init__()
        self.d_reshape = d_reshape
        # self.data_transform = nn.Sequential(
        #     nn.Conv2d(in_channels, 20, kernel_size=5),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.LeakyReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        if extractor == "cnn":
            self.data_transform = nn.Sequential(
                nn.Conv2d(in_channels, 20, kernel_size=5),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
        elif extractor == "cnn2":
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        else:
            self.feature_extractor_part1 = models.resnet50(pretrained=True)
            self.feature_extractor_part1.fc = nn.Identity()
            self.feature_extractor_part1.requires_grad_(False)

        # self.embedding = nn.Sequential(nn.Linear(d_reshape, 100), nn.Dropout(drop_out), nn.LeakyReLU(),
        #                                nn.Linear(100, d_embedding), nn.Dropout(drop_out), nn.LeakyReLU())
        self.embedding = nn.Sequential(nn.Linear(d_reshape, d_embedding), nn.Dropout(drop_out), nn.LeakyReLU())

    def forward(self, X):
        X = self.data_transform(X).reshape(-1, self.d_reshape)

        return self.embedding(X)


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
