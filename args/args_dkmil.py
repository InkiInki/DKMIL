"""For knowledge"""
n_bag_center = 3
n_bag_density = 3

n_ins_center = 3
n_ins_density = 3

n_sample = 10
n_ins_msk = 3

"""For experiments"""
n_cv = 5
seed = 1

"""For model"""
n_mask = 10
r_mask = 0.1
drop_out = 0.1
H_k = 256
H_a = 128
D_a = 64
H_c = 16
H_s = 64
D_s = 128
lambda_l1 = 1e-5
epoch = 50
lr = 0.0001
weight_decay = 0.00005
# Elephant: H = 16, D = 8
# print((0.8 + 0.675 + 0.725 + 0.625 + 0.6) / 5)
