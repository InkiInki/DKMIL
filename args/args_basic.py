import os
import torch
import warnings
TORCH_SEED = 1
TORCH_DOWNLOAD_ROOT = "D:/Data/Pretrain/"
RECOMPUTE = False

warnings.filterwarnings("ignore")
os.environ["TORCH_HOME"] = TORCH_DOWNLOAD_ROOT
torch.manual_seed(TORCH_SEED)
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
