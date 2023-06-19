import torch
import os
import warnings
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
TORCH_DOWNLOAD_ROOT = "D:/Data/Pretrain/"
os.environ["TORCH_HOME"] = TORCH_DOWNLOAD_ROOT
warnings.filterwarnings("ignore")

torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(1)
