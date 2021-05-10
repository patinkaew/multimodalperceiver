import os, sys
sys.path.append(os.path.dirname(sys.path[0]))

import argparse
import torch
from torch import optim
from solver import *
#from torch.utils.tensorboard import SummaryWriter

# setup device
use_gpu = True
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device: {}".format(device))

# parse arguments

# run training

if __name__ = "__main__":
    dataset = torchvision.datasets.CocoCaptions()
    loader_train = torch.utils.Data.DataLoader(dataset_train, batchsize=25, )
    model = CaptioningTransformer()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    solver = CaptioningSolver()
