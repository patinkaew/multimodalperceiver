import numpy as np
import torch
import torch.nn as nn

def getNumModelParameters(model):
    return sum(p.numel() for p in model.parameters())
