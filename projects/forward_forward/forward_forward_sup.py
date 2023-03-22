"""Simple supervised version of the FF algorithm where the label is included in the input.
Positive examples have the correct label in the input image.
Negative examples have an incorrect labe in the input image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython

class ForwardForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass


m = ForwardForwardNet()