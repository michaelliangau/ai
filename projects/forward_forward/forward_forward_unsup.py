"""Simple supervised version of the FF algorithm where the label is included in the input.
Positive examples have the correct label in the input image.
Negative examples have an incorrect labe in the input image.
"""
# Project imports
import sys

sys.path.append("../..")
import common.utils as common_utils

# Third party imports
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import IPython
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import shutil
import argparse

# Parse argments
parser = argparse.ArgumentParser()
parser.add_argument("--delete_output_dir", type=bool, default=False)
args = parser.parse_args()

# Comet logging
common_utils.start_comet_ml_logging("michaelliang-dev")

# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load .pt dataset
train_neg_input = torch.load("mnist_unsup_neg/train.pt")
# Load normal mnist dataset
train_input = datasets.MNIST(
    root="data", train=True, transform=transforms.ToTensor(), download=True
)
test_input = datasets.MNIST(
    root="data", train=False, transform=transforms.ToTensor(), download=True
)

train_neg_input_tensor = torch.zeros(len(train_neg_input), 784)
train_neg_label_tensor = torch.zeros(len(train_neg_input), 1)

for i, data in tqdm(enumerate(train_neg_input), total=len(train_neg_input)):
    input, label = data
    input = input.view(-1, 784)
    train_neg_input_tensor[i] = input
    train_neg_label_tensor[i] = label

# do the same thing as above in the for loop but for the train_input and test_input
train_input_tensor = torch.zeros(len(train_input), 784)
train_label_tensor = torch.zeros(len(train_input), 1)

for i, data in tqdm(enumerate(train_input), total=len(train_input)):
    input, label = data
    input = input.view(-1, 784)
    train_input_tensor[i] = input
    train_label_tensor[i] = label

test_input_tensor = torch.zeros(len(test_input), 784)
test_label_tensor = torch.zeros(len(test_input), 1)

for i, data in tqdm(enumerate(test_input), total=len(test_input)):
    input, label = data
    input = input.view(-1, 784)
    test_input_tensor[i] = input
    test_label_tensor[i] = label

# Move to cuda
train_neg_input = train_neg_input_tensor.to(device)
train_neg_label = train_neg_label_tensor.to(device)
train_input = train_input_tensor.to(device)
train_label = train_label_tensor.to(device)
test_input = test_input_tensor.to(device)
test_label = test_label_tensor.to(device)

# Build the forward forward net
if args.delete_output_dir and os.path.exists("output"):
    shutil.rmtree("output")
    os.makedirs("output")
elif not os.path.exists("output"):
    os.makedirs("output")

torch.manual_seed(42)

# Hyperparameters
lr = 1e-5
batch_size = 1024
num_epoch = 100
eval_save_interval = 10000


class ForwardForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = LinearBlock(in_features=784, out_features=2000)
        self.hidden2 = LinearBlock(in_features=2000, out_features=2000)
        self.hidden3 = LinearBlock(in_features=2000, out_features=2000)
        self.hidden4 = LinearBlock(in_features=2000, out_features=2000)
        self.output = nn.Linear(6000, 10)
        self.softmax = nn.Softmax(dim=1)
        self.output_optimizer = torch.optim.Adam(self.output.parameters(), lr=lr)

    def forward(self, x_pos, x_neg, train_target):
        out1_pos, out1_neg, loss1 = self.hidden1(x_pos, x_neg)
        out2_pos, out2_neg, loss2 = self.hidden2(out1_pos, out1_neg)
        out3_pos, out3_neg, loss3 = self.hidden3(out2_pos, out2_neg)
        out4_pos, _, loss4 = self.hidden4(out3_pos, out3_neg)

        # # concatenate final 3 layers and put it through a classifier layer
        cat_pos = torch.cat((out2_pos, out3_pos, out4_pos), dim=1)
        logits_pos = self.output(cat_pos)
        train_target = train_target.squeeze().long()
        loss = F.cross_entropy(logits_pos, train_target)
        self.output_optimizer.zero_grad()
        loss.backward()
        self.output_optimizer.step()

        # average all loss values
        loss = (loss1 + loss2 + loss3 + loss4 + loss) / 5
        return loss

    def inference(self, x):
        out1 = self.hidden1.inference(x)
        out2 = self.hidden2.inference(out1)
        out3 = self.hidden3.inference(out2)
        out4 = self.hidden4.inference(out3)
        cat = torch.cat((out2, out3, out4), dim=1)
        logits = self.output(cat)
        probs = self.softmax(logits)
        return probs


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(out_features)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.threshold = 2.0

    def forward(self, x_pos, x_neg):
        # Compute goodness
        g_pos = self.relu(self.linear(x_pos.detach())).pow(2).mean(1)
        g_neg = self.relu(self.linear(x_neg.detach())).pow(2).mean(1)
        loss = torch.log(
            1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Layer Norm
        out_pos = self.layernorm(self.relu(self.linear(x_pos))).detach()
        out_neg = self.layernorm(self.relu(self.linear(x_neg))).detach()

        return out_pos, out_neg, loss

    def inference(self, x):
        return self.layernorm(self.relu(self.linear(x)))


# Merge train_input and train_neg_input and create a dataloader out of the merge
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_input, train_neg_input, train_label),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_input, test_label),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)


def evaluate_model():
    # Evaluate
    m.eval()
    correct, total = 0, 0
    for iter, (x, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = x.view(batch_size, -1).float()
        target = target.squeeze().long()
        probs = m.inference(x)
        _, pred = torch.max(probs, dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    print(f"Accuracy: {correct/total}")
    m.train()


m = ForwardForwardNet()
m.to(device)

# Print num params
num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

# Load checkpoint
# m.load_state_dict(torch.load("./final_sup_checkpoint.pt"))

# Train the forward forward net
for epoch in range(num_epoch):
    for iter, (x_pos, x_neg, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        x_pos = x_pos.view(batch_size, -1).float()
        x_neg = x_neg.view(batch_size, -1).float()
        loss = m(x_pos, x_neg, target)

    # print and save end of every epoch
    print(f"Epoch {epoch+1}: Loss {loss.item()}")
    torch.save(m.state_dict(), f"./output/ff_net_{epoch+1}.pt")
    evaluate_model()
