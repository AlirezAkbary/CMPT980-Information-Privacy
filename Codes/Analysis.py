
import argparse

import numpy as np
import torch
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm
from Model import *
import random
import os
random.seed(1)

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Argument
device = 'cpu'
sample_size = 20

weights_change = {'conv1.weight': [], 'conv2.weight': [], 'fc1.weight': [], 'fc2.weight': []}

for i in range(sample_size):
    model_True = SampleConvNet().to(device)
    model_name = 'mnist_cnn_mnist_' + str(i) + '_True'
    prefixed = [filename for filename in os.listdir('.') if filename.startswith(model_name)]
    print(prefixed)
    model_True.load_state_dict(torch.load(prefixed[0]))
    model_True.eval()

    model_False = SampleConvNet().to(device)
    model_name = 'mnist_cnn_mnist_' + str(i) + '_False'
    prefixed = [filename for filename in os.listdir('.') if filename.startswith(model_name)]
    print(prefixed)
    model_False.load_state_dict(torch.load(prefixed[0]))
    model_False.eval()

    for param_tensor in model_True.state_dict():
        if 'weight' in param_tensor:
            weights_change[param_tensor].append(torch.norm(model_True.state_dict()[param_tensor] - model_False.state_dict()[param_tensor]))
            #print(param_tensor, "\t", torch.norm(model_True.state_dict()[param_tensor] - model_False.state_dict()[param_tensor]))



for a in weights_change:
    nparr = np.array(weights_change[a])
    print(a, " : ", np.mean(nparr))


