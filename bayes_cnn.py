"""
Code for standard Stochastic Gradient MCMC
(c) Wei Deng
Nov 1, 2020
"""

#!/usr/bin/python

import math
import copy
import sys
import os
import timeit
import csv
import argparse
from math import exp
from sys import getsizeof
import numpy as np
import random
import pickle
## import pytorch modules
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets

from tools import loader
from trainer import sampling

import models.cifar as cifar_models

def main():
    parser = argparse.ArgumentParser(description='Grid search')
    parser.add_argument('-aug', default=1, type=float, help='Data augmentation or not')
    parser.add_argument('-sn', default=1000, type=int, help='Sampling Epochs')
    parser.add_argument('-wdecay', default=25, type=float, help='Samling weight decay')
    parser.add_argument('-lr', default=2e-6, type=float, help='Sampling learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='Sampling momentum learning rate')

    parser.add_argument('-T', default=0.01, type=float, help='Inverse temperature for high temperature chain')
    parser.add_argument('-anneal', default=1.002, type=float, help='temperature annealing factor')
    parser.add_argument('-lr_anneal', default=0.992, type=float, help='lr annealing factor')

    parser.add_argument('-data', default='cifar100', dest='data', help='Fashion MNIST/ CIFAR10/ CIFAR100')
    parser.add_argument('-model', default='resnet', type=str, help='resnet')
    parser.add_argument('-depth', type=int, default=2, help='Model depth.')
    parser.add_argument('-total', default=50000, type=int, help='Total data points')
    parser.add_argument('-batch', default=256, type=int, help='batch size')
    parser.add_argument('-seed', default=random.randint(1, 1e6), type=int, help='Random Seed')
    parser.add_argument('-gpu', default=0, type=int, help='Default GPU')


    pars = parser.parse_args()
    """ Step 0: Numpy printing setup and set GPU and Seeds """
    print(pars)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    try:
        torch.cuda.set_device(pars.gpu)
    except: # in case the device has only one GPU
        torch.cuda.set_device(0) 
    torch.manual_seed(pars.seed)
    torch.cuda.manual_seed(pars.seed)
    np.random.seed(pars.seed)
    random.seed(pars.seed)
    torch.backends.cudnn.deterministic=True

    """ Step 1: Preprocessing """
    if not torch.cuda.is_available():
        exit("CUDA does not exist!!!")
    if pars.model == 'resnet':
        if pars.data == 'fmnist':
            net = fmnist_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()
        elif pars.data == 'cifar10':
            net = cifar_models.__dict__['resnet'](num_classes=10, depth=pars.depth).cuda()
        elif pars.data == 'cifar100':
            net = cifar_models.__dict__['resnet'](num_classes=100, depth=pars.depth).cuda()

    """ Step 2: Load Data """
    train_loader, test_loader, targetloader = loader(pars.batch, pars.batch, pars)
    
    """ Step 3: Bayesian Sampling """
    sampling(net, train_loader, test_loader, pars)
    

if __name__ == "__main__":
    main()
