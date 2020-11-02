#!/usr/bin/python
import math
import copy
import sys
import os
import timeit
import csv
import dill
import argparse
import random
from random import shuffle

from tqdm import tqdm ## better progressbar
from math import exp
from sys import getsizeof
import numpy as np

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

## Import helper functions
from tools import model_eval, BayesEval
from sgmcmc import Sampler

CUDA_EXISTS = torch.cuda.is_available()

def sampling(net, train_loader, test_loader, pars):
    bma = BayesEval()
    
    start = timeit.default_timer()
    if pars.model.startswith('resnet'):
        criterion = nn.CrossEntropyLoss()
        sampler = Sampler(net, criterion, lr=pars.lr, wdecay=pars.wdecay, T=pars.T, total=pars.total)
    
    for epoch in range(pars.sn):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda() if CUDA_EXISTS else Variable(images)
            labels = Variable(labels).cuda() if CUDA_EXISTS else Variable(labels)
            loss = sampler.step(images, labels)

        """ Anneal learning rate and temperature """
        if epoch > (0.4 * pars.sn):
            sampler.eta *= pars.lr_anneal
        sampler.T /= pars.anneal
        sampler.update_hyper()
        """ Bayesian model average or ensemble """
        bma.eval(net, test_loader, weight=1.0, bma=True, burnIn=int(0.6*pars.sn))

        print('Epoch {} Acc: {:0.2f} BMA: {:0.2f} Best Acc: {:0.2f} Best BMA: {:0.2f} T: {:.3E}  Loss: {:0.1f}'.format(\
                    epoch, bma.cur_acc, bma.bma_acc, bma.best_cur_acc, bma.best_bma_acc, sampler.T, loss))
        
    end = timeit.default_timer()
    print("Sampling Time used: {:0.1f}".format(end - start))
    model_eval(net, test_loader)

