#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:49:37 2019

@author: xiping
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from network import Net

batch_size = 1

# Normalize:
# input[channel] = (input[channel] - mean[channel]) / std[channel]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def main(mid_model_name):
    print("=================================================")
    print("test model:", mid_model_name)
    
    checkpoint = torch.load(mid_model_name)
    
    net = Net()
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # must be call
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    print(len(images), type(images[0]))
    
    outputs = net(images)
    
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
        
    prob, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
    print('Predicted: ', ' '.join('%2.2f' % prob[j] for j in range(batch_size)))
    
if __name__ == '__main__':
    mid_model_name = '/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.297938.pt'
    main(mid_model_name)