# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:12:25 2019

@author: Sandy
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

from timeit import default_timer as timer
import os

from network import Net
    
#####################################################################
output_dir="./output/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
if torch.cuda.is_available():
    print("Support GPU")
else:
    print("Not support GPU, train based on CPU")

# Normalize:
# input[channel] = (input[channel] - mean[channel]) / std[channel]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
####################################################################
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

################################################################



net = Net()

###################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('device = ', device)
net.to(device)

###################################################################
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def save_mid_model(epoch, i, loss_val, net, optimizer):
    mid_name = str(epoch) + "_" + str(i+1) + "_loss_" + str(round(loss_val, 4)) + ".pt"
    print("save midel result:", mid_name)
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, output_dir + mid_name)
    
###################################################################
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    tm1 = timer()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            tm2 = timer()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000), " tm = ", (tm2-tm1), "s")
            save_mid_model(epoch, i, running_loss / 2000, net, optimizer)
            running_loss = 0.0
            tm1 = tm2           

print('Finished Training')