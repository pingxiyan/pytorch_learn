# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:12:25 2019

@author: Sandy
"""

import torch
from network import Net

def cvt_model(pickle_model, script_model):
    print("start convert")
    checkpoint = torch.load(pickle_model)
    
    net = Net()
    net.load_state_dict(checkpoint['model_state_dict'])
    #net.cuda()
    net.eval()  # must be call
    
    #example = torch.rand(1,3,32,32).cuda() 
    example = torch.rand(1,3,32,32).cpu()
    
    traced_script_module = torch.jit.trace(net, example)
    traced_script_module.save(script_model)
    
    print("convert complete")
    
if __name__ == '__main__':
    #pickle_model = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2663.pt"
    #script_model = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2831.pts"
    
    pickle_model = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.2715.pt"
    script_model = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.2715.pts"
    cvt_model(pickle_model, script_model)