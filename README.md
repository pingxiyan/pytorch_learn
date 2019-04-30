# README
Pytorch study website:	<br>

https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html <br>
https://pytorch.org/  <br>
https://morvanzhou.github.io/tutorials/machine-learning/torch/ <br>

# Install
#### Windows Binary

    $ start anaconda prompt by adminstrator
    $ conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

Windows Know issues:

**Q:** File "C:\ProgramData\Anaconda3\lib\multiprocessing\reduction.py", line 60, in dump   <br>
ForkingPickler(file, protocol).dump(obj)    <br>
BrokenPipeError: [Errno 32] Broken pipe     <br>
    
**A:** Windows OS have problem about multi-thread load data, so you just need to set num_workers=0  <br>
For example: <br>
torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0) <br>
                                          
#### Ubuntu

    $ pip3 install torch torchvision
    
# Train CNN classifier

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py <br>

#### Classify CIFAR10 Get Started

**pytorch train**
refer: pytorch_learn/train_cnn_cifar10/train.py

**pytorch test**
refer: pytorch_learn/train_cnn_cifar10/test.py

**cpp inference**
refer: pytorch_learn/test_cifar10_cpp
detail refer:pytorch_learn/test_cifar10_cpp/readme.md