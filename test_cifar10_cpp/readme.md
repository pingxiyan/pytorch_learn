# README
test inference based on pytorch cpp interface.

# Requirement
1. Need to download "libtorch" to inference for cpp

#### Ubuntu
	$ wget https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip
#### Windows

# Convert pickle model to libtorch script model.

	$ train_cnn_cifar10/cvt_model2torchscript.py # refer this script

# Test CPU
In this test. inference image buffer to get same result, that buffer is from pytorch test.

	mkdir build
	cd build
	cmake -DOpenCV_DIR=[path] -DCMAKE_PREFIX_PATH=[path] -DCUDANN_ROOT_DIR=[your cudnn path] ..
	make -j8
	./testapp

**Result:**

	GroundTruth: cat
	Predicted: dog
	Predicted: 97.7508
	
# Test CUDA

	$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
	$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
	$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
	
	refer:https://github.com/zccyman/pytorch-inference/blob/master/pytorch/src/pytorch_interface.cpp
https://oldpan.me/archives/pytorch-c-libtorch-inference

# Known issues
If we use ourself builded OpenCV, don't known why can't link opencv libraries. Errors log as follow: <br>

	$ cmake -DOpenCV_DIR=/home/xiping/opensource/opencv/build ..
	main.cpp:(.text+0xfe1): undefined reference to `cv::imread(std::string const&, int)'
	main.cpp:(.text+0x10bb): undefined reference to `cv::namedWindow(std::string const&, int)'
	main.cpp:(.text+0x1132): undefined reference to `cv::imshow(std::string const&, cv::_InputArray const&)'

Using this /opt/anaconda/anaconda3/share/OpenCV, can normornly link. <br>

	$ cmake -DOpenCV_DIR=/opt/anaconda/anaconda3/share/OpenCV ..
