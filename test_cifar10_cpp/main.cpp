#include <torch/torch.h>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/script.h> // One-stop header.

// class Net(nn.Module):
//     def __init__(self):
//         super(Net, self).__init__()
//         self.conv1 = nn.Conv2d(3, 6, 5)
//         self.pool = nn.MaxPool2d(2, 2)
//         self.conv2 = nn.Conv2d(6, 16, 5)
//         self.fc1 = nn.Linear(16 * 5 * 5, 120)
//         self.fc2 = nn.Linear(120, 84)
//         self.fc3 = nn.Linear(84, 10)
//     def forward(self, x):
//         x = self.pool(F.relu(self.conv1(x)))
//         x = self.pool(F.relu(self.conv2(x)))
//         x = x.view(-1, 16 * 5 * 5)
//         x = F.relu(self.fc1(x))
//         x = F.relu(self.fc2(x))
//         x = self.fc3(x)
//         #x = F.softmax(x) # only for test
//         return x

//struct Net : torch::nn::Module {
//  Net()
//      : conv1(torch::nn::Conv2dOptions(3, 6, /*kernel_size=*/5)),
//        conv2(torch::nn::Conv2dOptions(6, 16, /*kernel_size=*/5)),
//        fc1(6*16*5, 120),
//        fc2(120, 84),
//        fc3(84, 10) {
//    register_module("conv1", conv1);
//    register_module("conv2", conv2);
//    register_module("fc1", fc1);
//    register_module("fc2", fc2);
//    register_module("fc3", fc3);
//  }
//
//  torch::Tensor forward(torch::Tensor x) {
//    x = torch::max_pool2d(torch::relu(conv1->forward(x)), 2);
//    x = torch::max_pool2d(torch::relu(conv2->forward(x)), 2);
//    x = x.view({-1, 16*5*5});
//    x = torch::relu(fc1->forward(x));
//    x = torch::relu(fc2->forward(x));
//    x = fc3->forward(x);
//    return torch::log_softmax(x, /*dim=*/1);
//  }
//
//  torch::nn::Conv2d conv1;
//  torch::nn::Conv2d conv2;
//  torch::nn::Linear fc1;
//  torch::nn::Linear fc2;
//  torch::nn::Linear fc3;
//};
//
//void test(std::string model_fn, cv::Mat src) {
//	torch::DeviceType device_type;
//	if (torch::cuda::is_available()) {
//		std::cout << "CUDA available! Training on GPU." << std::endl;
//		device_type = torch::kCUDA;
//	} else {
//		std::cout << "Training on CPU." << std::endl;
//		device_type = torch::kCPU;
//  	}
//
//	torch::Device device(device_type);
//
//	Net model();
//	model.to(device);
//
////	torch::load()
//}

//"1_12000_loss_1.2968.pt"
//"1_12000_loss_1.297938.pt""


int main() {
#ifdef WIN32
	std::string mpath = "C:\\SandyWork\\chongqing_work\\Human-Segmentation-PyTorch\\UNet_MobileNetV2.tar\\UNet_MobileNetV2\\";
	std::string model_path = mpath + "UNet_MobileNetV2.pth";
	model_path = "C:\\SandyWork\\mygithub\\pytorch_learn\\train_cnn_cifar10\\output\\1_12000_loss_1.2831.pts";
	std::string image_path = "../cat.jpg";
#else
	std::string model_path = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/output/1_12000_loss_1.2968.pt";
	std::string image_path = "/home/xiping/mygithub/pytorch_learn/train_cnn_cifar10/bb.bmp";
#endif

	cv::Mat image = cv::imread(image_path);
	cv::Mat rsz;
	cv::resize(image, rsz, cv::Size(32, 32));
	//cv::namedWindow("test", 1);
	//cv::imshow("test", image);
	//cv::waitKey(0);

	std::cout << "run to:" << __LINE__ << std::endl;
	std::shared_ptr<torch::jit::script::Module> module;
	std::cout << "run to:" << __LINE__ << std::endl;
	std::cout << model_path << std::endl;

	// Deserialize the ScriptModule from a file using torch::jit::load().
	module = torch::jit::load(model_path);
	std::cout << "run to:" << __LINE__ << std::endl;
	assert(module != nullptr);

	std::vector<torch::jit::IValue> inputs;
	std::cout << "run to:" << __LINE__ << std::endl;

	at::Tensor tensor_image = torch::from_blob(rsz.data, {1, 3, rsz.rows, rsz.cols}, at::kByte);
	tensor_image = tensor_image.to(at::kFloat);

	inputs.push_back(tensor_image);

	// Execute the model and turn its output into a tensor.
	at::Tensor output = module->forward(inputs).toTensor();

	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';
}
