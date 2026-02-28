#include <opencv2/opencv.hpp>
/*
利用opencv 直接读取onnx 模型进行分类推理
*/
cv::Mat dataProcess(std::string imagepath, int inputW, int inputH)
{
	cv::Mat image = cv::imread(imagepath);
	cv::Mat imageRGB, imageResize, imageResizeFloat, imageTrans;
	cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
	cv::resize(imageRGB, imageResize, cv::Size(inputW, inputH));
	imageResize.convertTo(imageResizeFloat, CV_32FC3, 1.0 / 255.0);

	cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
	cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225);

	// IMAGE - MEAN / STD
	imageResizeFloat = imageResizeFloat - mean;// mean
	// std
	std::vector<cv::Mat> bgrChannels(3);
	cv::split(imageResizeFloat, bgrChannels);
	for (int i = 0; i < 3; i++)
	{
		bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std[i]);
	}

	cv::merge(bgrChannels, imageTrans);

	return imageTrans;
}


int  mainOnnx()
{
	// data 
	int inputW =224, inputH = 224;
	int classNum = 5;

	std::string imagepath = "D:/study/Dinov2/data/0366AD2509H03D27.bmp"; 
	cv::Mat image = cv::imread(imagepath);
	cv::Mat inputImageFloat = dataProcess(imagepath, inputW, inputH);
	cv::Mat _input = cv::dnn::blobFromImage(inputImageFloat);

	// model 
	std::string modelpath = "D:/study/Dinov2/best_cls_finetune_224x224.onnx";
	cv::dnn::Net net = cv::dnn::readNetFromONNX(modelpath);

	// inference	
	net.setInput(_input);
	std::vector<cv::Mat> output_probs;
	std::vector<cv::String> output_layer_names = net.getUnconnectedOutLayersNames();// 获取多输出对应的名称
	net.forward(output_probs, output_layer_names);

	// 3. process outputs
	if (output_probs.size() <= 0)
		return 0;
	
	float* _output_data_host = (float*)output_probs[0].data; // 1 * (4 + nc + 32) * 8400 

	int clss = -1; float max_score = 0.0;
	for(int i = 0; i < classNum; i++)
	{
		if(max_score < _output_data_host[i])
		{
			max_score = _output_data_host[i];
			clss = i;
		}
		std::cout << _output_data_host[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "image's class is: " << clss << std::endl;

	return 1;
}





