#include <opencv2/opencv.hpp>
#include <fstream>
#include "NvInfer.h"
#include "NvOnnxParser.h"

/*
将onnx model转 engine ，然后用trt推理
*/

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR: return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO: return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknown";
	}
}


class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			if (severity == Severity::kWARNING) printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity == Severity::kERROR) printf("\031[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else printf("%s: %s\n", severity_string(severity), msg);
		}
	}
};


bool isFileExists(const char* filename)
{
	std::ifstream f(filename);
	return f.good();
}



bool buildModel(const char* onnxPath, const char* enginePath)
{

	TRTLogger logger;

	// 下面的builder, config, network是基本需要的组件
	// 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	// 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	// 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// onnx parser解析器来解析onnx模型
	auto parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile(onnxPath, 1)) {
		printf("Failed to parse classifier.onnx.\n");
		return false;
	}

	// 设置工作区大小
	printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f);
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);

	// Float 
	// config->setFlag(nvinfer1::BuilderFlag::kFP16);

	// 需要通过profile来使得batchsize时动态可变的，这与我们之前导出onnx指定的动态batchsize是对应的
	int maxBatchSize = 1;
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();

	// 设置batchsize的最大/最小/最优值
	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);

	// 直接构建序列化模型（替换 buildEngineWithConfig + serialize）
	nvinfer1::IHostMemory* model_data = builder->buildSerializedNetwork(*network, *config);
	if (model_data == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	// 将序列化数据写入文件
	FILE* f = fopen(enginePath, "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// 逆序destory掉指针
	delete   model_data;
	delete  network;
	delete  config;
	delete  builder;

	printf("Build Done.\n");
	return true;
}


// ******************************加载模型*****************************
std::vector<unsigned char> load_file(const std::string& file)
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open()) return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}



std::vector<float> dataProcessTrt(std::string imagepath, int inputW, int inputH)
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

	// 分配 CHW 缓冲区
	std::vector<float> chw(3 * inputW * inputH);
	float* hwc_ptr = (float*)imageResizeFloat.data;
	for (int i = 0; i < inputW * inputH; ++i) {
		for (int c = 0; c < 3; ++c) {
			chw[c * inputW * inputH + i] = hwc_ptr[i * 3 + c] / std[c];
		}
	}
	return chw;
}



void main() 
{
	// data 
	int inputW = 224, inputH = 224;
	int classNum = 5;
	std::string imagepath = "D:/study/Dinov2/data/0366AD2509H03D27.bmp";
	std::string onnxPath = "D:/study/Dinov2/best_cls_finetune_224x224.onnx";
	std::string enginePath = "D:/study/Dinov2/best_cls_finetune_224x224.engine";
	
	// build
	if (!isFileExists(enginePath.c_str()))
	{
		std::cout << "Engine file not exists, building engine..." << std::endl;
		buildModel(onnxPath.c_str(), enginePath.c_str());
	}

	// load model
	TRTLogger logger;

	std::vector<unsigned char> engine_data = load_file(enginePath);
	nvinfer1::IRuntime*  _runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine*  _engine = (_runtime)->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (_engine == nullptr) {
		printf("Deserialize cuda engine failed.\n");
		delete  (_runtime);
		return ;
	}
	int nbIOTensors = _engine->getNbIOTensors();
	if (nbIOTensors != 2 && nbIOTensors != 3) {
		printf("Must be single input, single or two Output, got %d output.\n", nbIOTensors - 1);
		return;
	}

	nvinfer1::IExecutionContext*  _execution_context = _engine->createExecutionContext();
	cudaStream_t _stream = nullptr;
	cudaStreamCreate(&_stream);

	// memory
	float* _output_data_host = nullptr;
	float* _input_data_device = nullptr, * _output_data_device = nullptr;
	long _inputLength = 1 * 3 * inputW * inputH;
	long _outputLength = 1 * classNum;

	cudaMallocHost((void**)&_output_data_host, _outputLength * sizeof(float));
	cudaMalloc((void**)&_input_data_device, _inputLength * sizeof(float));
	cudaMalloc((void**)&_output_data_device, _outputLength * sizeof(float));
	

	// 1. data process 
	std::vector<float>  inputdata = dataProcessTrt(imagepath, inputW, inputH);
	
	// cpu->gpu
	cudaMemcpyAsync(_input_data_device, inputdata.data(), _inputLength * sizeof(float), cudaMemcpyHostToDevice, _stream);

	// 2. runNet
	// 2. 设置输入输出张量地址（TensorRT 10 需要提前绑定）
	const char* input_name = _engine->getIOTensorName(0);      // 假设第一个是输入
	const char* output_name = _engine->getIOTensorName(1);     // 第二个是输出
	_execution_context->setInputShape(input_name, nvinfer1::Dims4{ 1, 3, inputH, inputW });
	_execution_context->setTensorAddress(input_name, _input_data_device);
	_execution_context->setTensorAddress(output_name, _output_data_device);

	bool success = _execution_context->enqueueV3(_stream);
	cudaStreamSynchronize(_stream); // 同步
	// gpu -> cpu
	
	cudaMemcpyAsync(_output_data_host, _output_data_device, _outputLength * sizeof(float), cudaMemcpyDeviceToHost, _stream);
	
	int clss = -1; float max_score = 0.0;
	for (int i = 0; i < classNum; i++)
	{
		if (max_score < _output_data_host[i])
		{
			max_score = _output_data_host[i];
			clss = i;
		}
		std::cout << _output_data_host[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "image's class is: " << clss << std::endl;

	// 释放资源
	cudaStreamDestroy(_stream);
	cudaFree(_input_data_device);
	cudaFree(_output_data_device);
	cudaFreeHost(_output_data_host);
	delete  _execution_context;
	delete _engine;
	delete _runtime;

	return;
}
