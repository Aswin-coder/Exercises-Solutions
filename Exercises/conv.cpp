#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>

const int inputDepth = 3;
const int inputHeight = 224;
const int inputWidth = 224;

const int filterSize = 3;

std::vector<int> calculateOutputShape(const std::vector<int>& inputShape, int filterSize, int stride, int padding) {
    int depth = 1 + (inputShape[0] - filterSize + 2 * padding) / stride;
    int height = 1 + (inputShape[1] - filterSize + 2 * padding) / stride;
    int width = 1 + (inputShape[2] - filterSize + 2 * padding) / stride;
    return {depth, height, width};
}

std::string readKernelFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << fileName << std::endl;
        exit(1);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

std::vector<float> verifyConvolution(const std::vector<float>& input_data, const std::vector<float>& kernel, int stride, int padding, const std::vector<int>& outputShape,int inputDepth ,int  inputHeight ,int inputWidth) {
    
    std::vector<float> padded_input((inputDepth + 2 * padding) * (inputHeight + 2 * padding) * (inputWidth + 2 * padding), 0.0f);
    std::vector<float> output_data(outputShape[0] * outputShape[1] * outputShape[2], 0.0f);

    for (int z = 0; z < inputDepth; ++z) {
        for (int y = 0; y < inputHeight; ++y) {
            for (int x = 0; x < inputWidth; ++x) {
                padded_input[(z + padding) * (inputHeight + 2 * padding) * (inputWidth + 2 * padding) + (y + padding) * (inputWidth + 2 * padding) + (x + padding)] = input_data[z * inputHeight * inputWidth + y * inputWidth + x];
            }
        }
    }

    for (int i = 0; i < outputShape[0]; ++i) {
        for (int j = 0; j < outputShape[1]; ++j) {
            for (int k = 0; k < outputShape[2]; ++k) {
                for (int kz = 0; kz < filterSize; ++kz) {
                    for (int ky = 0; ky < filterSize; ++ky) {
                        for (int kx = 0; kx < filterSize; ++kx) {
                            output_data[i * outputShape[1] * outputShape[2] + j * outputShape[2] + k] +=
                                padded_input[(i * stride + kz) * (inputHeight + 2 * padding) * (inputWidth + 2 * padding) +
                                             (j * stride + ky) * (inputWidth + 2 * padding) +
                                             (k * stride + kx)] * kernel[kz * filterSize * filterSize + ky * filterSize + kx];
                        }
                    }
                }
            }
        }
    }

    return output_data;
}

int main() {

    cl::Context context(CL_DEVICE_TYPE_CPU);
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);


    std::vector<float> input_data(inputDepth * inputHeight * inputWidth, 1.0f);
    std::vector<float> filter(filterSize * filterSize * filterSize, 0.5f);

    int stride = 2;
    int padding = 1;

    std::vector<int> outputShape = calculateOutputShape({inputDepth, inputHeight, inputWidth}, filterSize, stride, padding);
    printf("%d,%d,%d\n",outputShape[0] , outputShape[1] , outputShape[2]);


    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_data.size(), input_data.data());
    cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * filter.size(), filter.data());
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * outputShape[0] * outputShape[1] * outputShape[2]);


    std::string kernelSource = readKernelFile("conv.cl");

    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});
    cl::Program program(context, sources);
    program.build("-cl-std=CL1.2");

    cl::Kernel conv_kernel(program, "conv");

    conv_kernel.setArg(0, inputBuffer);
    conv_kernel.setArg(1, filterBuffer);
    conv_kernel.setArg(2, outputBuffer);
    conv_kernel.setArg(3, inputDepth);
    conv_kernel.setArg(4, inputHeight);
    conv_kernel.setArg(5, inputWidth);
    conv_kernel.setArg(6, filterSize);
    conv_kernel.setArg(7, stride);
    conv_kernel.setArg(8, padding);
    conv_kernel.setArg(9, outputShape[0]);
    conv_kernel.setArg(10, outputShape[1]);
    conv_kernel.setArg(11, outputShape[2]);


    queue.enqueueNDRangeKernel(conv_kernel, cl::NullRange, cl::NDRange(outputShape[0], outputShape[1], outputShape[2]));


    std::vector<float> outputData(outputShape[0] * outputShape[1] * outputShape[2]);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * outputData.size(), outputData.data());

    for (int i = 0; i < outputShape[0] * outputShape[1] * outputShape[2]; ++i) {
        //std::cout << outputData[i] << " ";
    }
    std::vector<float> output_cpp = verifyConvolution(input_data, filter, stride, padding,outputShape,inputDepth , inputHeight , inputWidth);

    if (std::equal(outputData.begin(), outputData.end(), output_cpp.begin(), output_cpp.end())) {
        std::cout << "Matched" << std::endl;
    } else {
        std::cout << "Not Matched" << std::endl;
    }

    return 0;
}
