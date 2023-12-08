#define __CL_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include "util.hpp"

const int SIZE = 10;

std::string readKernelFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << fileName << std::endl;
        exit(1);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

int main() {
    try {
        cl::Context context(CL_DEVICE_TYPE_CPU);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::CommandQueue queue(context, devices[0]);

        int numRows = SIZE;
        int numCols = SIZE;
        int N = SIZE;

        std::vector<int> A(numRows * numCols);
        std::vector<int> B(numRows * numCols);
        std::vector<int> C(numRows * numCols, 0);

        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<float> distr(0, 10);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                A[i * numCols + j] = distr(eng);
                B[i * numCols + j] = distr(eng);
            }
        }
        float mflops;
        double start_time;     
        double run_time;        
        util::Timer timer; 

        start_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numRows * numCols, A.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numRows * numCols, B.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * numRows * numCols);

        std::string kernelSource = readKernelFile("logicalAND.cl");
        cl::Program::Sources sources;
        sources.push_back({kernelSource.c_str(), kernelSource.length()});
        cl::Program program(context, sources);

        program.build(devices);
        cl::Kernel kernel(program, "logicalAND");

        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, numRows);
        kernel.setArg(4, numCols); 

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(numRows * numCols), cl::NullRange);
        queue.finish();

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * numRows * numCols, C.data());

        std::cout << "Matrix A:\n";
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                std::cout << A[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nMatrix B:\n";
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                std::cout << B[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\nLogical AND:\n";
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                std::cout << C[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }

        kernelSource = readKernelFile("logicalOR.cl");
        cl::Program::Sources sources1;
        sources1.push_back({kernelSource.c_str(), kernelSource.length()});
        cl::Program program1(context, sources1);

        program1.build(devices);
        cl::Kernel kernel1(program1, "logicalOR");

        kernel1.setArg(0, bufferA);
        kernel1.setArg(1, bufferB);
        kernel1.setArg(2, bufferC);
        kernel1.setArg(3, numRows);
        kernel1.setArg(4, numCols); 

        queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(numRows * numCols), cl::NullRange);
        queue.finish();

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * numRows * numCols, C.data());

        std::cout << "\nLogical OR:\n";
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                std::cout << C[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }

        kernelSource = readKernelFile("logicalNOT.cl");
        cl::Program::Sources sources2;
        sources2.push_back({kernelSource.c_str(), kernelSource.length()});
        cl::Program program2(context, sources2);

        program2.build(devices);
        cl::Kernel kernel2(program2, "logicalNOT");

        kernel2.setArg(0, bufferA);
        kernel2.setArg(1, bufferC);
        kernel2.setArg(2, numRows);
        kernel2.setArg(3, numCols); 

        queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(numRows * numCols), cl::NullRange);
        queue.finish();

        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * numRows * numCols, C.data());

        std::cout << "\nLogical NOT of Matrix A:\n";
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                std::cout << C[i * numCols + j] << " ";
            }
            std::cout << std::endl;
        }
        run_time  = (static_cast<double>(timer.getTimeMilliseconds()) / 1000.0) - start_time;
        mflops = 2.0 * N * N * N/(1000000.0f * run_time);
        printf("%.2f seconds at %.1f MFLOPS \n",  run_time,mflops);
        timer.reset();


    } catch (cl::Error err) {
        std::cerr << "OpenCL error: " << err.what() << " (" << err.err() << ")" << std::endl;
    }

    return 0;
}
