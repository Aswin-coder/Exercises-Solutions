/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

The is the original sequential program.  It uses the timer
from the OpenMP runtime library

History: Written by Tim Mattson, 11/99.
         Ported to C++ by Tom Deakin, August 2013

*/

#include "util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include "cl.hpp"
#include "device_picker.hpp"
#include <err_code.h>

#include <cstdio>
static long num_steps = 100000000;
double step;
extern double wtime();   // returns time since some fixed past point (wtime.c)


int main ()
{
    int i;
    double x, pi, sum = 0.0;


    step = 1.0/(double) num_steps;
    try{
        cl_uint deviceIndex = 0;

        // Get list of devices
        std::vector<cl::Device> devices;
        unsigned numDevices = getDeviceList(devices);
        // Check device index in range
        if (deviceIndex >= numDevices)
        {
          std::cout << "Invalid device index (try '--list')\n";
          return EXIT_FAILURE;
        }
        cl::Device device = devices[deviceIndex];
        std::string name;
        getDeviceName(device, name);
        std::cout << "\nUsing OpenCL device: " << name << "\n";
        std::vector<cl::Device> chosen_device;
        chosen_device.push_back(device);
        cl::Context context(chosen_device);
        cl::CommandQueue queue(context, device);

        cl::Buffer bufferNumSteps(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_ulong), &num_steps);
        cl::Buffer bufferStep(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double), &step);
        cl::Buffer bufferSum(context, CL_MEM_WRITE_ONLY, sizeof(double));
        
        util::Timer timer;
        
        cl::Program program(context, util::loadProgram("pi.cl"), true);
        
        // Use cl::Buffer in the kernel declaration
        cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer> picl(program, "pi");
        
        cl::NDRange global(1024);
        
        // Pass bufferSum as the kernel argument
        picl(cl::EnqueueArgs(queue, global), bufferNumSteps, bufferStep, bufferSum);
        
        queue.finish();
        
        // Read the result back to the host
        float result_sum;
        queue.enqueueReadBuffer(bufferSum, CL_TRUE, 0, sizeof(double), &result_sum);
        
        double pi = step * result_sum;
        double run_time = static_cast<double>(timer.getTimeMilliseconds()) / 1000.0;
        printf("\n pi with %ld steps is %lf in %lf seconds\n", num_steps, pi, run_time);
        

    } catch(int n)
    {
        std::cout << "\nError";
    }
}

