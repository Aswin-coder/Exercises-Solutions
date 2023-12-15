import numpy as np
import pyopencl as cl

def calculate_output_shape(input_shape, filter_size, stride, padding):
    depth = 1 + (input_shape[0] - filter_size[0] + 2 * padding) // stride
    height = 1 + (input_shape[1] - filter_size[1] + 2 * padding) // stride
    width = 1 + (input_shape[2] - filter_size[2] + 2 * padding) // stride
    return (depth, height, width)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

def conv(input_data, kernel, stride, padding):
    output_shape = calculate_output_shape(input_data.shape, kernel.shape, stride, padding)

    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_data)
    kernel_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output_shape[0] * output_shape[1] * output_shape[2] * np.dtype(np.float32).itemsize)

    kernel_code = """
        __kernel void conv(__global const float* input, __global const float* kernela, __global float* output,
                              const int input_depth, const int input_height, const int input_width,
                              const int kernel_size, const int stride, const int padding,
                              const int output_depth, const int output_height, const int output_width)
{
    int gidz = get_global_id(0);
    int gidy = get_global_id(1);
    int gidx = get_global_id(2);

    int output_index = gidz * output_height * output_width + gidy * output_width + gidx;

    int start_z = gidz * stride - padding;
    int start_y = gidy * stride - padding;
    int start_x = gidx * stride - padding;

    float sum = 0.0f;
    for (int kz = 0; kz < kernel_size; ++kz) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int cur_z = start_z + kz;
                int cur_y = start_y + ky;
                int cur_x = start_x + kx;
                if (cur_z >= 0 && cur_z < input_depth && cur_y >= 0 && cur_y < input_height && cur_x >= 0 && cur_x < input_width) {
                    int input_index = cur_z * input_height * input_width + cur_y * input_width + cur_x;
                    sum += input[input_index] * kernela[kz * kernel_size * kernel_size + ky * kernel_size + kx];
                }
            }
        }
    }

    output[output_index] = sum;
}

    """

    program = cl.Program(context, kernel_code).build()
    conv_kernel = program.conv

    conv_kernel.set_args(input_buffer, kernel_buffer, output_buffer,
                       np.int32(input_data.shape[0]), np.int32(input_data.shape[1]), np.int32(input_data.shape[2]),
                       np.int32(kernel.shape[0]),
                       np.int32(stride), np.int32(padding),
                       np.int32(output_shape[0]), np.int32(output_shape[1]), np.int32(output_shape[2]))



    global_size = (output_shape[0], output_shape[1], output_shape[2])
    local_size = None 
    cl.enqueue_nd_range_kernel(queue, conv_kernel, global_size, local_size)

    output = np.empty(output_shape, dtype=np.float32)
    cl.enqueue_copy(queue, output, output_buffer).wait()

    return output

input_size = (224, 224, 3)
input_data = np.random.random(input_size).astype(np.float32)



filter_size = 3
filter = np.random.random((filter_size,filter_size,filter_size)).astype(np.float32)


stride = 2
padding = 1  

output_opencl = conv(input_data, filter, stride, padding)


print("Input Data:")
print(input_data.shape)
print("Filter:")
print(filter.shape)
#print("Output Volume after 3D Convolution with Stride and Padding using OpenCL:")
#print(output_volume_opencl)
print("Stride : "+str(stride))
print("Padding : "+str(padding))
print("Output Shape:")
print(output_opencl.shape)
#print(output_opencl)

def conv(input_data, kernel, stride, padding):
    padded_input = np.pad(input_data, padding, mode='constant')
    output_shape = calculate_output_shape(input_data.shape, kernel.shape, stride, padding)
    output_data = np.zeros(output_shape)

    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            for k in range(output_shape[2]):
                output_data[i, j, k] = np.multiply(
                    padded_input[i*stride:i*stride+kernel.shape[0],
                                 j*stride:j*stride+kernel.shape[1],
                                 k*stride:k*stride+kernel.shape[2]], kernel).sum()

    return output_data

output_native = conv(input_data, filter, stride, padding)
print(output_native.shape)
#print(output_native)

if(np.allclose(output_native,output_opencl,rtol=1e-05, atol=1e-08)):
    print("Matched")
