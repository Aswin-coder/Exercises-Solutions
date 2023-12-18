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
