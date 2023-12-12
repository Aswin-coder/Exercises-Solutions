__kernel void rotate(__global uchar* input, __global uchar* output, int width, int height, float angle, __local uchar* local_buffer)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);
    int lsize_x = get_local_size(0);
    int lsize_y = get_local_size(1);

    angle*=0.01745329f;

    float cos_a = cos(angle);
    float sin_a = sin(angle);
    
    int original_x = (int)((gid_x - width / 2) * cos_a + (gid_y - height / 2) * sin_a + width / 2);
    int original_y = (int)((gid_y - height / 2) * cos_a - (gid_x - width / 2) * sin_a + height / 2);

    local_buffer[lid_y * lsize_x * 3 + lid_x * 3] = input[original_y * width * 3 + original_x * 3];
    local_buffer[lid_y * lsize_x * 3 + lid_x * 3 + 1] = input[original_y * width * 3 + original_x * 3 + 1];
    local_buffer[lid_y * lsize_x * 3 + lid_x * 3 + 2] = input[original_y * width * 3 + original_x * 3 + 2];

    barrier(CLK_LOCAL_MEM_FENCE);

    int rotated_x = lid_x;
    int rotated_y = lid_y;
    
    int rotated_index = rotated_y * lsize_x * 3 + rotated_x * 3;

    output[gid_y * width * 3 + gid_x * 3] = local_buffer[rotated_index];
    output[gid_y * width * 3 + gid_x * 3 + 1] = local_buffer[rotated_index + 1];
    output[gid_y * width * 3 + gid_x * 3 + 2] = local_buffer[rotated_index + 2];
}
