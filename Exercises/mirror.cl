__kernel void mirror(__global uchar* input, __global uchar* output, int width,int height, int horizontal) {
        int gid_x = get_global_id(0);
        int gid_y = get_global_id(1);

        int original_index = gid_y * width * 3 + gid_x * 3;
        int mirrored_index;

        if(horizontal==1){
            mirrored_index = gid_y * width * 3 + (width - 1 - gid_x) * 3;
        }
        else{
            mirrored_index = (height - 1 - gid_y) * width * 3 + gid_x * 3;
        }

        for (int i = 0; i < 3; ++i) {
            output[original_index + i] = input[mirrored_index + i];
        }
    }

