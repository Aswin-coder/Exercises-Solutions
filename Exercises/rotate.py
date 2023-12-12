import pyopencl as cl
import numpy as np
from PIL import Image

def rotate_opencl(input_image_path, output_image_path, angle_degrees):

    original_image = Image.open(input_image_path)
    original_np_array = np.array(original_image)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=original_np_array)
    output_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=original_np_array.nbytes)

    with open("rotate.cl", "r") as f:
        program_source = f.read()

    program = cl.Program(context, program_source).build()

    height, width, channel = original_np_array.shape

    global_size = (width, height)
    local_size = (16,16)
    local_buffer=cl.LocalMemory(256)

    program.rotate(queue, global_size, local_size, input_buffer, output_buffer, np.int32(width), np.int32(height), np.float32(angle_degrees),local_buffer).wait()

    rotated_np_array = np.empty_like(original_np_array)
    cl.enqueue_copy(queue, rotated_np_array, output_buffer).wait()

    rotated_image = Image.fromarray(rotated_np_array.astype(np.uint8))
    rotated_image.save(output_image_path)

if __name__ == "__main__":
    input_image_path = "image.jpg"
    output_image_path = "rotated_image.jpg"
    rotation_angle = 45.0 

    rotate_opencl(input_image_path, output_image_path, rotation_angle)
