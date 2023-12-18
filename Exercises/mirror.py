import pyopencl as cl
import numpy as np
from PIL import Image


def opencl(image_np):

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    input_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=image_np)
    outputh_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=image_np.nbytes)
    outputv_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=image_np.nbytes)

    with open("image.cl", "r") as f:
        program_source = f.read()

    program = cl.Program(context, program_source).build()

    global_size = (image_np.shape[1], image_np.shape[0])

    program.mirror(queue, global_size, None, input_buffer, outputh_buffer, np.int32(image_np.shape[1]),np.int32(image_np.shape[0]),np.int32(1))
    program.mirror(queue, global_size, None, input_buffer, outputv_buffer, np.int32(image_np.shape[1]),np.int32(image_np.shape[0]),np.int32(0))

    resulth = np.empty_like(image_np)
    resultv = np.empty_like(image_np)

    cl.enqueue_copy(queue, resulth, outputh_buffer).wait()

    cl.enqueue_copy(queue, resultv, outputv_buffer).wait()

    result = np.empty((3 * image_np.shape[0], 3 * image_np.shape[1], image_np.shape[2]), dtype=np.uint8)

    result[:image_np.shape[0], image_np.shape[1]:2*image_np.shape[1], :] = resultv  

    result[image_np.shape[0]:2*image_np.shape[0], :image_np.shape[1], :] = resulth
    result[image_np.shape[0]:2*image_np.shape[0], image_np.shape[1]:2*image_np.shape[1], :] = image_np 
    result[image_np.shape[0]:2*image_np.shape[0], 2*image_np.shape[1]:, :] = resulth 

    result[2*image_np.shape[0]:, image_np.shape[1]:2*image_np.shape[1], :] = resultv

    print(result.shape)



    return result

if __name__ == "__main__":
    input_image_path = "image.jpg"
    output_image_path = "output_image.jpg"

    original_image = Image.open(input_image_path)

    original_np_array = np.array(original_image)
    print("input"+str(original_np_array.shape))

    opencl_array = opencl(original_np_array)

    opencl_image = Image.fromarray(opencl_array)

    opencl_image.save(output_image_path)
