__kernel void pi(
    __global long* num_steps,
    __global double* step,
    __global float* sum)
{

    double local_sum = 0.0;


    for (long i = 0; i < *num_steps; i=i+8) {
        double x1 = (i + 0.5) * (*step);
        double x2 = (i +1+ 0.5) * (*step);
        double x3 = (i + 2+0.5) * (*step);
        double x4 = (i + 3+0.5) * (*step);
        double x5 = (i + 4+0.5) * (*step);
        double x6 = (i +5+ 0.5) * (*step);
        double x7 = (i + 6+0.5) * (*step);
        double x8 = (i + 7+0.5) * (*step);
        local_sum = local_sum + 4.0 / (1.0 + x1 * x1);
        local_sum = local_sum + 4.0 / (1.0 + x2 * x2);
        local_sum = local_sum + 4.0 / (1.0 + x3 * x3);
        local_sum = local_sum + 4.0 / (1.0 + x4 * x4);
        local_sum = local_sum + 4.0 / (1.0 + x5 * x5);
        local_sum = local_sum + 4.0 / (1.0 + x6 * x6);
        local_sum = local_sum + 4.0 / (1.0 + x7 * x7);
        local_sum = local_sum + 4.0 / (1.0 + x8 * x8);
    }

    for (long i = *num_steps - (*num_steps % 8); i < *num_steps; ++i) {
        double x = (i + 0.5) * (*step);
        local_sum += 4.0 / (1.0 + x * x);
    }

    *sum = local_sum;
}
