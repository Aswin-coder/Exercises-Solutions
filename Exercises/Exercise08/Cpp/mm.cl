
__kernel void mmul(
    const int N,
    __global float* A,
    __global float* B,
    __global float* C,
    __local float* Bwrk)
{
    int k, j;
    int i = get_global_id(0);
    int iloc  =get_local_id(0);
    int nloc =get_local_size(0);

    float Awrk[1024];
    float tmp;
    if (i < N) {
        for (k = 0; k < N; k++)
            Awrk[k] = A[i*N+k];

        for (j = 0; j < N; j++) {
            tmp = 0.0f;
            for (k = 0; k < N; k++)
                tmp += Awrk[k] * B[k*N+j];
            C[i*N+j] = tmp;
        }
    }
}