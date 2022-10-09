//实现两个长度为n的tensor的相加, add2_kernel是kernel函数，运行在GPU端的
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

//每个block有1024个线程，一共有n/1024个block。launch_add2是CPU端的执行函数，调用kernel。
void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}