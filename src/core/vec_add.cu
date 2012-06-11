// kernel 
__global__ void kernel_vec_add(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

extern "C"
void ivk_krnl_vec_add(const float *a, const float *b, float *c, int n)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3((n + block_size.x - 1) / block_size.x, 1, 1); 
	kernel_vec_add<<<grid_size, block_size>>>(a, b, c, n); 
}
