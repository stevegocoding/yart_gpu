#include "test_mt.h"
#include "cuda_rng.h"
#include "cuda_mem_pool.h"

__global__ void kernel_test_mt(float *d_rands, c_randoms_chunk out_chunk)
{
	uint32 elem_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	float rnd = d_rands[elem_idx]; 
	
	out_chunk.d_array[elem_idx] = rnd; 
}

extern "C"
void kernel_wrapper_test_mt(c_randoms_chunk& out_chunk)
{
	uint32 n = out_chunk.num_elems;
	
	dim3 block_size = dim3(16, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(n, block_size.x), 1); 
	
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	uint32 num_rand = rng.get_aligned_cnt(n); 
	c_cuda_memory<float> d_rands(num_rand);

	rng.seed(rand()); 
	rng.gen_rand(d_rands.buf_ptr(), num_rand); 
	
	kernel_test_mt<<<grid_size, block_size>>>(d_rands.buf_ptr(), out_chunk); 
}