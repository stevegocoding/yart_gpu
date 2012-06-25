#include "cuda_utils.h"
#include "cuda_defs.h"

// ---------------------------------------------------------------------
/*
	Kernels
*/ 
// ---------------------------------------------------------------------

template <typename V, typename S>
__global__ void kernel_scale_vector_array(V *d_vec, uint32 count, S scalar)
{
	uint32 tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (tid < count)
		d_vec[tid] *= scalar; 
}


// ---------------------------------------------------------------------
/*
	Kernel Wrappers
*/ 
// ---------------------------------------------------------------------
template void kernel_wrapper_scale_vector_array<float4, float>(float4 *d_vec, uint32 count, float scalar); 


extern "C++"
template <typename V, typename S> 
void kernel_wrapper_scale_vector_array(V *d_vec, uint32 count, S scalar)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(CUDA_DIVUP(count, block_size.x), 1, 1); 
	
	kernel_scale_vector_array<V, S><<<grid_size, block_size>>>(d_vec, count, scalar);
}

