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
/// \brief	Moves data from device memory \a d_vals to device memory \a d_array using \em source
/// 		addresses specified in \a d_srcAddr.
/// 		
/// 		\code d_array[i] = d_vals[d_srcAddr[i]] \endcode
/// 		
/// 		When the source address is \c 0xffffffff, the corresponding target entry will get zero'd.
///			This can be helpful for some algorithms.
/// 		
/// 		\warning	Heavy uncoalesced access possible. Depends on addresses. 
*/ 
// ---------------------------------------------------------------------
template <class T>
__global__ void kernel_set_from_address(T* d_array, uint* d_src_addr, T* d_vals, uint count_target)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < count_target)
	{
		uint addr = d_src_addr[idx];
		T val = {0};
		if(addr != 0xffffffff)
			val = d_vals[addr];
		d_array[idx] = val;
	}
}

__global__ void kernel_init_identity(uint32 *d_buffer, uint32 count)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count)
		d_buffer[idx] = idx;
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

extern "C++"
template <typename T>
void kernel_wrapper_set_from_address(T *d_array, uint32 *d_src_addr, T *d_vals, uint32 count_target)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(count_target, block_size.x), 1, 1);
	
	kernel_set_from_address<<<grid_size, block_size>>>(d_array, d_src_addr, d_vals, count_target);
}

extern "C"
void kernel_wrapper_init_identity(uint32 *d_buffer, uint32 count)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(count, block_size.x), 1, 1);
	
	kernel_init_identity<<<grid_size, block_size>>>(d_buffer, count);
	CUDA_CHECKERROR;
}