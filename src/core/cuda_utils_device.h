#ifndef __cuda_utils_device_h__
#define __cuda_utils_device_h__

#include "math_utils.h"
#include "cuda_utils.h"

 
// ---------------------------------------------------------------------
/*
/// \brief	Fast parallel reduction using loop-unrolling and sync-free operations within warps. 
///
///			Implemented after NVIDIA's CUDA SDK example. 
*/ 
// ---------------------------------------------------------------------

template <typename T, uint32 block_size, class op_functor>
__device__ T device_reduce_fast(T *s_data, op_functor op = op_functor()) 
{ 
	uint32 tid = threadIdx.x;

	if (block_size >= 512)
	{
		if (tid < 256)
			s_data[tid] = op(s_data[tid], s_data[tid + 256]);
		__syncthreads();
	}
	
	if (block_size >= 256)
	{
		if (tid < 128)
			s_data[tid] = op(s_data[tid], s_data[tid + 128]);
		__syncthreads();
	}

	if (block_size >= 128)
	{
		if (tid < 64)
			s_data[tid] = op(s_data[tid], s_data[tid + 64]);
		__syncthreads();
	} 

	// Instructions are SIMD synchronous within a warp (32 threads). Therefore no
	// synchronization is required here.
	
	if(tid < 32)
	{
		// Use volatile to avoid compiler optimizations that reorder the store operations.
		volatile T* sv_data = s_data;

		if(block_size >= 64)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 32]);
		if(block_size >= 32)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 16]);
		if(block_size >= 16)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 8]);
		if(block_size >= 8)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 4]);
		if(block_size >= 4)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 2]);
		if(block_size >= 2)
			sv_data[tid] = op(sv_data[tid], sv_data[tid + 1]);
	}
	
	// Need a sync here since only then all threads will return the same value.
	__syncthreads();
	return s_data[0]; 
}


inline __device__ uint32 device_count_bits(uint32 value)
{
#define TWO(c) (0x1u << (c))
#define MASK(c) (((unsigned long long int)(-1)) / (TWO(TWO(c)) + 1u))
#define COUNT(x,c) ((x) & MASK(c)) + (((x) >> (TWO(c))) & MASK(c))

	value = COUNT(value, 0);
	value = COUNT(value, 1);
	value = COUNT(value, 2);
	value = COUNT(value, 3);
	value = COUNT(value, 4);
	// value = COUNT(value, 5); Shift count too large warning when compiling...
	return value;
}

inline __device__ uint32 device_count_bits(unsigned long long int value)
{
	uint result = device_count_bits((uint32)(value & 0x00000000FFFFFFFF));
	result += device_count_bits((uint32)((value & 0xFFFFFFFF00000000) >> 32));
	return result;
}


#endif // __cuda_utils_device_h__