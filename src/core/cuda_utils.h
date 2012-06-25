#pragma once 

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <assert.h>
#include "utils.h"
#include "cuda_defs.h" 

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define cuda_safe_call_no_sync(err)	_cuda_safe_call_no_sync(err, __FILE__, __LINE__)

inline void _cuda_safe_call_no_sync(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		yart_log_message("%s(%i) : CUDA Runtime API error : %s.\n", file, line, cudaGetErrorString(err));
		assert(false); 
	}
}

#define CUDA_DIVUP(count, chunkSize) ((count) / (chunkSize) + (((count) % (chunkSize))?1:0))

// ---------------------------------------------------------------------
/*
	Return the aligned size in bytes for a given alignment
*/ 
// ---------------------------------------------------------------------
#define CUDA_ALIGN_BYTES(size, alignment) \
	( ( ((size) % (alignment)) == 0 ) ? (size) : ((size) + (alignment) - ((size) % (alignment))) )


// ---------------------------------------------------------------------
/*
	Return the aligned element count for a given alignment
*/ 
// ---------------------------------------------------------------------
#define CUDA_ALIGN_EX(count, alignment) \
	( ( ((count) % (alignment)) == 0 ) ? (count) : ((count) + (alignment) - ((count) % (alignment))) )


//////////////////////////////////////////////////////////////////////////

extern "C++"
template <class V, class S>
void kernel_wrapper_scale_vector_array(V *d_vec, uint32 count, S scalar); 
