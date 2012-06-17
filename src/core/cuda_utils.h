#pragma once 

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <assert.h>

#ifdef _DEBUG
	#define CUDA_CHECK_ERROR CUDA_SAFE_CALL_NO_SYNC(cuda_check_error(true))
#else
	#define CUDA_CHECK_ERROR CUDA_SAFE_CALL_NO_SYNC(cuda_check_error(false))
#endif

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define check_cuda_error(err)  __checkCudaErrors (err, __FILE__, __LINE__)



// ---------------------------------------------------------------------
/*
	Checks for pending errors and returns them.
*/ 
// ---------------------------------------------------------------------
cudaError_t cuda_check_error(bool bforce = true);


inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
	if(cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);        
	}
}

inline int cuda_div_up(int a, int b) 
{
	assert(b != 0);
	return (a / b) + (a % b) ? 1 : 0;  
}

// ---------------------------------------------------------------------
/*
	Return the aligned size in bytes for a given alignment
*/ 
// ---------------------------------------------------------------------
inline int cuda_align_bytes(int size, int alignment)
{
	return (size % alignment == 0) ? size : (size + alignment - (size % alignment));
}


// ---------------------------------------------------------------------
/*
	Return the aligned element count for a given alignment
*/ 
// ---------------------------------------------------------------------
inline int cuda_align_ex(int count, int alignment)
{
	return (count % alignment) ? count : (count + alignment - (count % alignment));
}