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

// ---------------------------------------------------------------------
/*
	Divides \a count by \a chunkSize and adds 1 if there is some remainder.
*/ 
// ---------------------------------------------------------------------
inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
	if(cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);        
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
