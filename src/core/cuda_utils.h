#pragma once 

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <assert.h>
#include "utils.h"
#include "cuda_defs.h" 

#include "thrust/device_allocator.h"

/// Performs a CUDA synchronization and checks for returned errors (optional in release mode).
#ifdef _DEBUG
#define CUDA_CHECKERROR cuda_safe_call_no_sync(cuda_check_error(true))
#else
#define CUDA_CHECKERROR cuda_safe_call_no_sync(cuda_check_error(false))
#endif

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define cuda_safe_call_no_sync(err)	_cuda_safe_call_no_sync(err, __FILE__, __LINE__)

enum e_cuda_op
{
	/// Addition.
	cuda_op_add,
	/// Subtraction.
	cuda_op_sub,
	/// Multiplication.
	cuda_op_mul,
	/// Division.
	cuda_op_div,
	/// Minimum.
	cuda_op_min,
	/// Maximum.
	cuda_op_max,
}; 
 
inline void _cuda_safe_call_no_sync(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		yart_log_message("%s(%i) : CUDA Runtime API error : %s %d. \n", file, line, cudaGetErrorString(err), (int)err);
		assert(false); 
	}
}

cudaError_t cuda_check_error(bool bForce = true);


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


// ---------------------------------------------------------------------
/*
/// \brief	Computes the aligned \em element count for an alignment of 16 \em elements. 
///
///			This is required to gain profit from coalesced access. 
*/ 
// ---------------------------------------------------------------------
#define CUDA_ALIGN(count) CUDA_ALIGN_EX(count, 16)

// ---------------------------------------------------------------------
/*
/// \brief	Avoids the maximum CUDA grid size by using two grid dimensions for a one dimensional
/// 		grid. I added this due to problems with exceeding 1D grid sizes. 
///
/// \todo	Evaluate impact of spawning many more threads, e.g. when we got only maxBlocks+1
///			threads. In this case, the second slice of blocks would also get maxBlocks blocks. 
*/ 
// ---------------------------------------------------------------------
#define CUDA_MAKEGRID2D(numBlocks, maxBlocks) dim3(min((numBlocks), (maxBlocks)), 1 + (numBlocks) / (maxBlocks), 1)

// ---------------------------------------------------------------------
/*
/// \brief	Calculates the one dimensional block index for the given 2D grid that was created by
/// 		MNCUDA_MAKEGRID2D(). 
*/ 
// ---------------------------------------------------------------------
#define CUDA_GRID2DINDEX  (blockIdx.x + (blockIdx.y*gridDim.x)) 

//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
/// \brief	Generates compact addresses using CUDPP's compact.
/// 		
/// 		These can be used to compact data corresponding to \a d_isValid without using CUDPP's
/// 		compact, but using ::mncudaCompactInplace(). To compact a structure of arrays, you'd
/// 		have to call this once and ::mncudaCompactInplace() for each array. In my tests I
/// 		observed that this is much more efficient than multiple \c cudppCompact calls.
///
/// \note	This corresponds to compacting an identity array. 
*/ 
// ---------------------------------------------------------------------
uint32 cuda_gen_compact_addresses(uint32 *d_is_valid, uint32  old_count, uint32 *d_out_src_addr);


// ---------------------------------------------------------------------
/*
/// \brief	Compacts the data array "inplace" using a temporary buffer, the given source 
///			addresses and count. 
///
///			Enables to compact a structure of arrays using only one real compact 
///			and multiple set from addresses. 
*/ 
// ---------------------------------------------------------------------
template <typename T>
void cuda_compact_in_place(T *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count);


// ---------------------------------------------------------------------
/*
/// \brief	Performs reduction on \a d_data.
*/ 
// ---------------------------------------------------------------------
template <typename T>
void cuda_reduce_add(T& result, T *d_data, size_t count, T identity);

// ---------------------------------------------------------------------
/*
/// \brief	Performs segmented reduction on \a d_data.
/// 		
/// 		Segments are defined by \a d_owner, where \a d_owner[i] contains the segment of \a
/// 		d_data[i]. The result is put into \a d_result. This array has to be preallocated and
/// 		should have space for all segment results. 
*/ 
// ---------------------------------------------------------------------
template <typename T> 
void cuda_segmented_reduce_add(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments);
template <typename T> 
void cuda_segmented_reduce_min(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments);
template <typename T> 
void cuda_segmented_reduce_max(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments);

template <typename T> 
void cuda_scan(T *d_data, size_t num_elems, bool is_inclusive, T *d_out);

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


template <typename T> 
void cuda_set_from_address(T *d_array, uint32 *d_src_addr, T *d_vals, uint32 count_target); 

// ---------------------------------------------------------------------
/*
/// \brief	Initializes the given buffer with the identity relation, that is buffer[i] = i. 
///
///			Each component is handled by it's own CUDA thread.
///
*/ 
// ---------------------------------------------------------------------
void cuda_init_identity(uint32 *d_buffer, uint32 count);

//////////////////////////////////////////////////////////////////////////

extern "C++"
template <class V, class S>
void kernel_wrapper_scale_vector_array(V *d_vec, uint32 count, S scalar); 


// ---------------------------------------------------------------------
/*
/// \brief	Performs constant operation on all array elements:
///
///			\code d_array[i] = d_array[i] op constant \endcode
///
///			Each spawned thread works on one array component. 
*/ 
// ---------------------------------------------------------------------

template <typename T>
void cuda_constant_add(T* d_array, uint32 count, T constant);
template <typename T>
void cuda_constant_sub(T* d_array, uint32 count, T constant);
template <typename T>
void cuda_constant_mul(T* d_array, uint32 count, T constant);