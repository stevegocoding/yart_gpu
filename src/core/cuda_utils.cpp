#include <iostream>
#include <iomanip>
#include <sstream>
#include "cuda_utils.h"
#include "cuda_primitives.h"
#include "cuda_mem_pool.h"

using namespace std;

bool f_bEnableErrorChecks = false;

// ---------------------------------------------------------------------
/*
	Declarations
*/ 
// ---------------------------------------------------------------------

extern "C"
void kernel_wrapper_init_identity(uint32 *d_buffer, uint32 count);

extern "C++"
template <typename T>
void kernel_wrapper_set_from_address(T *d_array, uint32 *d_src_addr, T *d_vals, uint32 count_target);

// ---------------------------------------------------------------------
/*
	Constant OP
*/ 
// ---------------------------------------------------------------------
extern "C++"
template <typename T>
void device_constant_add(T *d_array, uint32 count, T constant); 
extern "C++"
template <typename T>
void device_constant_sub(T *d_array, uint32 count, T constant); 
extern "C++"
template <typename T>
void device_constant_mul(T *d_array, uint32 count, T constant); 

// ---------------------------------------------------------------------
/*
	Compact
*/ 
// ---------------------------------------------------------------------
extern "C++"
template <typename T> 
void device_compact(T *d_in, unsigned *d_stencil, size_t count, T *d_out_campacted, uint32 *d_out_new_count);

// ---------------------------------------------------------------------
/*
	Reduction
*/ 
// ---------------------------------------------------------------------
extern "C++"
template <typename T>
void device_reduce_add(T& result, T *d_in, size_t count, T init);

extern "C++"
template <typename T>
void device_segmented_reduce_min(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments);

extern "C++"
template <typename T>
void device_segmented_reduce_max(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments);

//////////////////////////////////////////////////////////////////////////

cudaError_t cuda_check_error(bool bforce /* = true */)
{
	if (bforce || f_bEnableErrorChecks)
		return cudaDeviceSynchronize();
	else 
		return cudaSuccess; 
} 

//////////////////////////////////////////////////////////////////////////


template <typename T>
void cuda_constant_add(T* d_array, uint32 count, T constant)
{
	assert(d_array && count > 0);
	device_constant_add<T>(d_array, count, constant);
}

template <typename T>
void cuda_constant_sub(T* d_array, uint32 count, T constant)
{
	assert(d_array && count > 0);
	device_constant_sub<T>(d_array, count, constant);
}

template <typename T>
void cuda_constant_mul(T* d_array, uint32 count, T constant)
{
	assert(d_array && count > 0);
	device_constant_mul<T>(d_array, count, constant);
}

template <typename T>
void cuda_compact(T *d_in, unsigned *d_stencil, size_t count, T *d_out_compacted, uint32 *d_out_new_count)
{
	assert(d_in && d_stencil && count > 0 && d_out_compacted && d_out_new_count);
	device_compact<T>(d_in, d_stencil, count, d_out_compacted, d_out_new_count);
}

template <typename T> 
void cuda_set_from_address(T *d_array, uint32 *d_src_addr, T *d_vals, uint32 count_target)
{
	assert(d_array && d_src_addr && d_vals && count_target > 0); 
	kernel_wrapper_set_from_address(d_array, d_src_addr, d_vals, count_target);
}

template <typename T>
void cuda_compact_in_place(T *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count)
{
	if (new_count == 0)
		return;
	
	// Move data into temp buffer.
	c_cuda_memory<T> d_temp_buf(old_count); 
	cuda_safe_call_no_sync(cudaMemcpy(d_temp_buf.get_writable_buf_ptr(), d_data, old_count*sizeof(T), cudaMemcpyDeviceToDevice));

	cuda_set_from_address(d_data, d_src_addr, (T*)d_temp_buf.get_buf_ptr(), new_count); 
}

template <typename T> 
void cuda_reduce_add(T& result, T *d_data, size_t count, T identity)
{
	// if (op == e_cuda_op)
	assert(d_data && count > 0);	
	device_reduce_add(result, d_data, count, identity); 	
}

template <typename T> 
void cuda_segmented_reduce_add(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments)
{
	assert(d_data && d_owner && d_result && count > 0 && num_segments > 0);
	
}

template <typename T> 
void cuda_segmented_reduce_min(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments)
{
	assert(d_data && d_owner && d_result && count > 0 && num_segments > 0);
	device_segmented_reduce_min<T>(d_data, d_owner, count, identity, d_result, num_segments);
}

template <typename T>
void cuda_segmented_reduce_max(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments)
{
	assert(d_data && d_owner && d_result && count > 0 && num_segments > 0);
	device_segmented_reduce_max<T>(d_data, d_owner, count, identity, d_result, num_segments);
}

uint32 cuda_gen_compact_addresses(uint32 *d_is_valid, uint32 old_count, uint32 *d_out_src_addr)
{
	c_cuda_memory<uint32> d_new_count(1);
	c_cuda_memory<uint32> d_identity(old_count);

	// Compact indices array
	cuda_init_identity(d_identity.get_writable_buf_ptr(), old_count);
	device_compact(d_identity.get_buf_ptr(), d_is_valid, old_count, d_out_src_addr, d_new_count.get_writable_buf_ptr());

	uint32 new_count = d_new_count.read(0);
	
	return new_count;
}

void cuda_init_identity(uint32 *d_buffer, uint32 count)
{
	assert(d_buffer && count > 0);
	kernel_wrapper_init_identity(d_buffer, count); 
}

//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
// Avoid linker errors by explicitly defining the used templates here. See
// http://www.parashift.com/c++-faq-lite/templates.html#faq-35.
*/ 
// ---------------------------------------------------------------------

template void cuda_constant_add<float>(float *d_array, uint32 count, float constant);
template void cuda_constant_sub<float>(float *d_array, uint32 count, float constant);
template void cuda_constant_mul<float>(float *d_array, uint32 count, float constant);
template void cuda_constant_add<uint32>(uint32 *d_array, uint32 count, uint32 constant);
template void cuda_constant_sub<uint32>(uint32 *d_array, uint32 count, uint32 constant);
template void cuda_constant_mul<uint32>(uint32 *d_array, uint32 count, uint32 constant); 

template void cuda_set_from_address<uint32>(uint32 *d_array, uint32 *d_src_addr, uint32 *d_vals, uint32 count_target);
template void cuda_set_from_address<float>(float *d_array, uint32 *d_src_addr, float *d_vals, uint32 count_target);
template void cuda_set_from_address<float2>(float2 *d_array, uint32 *d_src_addr, float2 *d_vals, uint32 count_target);
template void cuda_set_from_address<float4>(float4 *d_array, uint32 *d_src_addr, float4 *d_vals, uint32 count_target);

template void cuda_compact_in_place<uint32>(uint32 *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count);
template void cuda_compact_in_place<float>(float *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count);
template void cuda_compact_in_place<float2>(float2 *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count);
template void cuda_compact_in_place<float4>(float4 *d_data, uint32 *d_src_addr, uint32 old_count, uint32 new_count);

template void cuda_compact<uint32>(uint32 *d_in, unsigned *d_stencil, size_t count, uint32 *d_out_compacted, uint32 *d_out_new_count);

template void cuda_reduce_add<uint32>(uint32& result, uint32 *d_data, size_t count, uint32 init);

template void cuda_segmented_reduce_min<float>(float *d_data, uint32 *d_owner, uint32 count, float identity, float *d_result, uint32 num_segments);
template void cuda_segmented_reduce_min<float4>(float4 *d_data, uint32 *d_owner, uint32 count, float4 identity, float4 *d_result, uint32 num_segments);
template void cuda_segmented_reduce_min<uint32>(uint32 *d_data, uint32 *d_owner, uint32 count, uint32 identity, uint32 *d_result, uint32 num_segments);

template void cuda_segmented_reduce_max<float>(float *d_data, uint32 *d_owner, uint32 count, float identity, float *d_result, uint32 num_segments);
template void cuda_segmented_reduce_max<float4>(float4 *d_data, uint32 *d_owner, uint32 count, float4 identity, float4 *d_result, uint32 num_segments);
template void cuda_segmented_reduce_max<uint32>(uint32 *d_data, uint32 *d_owner, uint32 count, uint32 identity, uint32 *d_result, uint32 num_segments);

template void cuda_segmented_reduce_add<float>(float *d_data, uint32 *d_owner, uint32 count, float identity, float *d_result, uint32 num_segments);
template void cuda_segmented_reduce_add<float4>(float4 *d_data, uint32 *d_owner, uint32 count, float4 identity, float4 *d_result, uint32 num_segments);
template void cuda_segmented_reduce_add<uint32>(uint32 *d_data, uint32 *d_owner, uint32 count, uint32 identity, uint32 *d_result, uint32 num_segments);