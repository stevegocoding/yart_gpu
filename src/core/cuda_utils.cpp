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


extern "C++"
template <e_cuda_op, typename T>
void kernel_wrapper_constant_op(T *d_array, uint32 count, T constant);

//////////////////////////////////////////////////////////////////////////

cudaError_t cuda_check_error(bool bforce /* = true */)
{
	if (bforce || f_bEnableErrorChecks)
		return cudaDeviceSynchronize();
	else 
		return cudaSuccess; 
}

//////////////////////////////////////////////////////////////////////////

template <e_cuda_op op, typename T>
void cuda_constant_op(T *d_array, uint32 count, T constant)
{
	assert(d_array && count > 0);
	kernel_wrapper_constant_op<op, T>(d_array, count, constant); 
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

uint32 cuda_gen_compact_addresses(uint32 *d_is_valid, uint32 old_count, uint32 *d_out_src_addr)
{
	c_cuda_primitives& cp = c_cuda_primitives::get_instance();
	c_cuda_memory<uint32> d_new_count(1);
	c_cuda_memory<uint32> d_identity(old_count);

	// Compact indices array
	cuda_init_identity(d_identity.get_writable_buf_ptr(), old_count);
	cp.compact(d_identity.get_buf_ptr(), d_is_valid, old_count, d_out_src_addr, d_new_count.get_writable_buf_ptr());

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

template void cuda_constant_op<cuda_op_add, float>(float* d_array, uint32 count, float constant);
template void cuda_constant_op<cuda_op_sub, float>(float* d_array, uint32 count, float constant);
template void cuda_constant_op<cuda_op_mul, float>(float* d_array, uint32 count, float constant);
template void cuda_constant_op<cuda_op_add, uint32>(uint32* d_array, uint32 count, uint32 constant);
template void cuda_constant_op<cuda_op_sub, uint32>(uint32* d_array, uint32 count, uint32 constant);
template void cuda_constant_op<cuda_op_mul, uint32>(uint32* d_array, uint32 count, uint32 constant);

