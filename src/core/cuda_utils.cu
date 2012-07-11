#include "cuda_utils.h"
#include "cuda_defs.h"
#include "cuda_mem_pool.h"
#include "cuda_utils_device.h"
#include "functor_device.h"

#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/transform.h"
#include "thrust/copy.h"
#include "thrust/reduce.h"
#include "thrust/gather.h"
#include "thrust/count.h"

 
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

extern "C++"
template <typename T> 
void device_constant_add(T *d_array, uint32 count, T constant)
{
	thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array); 
	thrust::transform(d_ptr, d_ptr+count, d_ptr, constant_add<T>(constant));
}
extern "C++"
template <typename T> 
void device_constant_sub(T *d_array, uint32 count, T constant)
{
	thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array); 
	thrust::transform(d_ptr, d_ptr+count, d_ptr, constant_sub<T>(constant));
}
extern "C++"
template <typename T> 
void device_constant_mul(T *d_array, uint32 count, T constant)
{
	thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array); 
	thrust::transform(d_ptr, d_ptr+count, d_ptr, constant_mul<T>(constant));
}

extern "C++"
template <typename T> 
void device_compact(T *d_in, unsigned *d_stencil, size_t num_elems, T *d_out_campacted, uint32 *d_out_new_count)
{	
	thrust::device_ptr<unsigned> d_stencil_ptr = thrust::device_pointer_cast(d_stencil);
	thrust::device_ptr<T> d_in_ptr = thrust::device_pointer_cast(d_in); 
	thrust::device_ptr<T> d_out_ptr = thrust::device_pointer_cast(d_out_campacted);

	uint32 new_count = thrust::copy_if(d_in_ptr, d_in_ptr+num_elems, d_stencil_ptr, d_out_ptr, is_valid()) - d_out_ptr;

	cuda_safe_call_no_sync(cudaMemcpy(d_out_new_count, &new_count, sizeof(uint32), cudaMemcpyHostToDevice)); 
}

extern "C++"
template <typename T>
void device_reduce_add(T& result, T *d_in, size_t count, T init)
{
	thrust::device_ptr<uint32> d_in_ptr = thrust::device_pointer_cast(d_in); 
	result = thrust::reduce(d_in_ptr, d_in_ptr+count, init, thrust::plus<T>());
}

extern "C++"
template <typename T>
void device_segmented_reduce_min(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments)
{
	c_cuda_memory<uint32> d_keys_output(count); 
	
	thrust::device_ptr<uint32> d_keys_ptr = thrust::device_pointer_cast<uint32>(d_owner); 
	thrust::device_ptr<T> d_data_ptr = thrust::device_pointer_cast<T>(d_data);
	thrust::device_ptr<T> d_result_ptr = thrust::device_pointer_cast<T>(d_result);
	thrust::device_ptr<uint32> d_keys_output_ptr = thrust::device_pointer_cast<uint32>(d_keys_output.get_writable_buf_ptr());  
	
	thrust::reduce_by_key(d_keys_ptr, d_keys_ptr+count, d_data_ptr, d_keys_output_ptr, d_result_ptr, thrust::equal_to<uint32>(), op_minimum<T>());
}

extern "C++"
template <typename T>
void device_segmented_reduce_max(T *d_data, uint32 *d_owner, uint32 count, T identity, T *d_result, uint32 num_segments)
{
	c_cuda_memory<uint32> d_keys_output(count); 

	thrust::device_ptr<uint32> d_keys_ptr = thrust::device_pointer_cast<uint32>(d_owner); 
	thrust::device_ptr<T> d_data_ptr = thrust::device_pointer_cast<T>(d_data);
	thrust::device_ptr<T> d_result_ptr = thrust::device_pointer_cast<T>(d_result);
	thrust::device_ptr<uint32> d_keys_output_ptr = thrust::device_pointer_cast<uint32>(d_keys_output.get_writable_buf_ptr());  

	thrust::reduce_by_key(d_keys_ptr, d_keys_ptr+count, d_data_ptr, d_keys_output_ptr, d_result_ptr, thrust::equal_to<uint32>(), op_maximum<T>());
}

template void device_constant_add<float>(float *d_array, uint32 count, float constant); 
template void device_constant_sub<float>(float *d_array, uint32 count, float constant); 
template void device_constant_mul<float>(float *d_array, uint32 count, float constant); 
template void device_constant_add<uint32>(uint32 *d_array, uint32 count, uint32 constant); 
template void device_constant_sub<uint32>(uint32 *d_array, uint32 count, uint32 constant); 
template void device_constant_mul<uint32>(uint32 *d_array, uint32 count, uint32 constant); 

template void device_compact<uint32>(uint32 *d_in, unsigned *d_stencil, size_t num_elems, uint32 *d_out_campacted, uint32 *d_out_new_count);

template void device_reduce_add<uint32>(uint32& result, uint32 *d_in, size_t count, uint32 init);

template void device_segmented_reduce_min<float>(float *d_data, uint32 *d_owner, uint32 count, float identity, float *d_result, uint32 num_segments);
template void device_segmented_reduce_min<float4>(float4 *d_data, uint32 *d_owner, uint32 count, float4 identity, float4 *d_result, uint32 num_segments);
template void device_segmented_reduce_min<uint32>(uint32 *d_data, uint32 *d_owner, uint32 count, uint32 identity, uint32 *d_result, uint32 num_segments);

/*
template void device_segmented_reduce_max<float>(float *d_data, uint32 *d_owner, uint32 count, float identity, float *d_result, uint32 num_segments);
template void device_segmented_reduce_max<float4>(float4 *d_data, uint32 *d_owner, uint32 count, float4 identity, float4 *d_result, uint32 num_segments);
template void device_segmented_reduce_max<uint32>(uint32 *d_data, uint32 *d_owner, uint32 count, uint32 identity, uint32 *d_result, uint32 num_segments);
*/

template void kernel_wrapper_set_from_address<uint32>(uint32 *d_array, uint32 *d_src_addr, uint32 *d_vals, uint32 count_target);
template void kernel_wrapper_set_from_address<float>(float *d_array, uint32 *d_src_addr, float *d_vals, uint32 count_target);
template void kernel_wrapper_set_from_address<float2>(float2 *d_array, uint32 *d_src_addr, float2 *d_vals, uint32 count_target);
template void kernel_wrapper_set_from_address<float4>(float4 *d_array, uint32 *d_src_addr, float4 *d_vals, uint32 count_target);
