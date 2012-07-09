#include "cuda_utils.h"
#include "cuda_mem_pool.h"

#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#include "thrust/transform.h"
#include "thrust/copy.h"
#include "thrust/gather.h"
#include "thrust/count.h"

template <typename T>
struct add_op
{
	add_op(T _val)
		: val(_val)
	{}
	
	__host__ __device__ 
	T operator() (T x) 
	{
		return x + val; 
	}

	T val;
};

struct is_valid
{
	__host__ __device__
	bool operator() (uint32 x)
	{
		return (x == 1);
	}
};

extern "C++"
template <typename T> 
void device_constant_add(T *d_array, uint32 count, T constant)
{
	thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array);
	thrust::transform(d_ptr, d_ptr+count, d_ptr, add_op<T>(constant));
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


template void device_constant_add<uint32>(uint32 *d_array, uint32 count, uint32 constant); 
template void device_compact<uint32>(uint32 *d_in, unsigned *d_stencil, size_t num_elems, uint32 *d_out_campacted, uint32 *d_out_new_count);