#include <fstream>

#include "cuda_utils.h"
#include "cuda_mem_pool.h" 

uint32 *d_array = NULL;
uint32 *h_array = NULL;
uint32 num_elems = 8;

extern "C++"
template <typename T> 
void device_constant_add(T *d_array, uint32 count, T constant);

extern "C++"
template <typename T> 
void device_compact(T *d_in, unsigned *d_stencil, size_t num_elems, T *d_out_campacted, uint32 *d_out_new_count);

uint32 gen_compact_addresses(uint32 *d_is_valid, uint32  old_count, uint32 *d_out_src_addr)
{
	c_cuda_memory<uint32> d_new_count(1);
	c_cuda_memory<uint32> d_identity(old_count);

	// Compact indices array
	cuda_init_identity(d_identity.get_writable_buf_ptr(), old_count);

	/*
	cp.compact(d_identity.get_buf_ptr(), d_is_valid, old_count, d_out_src_addr, d_new_count.get_writable_buf_ptr());

	uint32 new_count = d_new_count.read(0);
	
	return new_count;
	*/

	return 0; 
	
}

void initialize()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	mem_pool.initialise(256*1024*1024, 256*1024);
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_array, num_elems*sizeof(uint32)));
	
	// cuda_safe_call_no_sync(cudaMalloc((void**)&d_array, num_elems*sizeof(uint32)));
	h_array = new uint32[num_elems];
}

void destroy()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	cuda_safe_call_no_sync(mem_pool.release(d_array)); 
	
	// cuda_safe_call_no_sync(cudaFree(d_array));
	SAFE_DELETE_ARRAY(h_array);
}

void print_result(std::ostream& os)
{
	for (size_t i = 0; i < num_elems; ++i)
	{
		os << h_array[i] << "	";
	}
	
	os << std::endl;
}

int main(int argc, char **argv)
{ 
	initialize();
	
	std::ofstream ofs("output.txt");
	
	// Test init identity
	ofs << "Test cuda_init_identity" << std::endl; 
	cuda_init_identity(d_array, num_elems);
	cuda_safe_call_no_sync(cudaMemcpy(h_array, d_array, num_elems*sizeof(uint32), cudaMemcpyDeviceToHost));
	print_result(ofs);

	// Test constant op
	ofs << "Test constant add" << std::endl; 
	cuda_constant_mul<uint32>(d_array, num_elems, 10);
	cuda_safe_call_no_sync(cudaMemcpy(h_array, d_array, num_elems*sizeof(uint32), cudaMemcpyDeviceToHost));
	print_result(ofs); 

	// Test scan

	// Test reduce 
 
	// Test segmented reduce 
	
	// Test compact
	ofs << "Test compact" << std::endl;
	uint32 *stencil_array = new uint32[num_elems];
	memset(stencil_array,0,num_elems*sizeof(uint32));
	stencil_array[1] = 1; 
	stencil_array[3] = 1;
	stencil_array[4] = 1;
	stencil_array[6] = 1;
	c_cuda_memory<unsigned> d_stencil(num_elems);
	cuda_safe_call_no_sync(cudaMemcpy(d_stencil.get_writable_buf_ptr(), stencil_array, num_elems*sizeof(uint32), cudaMemcpyHostToDevice));
	c_cuda_memory<uint32> d_out_compacted(num_elems);
	c_cuda_memory<uint32> d_new_count(1); 
	cuda_safe_call_no_sync(cudaMemset(d_out_compacted.get_writable_buf_ptr(),0,num_elems*sizeof(uint32)));
	// device_compact<uint32>(d_array, d_stencil.get_buf_ptr(), num_elems, d_out_compacted.get_writable_buf_ptr(), d_new_count.get_writable_buf_ptr());
	uint32 new_count = cuda_gen_compact_addresses(d_stencil.get_buf_ptr(), num_elems, d_out_compacted.get_writable_buf_ptr());
	cuda_compact_in_place<uint32>(d_array, d_out_compacted.get_buf_ptr(), num_elems, new_count);
	cuda_safe_call_no_sync(cudaMemcpy(h_array, d_array, new_count*sizeof(uint32), cudaMemcpyDeviceToHost));
	print_result(ofs); 
	SAFE_DELETE_ARRAY(stencil_array);


	destroy();
	ofs.close();
	
	return 0; 
}