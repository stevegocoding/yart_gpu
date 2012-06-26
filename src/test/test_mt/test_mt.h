#ifndef __test_mt_h__
#define __test_mt_h__

#pragma once 

#include "cuda_defs.h"
#include "cuda_utils.h"

struct c_randoms_chunk 
{
	c_randoms_chunk(uint32 n)
		: num_elems(n)
	{}
	
	void alloc_device_memory();
	void free_device_memory();
	
	float *d_array;
	uint32 num_elems; 
};

extern "C"
void kernel_wrapper_test_mt(c_randoms_chunk& out_chunk);

#endif // __test_mt_h__
