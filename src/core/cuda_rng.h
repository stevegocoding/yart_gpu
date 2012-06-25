#ifndef __cuda_mt_h__
#define __cuda_mt_h__

#pragma once

#include <string>
#include <assert.h>
#include <cuda_runtime.h>
#include "prerequisites.h"

class c_cuda_rng
{
public:
	c_cuda_rng(); 

	bool init(const std::string& file_name); 

	bool seed(unsigned int seed);
	
	cudaError_t gen_rand(PARAM_OUT float *d_rand, int count); 
	
	unsigned int get_aligned_cnt(unsigned int count);
	 
	static c_cuda_rng& get_instance();
	
private:
	bool m_is_inited;
	
};



#endif // __cuda_mt_h__
