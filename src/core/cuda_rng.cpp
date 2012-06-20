#include "cuda_rng.h"
#include "cuda_mt/MersenneTwister.h"
#include "cuda_utils.h"

extern "C"
bool MersenneTwisterGPUInit(const char *fname); 

extern "C"
void MersenneTwisterGPUSeed(unsigned int seed);

extern "C"
void MersenneTwisterGPU(float* d_outRand, int nPerRNG);

static const std::string fn = "MersenneTwister.dat";

c_cuda_rng::c_cuda_rng()
	: m_is_inited(false)
{
	bool res = init(fn); 
	assert(res); 
}

bool c_cuda_rng::init(const std::string& file_name)
{
	bool res = MersenneTwisterGPUInit(file_name.c_str());
	assert(res); 
	
	m_is_inited = true;

	return res;
}

bool c_cuda_rng::seed(unsigned int seed)
{
	if (!m_is_inited)
		return false; 

	MersenneTwisterGPUSeed(seed); 
	
	return true; 
}

cudaError_t c_cuda_rng::gen_rand(PARAM_OUT float *d_rand, int count)
{
	if (!m_is_inited)
		return cudaErrorUnknown;

	if (count != get_aligned_cnt(count))
		return cudaErrorInvalidValue; 

	int num_per_rng = count / MT_RNG_COUNT;

	MersenneTwisterGPU(d_rand, num_per_rng); 

	return cudaSuccess; 
}

unsigned int c_cuda_rng::get_aligned_cnt(unsigned int count)
{
	// Taken from SDK 3.0 sample.
	unsigned int numPerRNG = CUDA_DIVUP(count, MT_RNG_COUNT);
	unsigned int numAligned = CUDA_ALIGN_EX(numPerRNG, 2);

	return numAligned * MT_RNG_COUNT; 	
}


c_cuda_rng& get_rng_instance()
{
	static c_cuda_rng cuda_rng; 

	return cuda_rng; 
}
