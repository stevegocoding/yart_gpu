#include <fstream>

#include "obj_makers.h"
#include "cuda_mem_pool.h"
#include "camera.h"
#include "ray_tracer.h"
#include "utils.h"
#include "cuda_rng.h"
#include "test_mt.h"

c_randoms_chunk *g_rands = NULL; 
uint32 g_size = 64; 
boost::shared_array<float> h_rands;

static const std::string mt_dat_file = "../data/mt/MersenneTwister.dat";

void c_randoms_chunk::alloc_device_memory()
{
	c_cuda_mem_pool& pool = c_cuda_mem_pool::get_instance();
	size_t align = 16;
	pool.request((void**)&d_array, num_elems*sizeof(float), "ray_pool", align); 
}

void c_randoms_chunk::free_device_memory()
{
	c_cuda_mem_pool& pool = c_cuda_mem_pool::get_instance();
	pool.release(d_array); 
}

void initialise()
{
	open_console_wnd();
	
	// Initialise the memory pool 
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	mem_pool.initialise(256*1024*1024, 256*1024);

	// Device memory 
	g_rands = new c_randoms_chunk(g_size); 
	g_rands->alloc_device_memory(); 

	// Host memory
	h_rands.reset(new float[g_size]);

	// Init CUDA MT
	srand(1337); 
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	bool ret = rng.init(mt_dat_file); 
	assert(ret); 
	ret = rng.seed(1337);
	assert(ret); 
}

void cleanup()
{
	g_rands->free_device_memory();
	SAFE_DELETE(g_rands);
}

void print_rands()
{
	std::ofstream ofs("debug_output.txt"); 

	cudaMemcpy(h_rands.get(), g_rands->d_array, sizeof(float)*g_size, cudaMemcpyDeviceToHost);
	
	for (uint32 i = 0; i < g_size; ++i)
	{
		ofs << h_rands[i] << std::endl;
	}
	
	ofs.close();
}

int main(int argc, char **argv)
{ 
	initialise(); 

	kernel_wrapper_test_mt(*g_rands);
	
	print_rands(); 

	cleanup();

	return 0; 
}