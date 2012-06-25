#include "renderer.h"
#include "ray_tracer.h"
#include "camera.h"
#include "cuda_rng.h"
#include "cuda_utils.h"

static const std::string mt_dat_file = "MersenneTwister.dat";

c_renderer::c_renderer(ray_pool_ptr ray_pool, perspective_cam_ptr camera)
	: m_ray_pool(ray_pool)
	, m_camera(camera) 
{
}

c_renderer::~c_renderer()
{ 
}

bool c_renderer::render_scene(uchar4 *d_screen_buf)
{
	return true; 
}

bool c_renderer::render_to_buf(PARAM_OUT float4 *d_radiance)
{
	cudaError_t err = cudaSuccess; 
	
	uint32 res_x = m_camera->res_x();
	uint32 res_y = m_camera->res_y();

	// Reset target buffer to black.
	err = cudaMemset(d_radiance, 0, res_x*res_y*sizeof(float4));
	assert(err == cudaSuccess); 

	m_ray_pool->gen_primary_rays(m_camera.get()); 
	
	while(m_ray_pool->has_more_rays())
	{
		// Get more rays from ray pool. This performs synchronization.
		c_ray_chunk *next_chunk = m_ray_pool->get_next_chunk();
		
		// Trace rays
		
		m_ray_pool->finalize_chunk(next_chunk); 
	}
	
	uint32 spp = m_ray_pool->get_spp();
	if (spp > 1)
		kernel_wrapper_scale_vector_array((float4*)d_radiance, res_x*res_y, 1.0f / spp);
 
	return true; 
}

void c_renderer::initialise(uint32 screen_size)
{
	// Initialise the MT
	srand(1337); 
	c_cuda_rng& rng = c_cuda_rng::get_instance();

	bool ret = rng.init(mt_dat_file); 
	assert(ret); 
	
	ret = rng.seed(1337);
	assert(ret); 
	
	// Shading points array
	m_shading_pts.alloc_mem(screen_size*screen_size);
	
	//
	
	
	
	yart_log_message("Core initialized.");
}

void c_renderer::destroy()
{ 
	
}