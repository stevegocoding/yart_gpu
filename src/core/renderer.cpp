#include "renderer.h"
#include "ray_tracer.h"
#include "camera.h"
#include "cuda_rng.h"
#include "cuda_utils.h"
#include "cuda_mem_pool.h"
#include "cuda_primitives.h"

static const std::string mt_dat_file = "MersenneTwister.dat";

// ---------------------------------------------------------------------
/*
	Kernel Functions
*/ 
// ---------------------------------------------------------------------
extern "C"
void launch_kernel_trace_rays(const c_ray_chunk& ray_chunk, PARAM_OUT c_shading_points_array& out_isects, uint32 *d_out_is_valid);


//////////////////////////////////////////////////////////////////////////

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
		trace_rays(next_chunk, &m_sp_shading);

		if (m_sp_shading.num_pts == 0)
		{ 
			m_ray_pool->finalize_chunk(next_chunk); 
			continue;
		}
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
	m_sp_shading.alloc_mem(screen_size*screen_size);
	
	//

	yart_log_message("Core initialized.");
}

void c_renderer::destroy()
{ 
	
}

uint32 c_renderer::trace_rays(c_ray_chunk *ray_chunk, c_shading_points_array *sp_shading, PARAM_OUT uint32 *d_out_src_addr)
{ 
	c_cuda_memory<uint32> d_is_valid(ray_chunk->num_rays);
	
	// Perform Tracing 
	launch_kernel_trace_rays(*ray_chunk, *sp_shading, d_is_valid.get_writable_buf_ptr());
	
	uint32 traced = ray_chunk->num_rays;

	// Compact shading points and ray chunk to avoid rays that hit nothing.
	if (d_out_src_addr)
	{
		uint32 new_count = cuda_gen_compact_addresses(d_is_valid.get_buf_ptr(), sp_shading->num_pts, d_out_src_addr);
		sp_shading->compact_src_addr(d_out_src_addr, new_count);
		ray_chunk->compact_src_addr(d_out_src_addr, new_count);
	}
	else 
	{
		c_cuda_memory<uint32> d_src_addr(sp_shading->num_pts);
		uint32 new_count = cuda_gen_compact_addresses(d_is_valid.get_buf_ptr(), sp_shading->num_pts, d_src_addr.get_writable_buf_ptr());
		sp_shading->compact_src_addr(d_src_addr.get_buf_ptr(), new_count);
		ray_chunk->compact_src_addr(d_src_addr.get_buf_ptr(), new_count);
	}
	
	/*
	// Create normals for shading points.
	if (sp_shading->num_pts != 0) 
	{
		
	}
	*/ 

	return traced;
	
}