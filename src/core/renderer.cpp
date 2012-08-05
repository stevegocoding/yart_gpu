#include "renderer.h"
#include "ray_tracer.h"
#include "camera.h"
#include "scene.h"
#include "cuda_rng.h"
#include "cuda_utils.h"
#include "cuda_mem_pool.h"
#include "cuda_primitives.h"
#include "kdtree_triangle.h"

static const std::string mt_dat_file = "../data/mt/MersenneTwister.dat";

// ---------------------------------------------------------------------
/*
	Kernel Functions
*/ 
// ---------------------------------------------------------------------
extern "C"
void kernel_wrapper_trace_rays(const c_ray_chunk& ray_chunk, 
							PARAM_OUT c_shading_points& out_isects, 
							uint32 *d_out_is_valid);

extern "C"
void kernel_wrapper_bsdf_get_normal_hit_pt(uint32 count, 
										int *d_tri_hit_idx, 
										float2 *d_hit_barycoords, 
										float4 *d_out_normals_geo, 
										float4 *d_out_normals_shading); 

extern "C"
void kernel_wrapper_solve_lte(const c_ray_chunk& ray_chunk,
							const c_shading_points& shading_pts, 
							const c_light_data& lights, 
							float4 *d_radiance_indirect, 
							bool is_direct_rt, 
							bool is_shadow_rays, 
							uint2 area_light_samples,
							float4 *d_io_radiance); 

extern "C"
void kernel_wrapper_img_float4_to_rgba(float4 *d_in_radiance, 
									uint32 num_pixels, 
									uchar4 *d_out_screen_buf);

extern "C"
void init_raytracing_kernels(); 

extern "C"
void cleanup_raytracing_kernels();

extern "C"
void update_raytracing_kernel_data(const c_light_data& lights, 
	const c_triangle_data& tris,
	const c_material_data& mats, 
	const c_kdtree_data& kdtree,
	float ray_epsilon);


//////////////////////////////////////////////////////////////////////////

c_renderer::c_renderer(scene_ptr scene)
	: m_scene(scene)
	, m_spp_x(1)
	, m_spp_y(1)
	, m_enable_direct_rt(true)
	, m_enable_shadow_rays(false)
	, m_area_light_samples_x(1)
	, m_area_light_samples_y(1)
	, m_is_dynamic_scene(false)
	, m_ray_epsilon(0.00005f)
{
}

c_renderer::~c_renderer()
{ 
	destroy(); 
}

bool c_renderer::render_scene(uchar4 *d_screen_buf)
{
	if (!rebuild_obj_kdtree())
	{
		yart_log_message("Building object kd-tree failed!"); 
		return false;
	}

	uint32 screen_w = m_scene->get_cam()->res_x();
	uint32 screen_h = m_scene->get_cam()->res_y(); 
	
	// Render the scene to a buffer
	c_cuda_memory<float4> d_radiance(screen_w*screen_h, "temp", 256);
	if (!render_to_buf(d_radiance.buf_ptr()))
		return false; 
		
	// Convert the radiance to the displayable RGB color.
	kernel_wrapper_img_float4_to_rgba(d_radiance.buf_ptr(), screen_w*screen_h, d_screen_buf); 
	
	return true; 
}

bool c_renderer::render_to_buf(PARAM_OUT float4 *d_radiance)
{
	assert(d_radiance);
	
	cudaError_t err = cudaSuccess;
	
	perspective_cam_ptr cam = m_scene->get_cam(); 
	uint32 res_x = cam->res_x();
	uint32 res_y = cam->res_y();

	// Set target buffer to black 
	cuda_safe_call_no_sync(cudaMemset(d_radiance, 0, res_x*res_y*sizeof(float4))); 
	
	// Generate primary rays for current camera position 
	m_ray_pool->gen_primary_rays(cam.get()); 

	uint2 area_light_samples = make_uint2(m_area_light_samples_x, m_area_light_samples_y);
	
	while(m_ray_pool->has_more_rays())
	{
		// Get more rays from ray pool. This performs synchronization.
		c_ray_chunk *ray_chunk = m_ray_pool->get_next_chunk();
		
		// Trace rays
		trace_rays(ray_chunk, &m_sp_shading);

		if (m_sp_shading.num_pts == 0)
		{ 
			m_ray_pool->finalize_chunk(ray_chunk); 
			continue;
		}

		c_cuda_memory<float4> d_radiance_indirect(m_sp_shading.num_pts, "temp", 256); 
		cuda_safe_call_no_sync(cudaMemset(d_radiance_indirect.buf_ptr(), 0, m_sp_shading.num_pts*sizeof(float4))); 


		kernel_wrapper_solve_lte(*ray_chunk, 
								m_sp_shading, 
								m_scene->get_light_data(), 
								d_radiance_indirect.buf_ptr(), 
								m_enable_direct_rt, 
								m_enable_shadow_rays, 
								area_light_samples, 
								d_radiance); 


		m_ray_pool->finalize_chunk(ray_chunk);
	}
	
	uint32 spp = m_ray_pool->get_spp();
	if (spp > 1)
		kernel_wrapper_scale_vector_array((float4*)d_radiance, res_x*res_y, 1.0f / spp);


	return true; 
}

void c_renderer::initialise()
{
	// Initialise the MT
	srand(1337); 
	c_cuda_rng& rng = c_cuda_rng::get_instance();

	bool ret = rng.init(mt_dat_file); 
	assert(ret); 
	
	ret = rng.seed(1337);
	assert(ret); 
	
	uint32 res_x = m_scene->get_cam()->res_x(); 
	uint32 res_y = m_scene->get_cam()->res_y();
	assert(res_x == res_y);
	
	// Shading points array
	m_sp_shading.initialise(res_x*res_x);
	
	// Ray pool
	m_ray_pool = make_ray_pool(256*1024, m_spp_x, m_spp_y); 

	init_raytracing_kernels(); 
}

void c_renderer::destroy()
{ 
	m_sp_shading.destroy();

	cleanup_raytracing_kernels(); 
}

bool c_renderer::rebuild_obj_kdtree()
{
	// No need to rebuild the tree for static scene 
	if (!m_is_dynamic_scene)
		return true; 

	if (m_kdtree_tri) 
		m_kdtree_tri.reset(); 
	
	m_kdtree_tri = make_kdtree_tri(m_scene->get_triangle_data()); 
	if (!m_kdtree_tri->build_tree())
		return 1; 

	update_raytracing_kernel_data(m_scene->get_light_data(), 
								m_scene->get_triangle_data(), 
								m_scene->get_material_data(),
								*(m_kdtree_tri->get_kdtree_data()), 
								m_ray_epsilon); 
	

	return true; 
}

uint32 c_renderer::trace_rays(c_ray_chunk *ray_chunk, c_shading_points *shading_pts, PARAM_OUT uint32 *d_out_src_addr)
{ 
	c_cuda_memory<uint32> d_is_valid(ray_chunk->num_rays);
	
	// Perform Tracing 
	kernel_wrapper_trace_rays(*ray_chunk, *shading_pts, d_is_valid.buf_ptr());
	uint32 traced = ray_chunk->num_rays;

	// Compact shading points and ray chunk to avoid rays that hit nothing.
	if (d_out_src_addr)
	{
		uint32 new_count = cuda_gen_compact_addresses(d_is_valid.buf_ptr(), shading_pts->num_pts, d_out_src_addr);
		shading_pts->compact_src_addr(d_out_src_addr, new_count);
		ray_chunk->compact_src_addr(d_out_src_addr, new_count);
	}
	else 
	{
		c_cuda_memory<uint32> d_src_addr(shading_pts->num_pts);
		uint32 new_count = cuda_gen_compact_addresses(d_is_valid.buf_ptr(), shading_pts->num_pts, d_src_addr.buf_ptr());
		shading_pts->compact_src_addr(d_src_addr.buf_ptr(), new_count);
		ray_chunk->compact_src_addr(d_src_addr.buf_ptr(), new_count);
	}
	
	// Create normals for shading points.
	if (shading_pts->num_pts != 0) 
	{
		kernel_wrapper_bsdf_get_normal_hit_pt(shading_pts->num_pts, 
											shading_pts->d_tri_indices, 
											shading_pts->d_isect_barycoords, 
											shading_pts->d_geo_normals, 
											shading_pts->d_shading_normals); 
	}
	
	return traced;
}