#include "thrust/device_malloc.h"
#include "ray_gen.h"
#include "camera.h"
#include "render_target.h"

using namespace thrust;

extern "C"
void ivk_krnl_gen_primary_rays(const c_perspective_camera *camera, 
	uint32 sample_idx_x, 
	uint32 num_samples_x, 
	uint32 sample_idx_y, 
	uint32 num_samples_y,
	c_ray_chunk *out_chunk); 

//////////////////////////////////////////////////////////////////////////

void c_ray_chunk::alloc_device_memory()
{
	// Allocate device memory 
	d_origins_array = device_malloc<float4>(sizeof(float4)*max_rays);
	d_dirs_array = device_malloc<float4>(sizeof(float4)*max_rays);
	d_pixels_array = device_malloc<uint32>(sizeof(uint32)*max_rays); 	
	
}

void c_ray_chunk::free_device_memory()
{
	thrust::device_free(d_origins_array);
	thrust::device_free(d_dirs_array);
	thrust::device_free(d_pixels_array);
}

//////////////////////////////////////////////////////////////////////////

void c_ray_pool::gen_primary_rays(const c_camera& cam)
{
	uint32 res_x = cam.get_render_target()->res_x(); 
	uint32 res_y = cam.get_render_target()->res_y();

	m_max_rays_per_chunk = res_x * res_y * m_spp_x * m_spp_y; 
	
	for (uint32 x = 0; x < m_spp_x; ++x)
	{
		for (uint32 y = 0; y < m_spp_y; ++y)
		{
			c_ray_chunk *ray_chunk = find_alloc_chunk();
			ivk_krnl_gen_primary_rays(&cam, x, m_spp_x, y, m_spp_y, ray_chunk); 
		}
	}
}

c_ray_chunk *c_ray_pool::find_alloc_chunk()
{
	/** 
		For now, all the sample rays all in a single chunk
	*/
	
	if (m_ray_chunks.size() > 0)
	{
		return m_ray_chunks[0];
	}
	
	c_ray_chunk *chunk = alloc_chunk(m_max_rays_per_chunk); 
	return chunk; 
}

c_ray_chunk *c_ray_pool::alloc_chunk(uint32 max_rays)
{
	c_ray_chunk *chunk = new c_ray_chunk(max_rays); 
	m_ray_chunks.push_back(chunk);

	return chunk;
}