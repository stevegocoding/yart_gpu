#pragma once

#include <vector>

#include "prerequisites.h"
#include "thrust/device_ptr.h"
#include "cuda_defs.h"

struct c_ray_chunk
{
public:
	
	explicit c_ray_chunk(uint32 _max_rays)
		: depth(0)
		, num_rays(0)
		, max_rays(_max_rays)
	{
		
	}

	void alloc_device_memory(); 
	void free_device_memory();

	thrust::device_ptr<float4> d_origins_array; 
	thrust::device_ptr<float4> d_dirs_array;
	thrust::device_ptr<uint32> d_pixels_array;
	
	uint32 depth; 
	uint32 num_rays; 
	uint32 max_rays;
};


//////////////////////////////////////////////////////////////////////////

class c_ray_pool
{
public:
	
	c_ray_pool();
	
	~c_ray_pool();

	void gen_primary_rays(const c_camera& cam);

private:

	c_ray_chunk *find_alloc_chunk();
	c_ray_chunk *alloc_chunk(uint32 max_rays);
	
	// Number of samples per pixel
	uint32 m_spp_x, m_spp_y;

	// Max number of rays per chunk
	uint32 m_max_rays_per_chunk; 

	// Memory chunks 
	std::vector<c_ray_chunk*> m_ray_chunks; 

	// Chunk that was used for the current task or NULL, if no current task.
	c_ray_chunk *m_chunk_task; 
};