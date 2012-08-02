#pragma once

#include <vector>
#include <assert.h>

#include "prerequisites.h"
#include "cuda_defs.h"

struct c_ray_chunk
{
#ifdef __cplusplus
	explicit c_ray_chunk(uint32 _max_rays); 
	
	void alloc_device_memory(); 
	void free_device_memory();

	// ---------------------------------------------------------------------
	/*
	/// \brief	Compacts this ray chunk using the given source address array.
	/// 		
	/// 		This operation assumes that the source addresses were generated before, e.g. using ::
	/// 		mncudaGenCompactAddresses(). The latter also returns the required new number of rays.
	/// 		Basically, this was done to allow compacting multiple structures using the same
	/// 		source addresses. 	
	*/ 
	// ---------------------------------------------------------------------
	void compact_src_addr(uint32 *d_src_addr, uint32 new_count);


#endif  
	float4 *d_origins; 
	float4 *d_dirs;
	uint32 *d_pixels;
	float4 *d_weights; 

	uint32 depth; 
	uint32 num_rays; 
	uint32 max_rays;
};

//////////////////////////////////////////////////////////////////////////

class c_ray_pool
{
public:
	
	c_ray_pool(uint32 max_rays, uint32 spp_x, uint32 spp_y);
	
	~c_ray_pool();

	void gen_primary_rays(const c_camera *cam);

	/// Returns true if there is an active chunk that is currently processed by the caller.
	bool is_chunk_active() { return m_chunk_task != NULL; }

	bool has_more_rays(); 
	c_ray_chunk* get_next_chunk(); 
	void finalize_chunk(c_ray_chunk *chunk);
	uint32 get_spp() const { return m_spp_x * m_spp_y; }

	c_ray_chunk* get_chunk(size_t idx) 
	{ 
		assert(idx < m_ray_chunks.size()); 
		return m_ray_chunks[idx]; 
	}
	
	size_t get_num_chuks() const {return m_ray_chunks.size();}
	
private:

	c_ray_chunk* find_alloc_chunk();
	c_ray_chunk* alloc_chunk(uint32 max_rays);
	c_ray_chunk* find_processing_chunk(); 
	
	void destroy(); 
	
	// Number of samples per pixel
	uint32 m_spp_x, m_spp_y;

	// Max number of rays per chunk
	uint32 m_max_rays_per_chunk; 

	// Memory chunks 
	std::vector<c_ray_chunk*> m_ray_chunks; 

	// Chunk that was used for the current task or NULL, if no current task.
	c_ray_chunk *m_chunk_task; 
}; 
