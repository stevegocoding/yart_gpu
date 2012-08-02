#include "ray_tracer.h"
#include "camera.h"
#include "render_target.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

extern "C"
void ivk_krnl_gen_primary_rays(const c_perspective_camera *camera, 
	uint32 sample_idx_x, 
	uint32 num_samples_x, 
	uint32 sample_idx_y, 
	uint32 num_samples_y,
	PARAM_OUT c_ray_chunk& out_chunk); 

//////////////////////////////////////////////////////////////////////////

c_ray_chunk::c_ray_chunk(uint32 _max_rays)
	: depth(0)
	, num_rays(0)
	, max_rays(_max_rays)
{
}

void c_ray_chunk::alloc_device_memory()
{
	// Allocate device memory 
	
	c_cuda_mem_pool& pool = c_cuda_mem_pool::get_instance();
	size_t align = 16;
	pool.request((void**)&d_origins, max_rays*sizeof(float4), "ray_pool", align);
	pool.request((void**)&d_dirs, max_rays*sizeof(float4), "ray_pool", align); 
	pool.request((void**)&d_pixels, max_rays*sizeof(uint32), "ray_pool"); 
	pool.request((void**)&d_weights, max_rays*sizeof(float4), "ray_pool",align); 
	
	/*
	cuda_safe_call_no_sync(cudaMalloc((void**)&d_origins_array, max_rays*sizeof(float4)));
	cuda_safe_call_no_sync(cudaMalloc((void**)&d_dirs_array, max_rays*sizeof(float4)));
	cuda_safe_call_no_sync(cudaMalloc((void**)&d_pixels_array, max_rays*sizeof(uint32)));
	cuda_safe_call_no_sync(cudaMalloc((void**)&d_weights_array, max_rays*sizeof(float4))); 
	*/
}

void c_ray_chunk::free_device_memory()
{	
	// Release device memory 
	
	c_cuda_mem_pool& pool = c_cuda_mem_pool::get_instance();
	pool.release(d_origins); 
	pool.release(d_dirs); 
	pool.release(d_pixels);
	pool.release(d_weights); 
	
	/*
	cuda_safe_call_no_sync(cudaFree(d_origins_array)); 
	cuda_safe_call_no_sync(cudaFree(d_dirs_array)); 
	cuda_safe_call_no_sync(cudaFree(d_pixels_array)); 
	cuda_safe_call_no_sync(cudaFree(d_weights_array)); 
	*/
}

void c_ray_chunk::compact_src_addr(uint32 *d_src_addr, uint32 new_count)
{
	if (new_count == 0)
	{
		num_rays = 0; 
		return;
	}

	// Move source data to destination data inplace.
	/*
	cuda_compact_in_place(d_origins_array, d_src_addr, num_rays, new_count);
	cuda_compact_in_place(d_dirs_array, d_src_addr, num_rays, new_count);
	cuda_compact_in_place(d_pixels_array, d_src_addr, num_rays, new_count);
	cuda_compact_in_place(d_weights_array, d_src_addr, num_rays, new_count);
	*/ 
	num_rays = new_count;
}

//////////////////////////////////////////////////////////////////////////

c_ray_pool::c_ray_pool(uint32 max_rays, uint32 spp_x, uint32 spp_y)
	: m_spp_x(spp_x)
	, m_spp_y(spp_y)
	, m_max_rays_per_chunk(max_rays)
	, m_chunk_task(NULL)
{
	
}

c_ray_pool::~c_ray_pool()
{
	destroy(); 
}

void c_ray_pool::gen_primary_rays(const c_camera *cam)
{
	uint32 res_x = cam->res_x(); 
	uint32 res_y = cam->res_y();

	// Ensure that all rays for one sample can fit into a single ray chunk.
	// NOTE: This is required for adaptive sample seeding as we would get notable separators in the
	//       final image when subdividing the screen's pixels into sections. The reason is that
	//       the clustering results don't fit together and pixels on both sides of the separator
	//       are calculated using different interpolation points.
	//       Therefore I'm abandoning the choise of putting all samples for a given pixel into the
	//		 same ray chunk to improve cache performance for primary rays for the sake of using
	//		 multisampling in combination with adaptive sample seeding.
	assert(res_x * res_y <= m_max_rays_per_chunk);
	
	for (uint32 x = 0; x < m_spp_x; ++x)
	{
		for (uint32 y = 0; y < m_spp_y; ++y)
		{
			c_ray_chunk *ray_chunk = find_alloc_chunk();
			ivk_krnl_gen_primary_rays((c_perspective_camera*)cam, x, m_spp_x, y, m_spp_y, *ray_chunk); 
		}
	}
}

c_ray_chunk* c_ray_pool::get_next_chunk()
{
	assert(!is_chunk_active()); 

	// Get best chunk
	c_ray_chunk *chunk = find_processing_chunk(); 
	if (!chunk)
		return NULL; 
	
	assert(chunk->num_rays > 0); 

	// Store the chunk to remember where the rays are in host memory. This
	// is used later when we generate child rays.
	m_chunk_task = chunk;
	
	return m_chunk_task; 
}

void c_ray_pool::finalize_chunk(c_ray_chunk *chunk)
{
	assert(!is_chunk_active() && chunk != m_chunk_task);

	// Chunk done, reset indices
	m_chunk_task->depth = 0; 
	m_chunk_task->num_rays = 0; 

	m_chunk_task = NULL; 
}

c_ray_chunk* c_ray_pool::find_alloc_chunk()
{
	for (size_t i = 0; i < m_ray_chunks.size(); ++i)
	{
		c_ray_chunk *chunk = m_ray_chunks[i]; 
		if (chunk != m_chunk_task && chunk->num_rays == 0)
			return chunk; 
	}

	assert(m_ray_chunks.size() <= 64);
	
	c_ray_chunk *chunk = alloc_chunk(m_max_rays_per_chunk); 
	return chunk; 
}

c_ray_chunk* c_ray_pool::alloc_chunk(uint32 max_rays)
{
	c_ray_chunk *chunk = new c_ray_chunk(max_rays); 
	chunk->alloc_device_memory();
	m_ray_chunks.push_back(chunk);

	return chunk;
}

c_ray_chunk* c_ray_pool::find_processing_chunk()
{
	// Use a chunk with the highest possible recursion depth. This should avoid
	// allocating to many chunks since it avoids following too many paths in the
	// recursion tree.

	c_ray_chunk *best_chunk = NULL; 
	int max_depth = -1; 
	for (size_t i = 0; i < m_ray_chunks.size(); ++i)
	{
		c_ray_chunk *chunk = m_ray_chunks[i]; 
		int depth = (int)chunk->depth; 
		if (depth > max_depth && chunk->num_rays > 0)
		{
			max_depth = depth;
			best_chunk = chunk; 
		}
	}

	return best_chunk;  
}

void c_ray_pool::destroy()
{
	assert(!is_chunk_active()); 
	
	m_chunk_task = NULL; 
	for (size_t i = 0; i < m_ray_chunks.size(); ++i)
	{
		c_ray_chunk *chunk = m_ray_chunks[i];
		chunk->free_device_memory();
		SAFE_DELETE(chunk);
	}
	m_ray_chunks.clear();
}

bool c_ray_pool::has_more_rays()
{
	return (find_processing_chunk() != NULL); 
}



//////////////////////////////////////////////////////////////////////////

