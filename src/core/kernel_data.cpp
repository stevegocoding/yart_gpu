#include "kernel_data.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

void c_shading_points_array::alloc_mem(uint32 _max_points)
{
	assert(_max_points > 0); 
	
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	max_pts = CUDA_ALIGN_EX(_max_points, mem_pool.get_allcated_size()/sizeof(float));
	num_pts = 0; 

	cuda_safe_call_no_sync(mem_pool.request((void**)&d_pixels_buf, max_pts*sizeof(uint32), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_tri_idices, max_pts*sizeof(int), "shading_pts"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_isect_pts, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_geo_normals, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_shading_normals, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_isect_bary_coords, max_pts*sizeof(float2), "shading_pts")); 
}

void c_shading_points_array::destroy()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	cuda_safe_call_no_sync(mem_pool.release(d_pixels_buf));
	cuda_safe_call_no_sync(mem_pool.release(d_tri_idices));
	cuda_safe_call_no_sync(mem_pool.release(d_isect_pts));
	cuda_safe_call_no_sync(mem_pool.release(d_geo_normals));
	cuda_safe_call_no_sync(mem_pool.release(d_shading_normals));
	cuda_safe_call_no_sync(mem_pool.release(d_isect_bary_coords));	
	
	max_pts = 0; 
	num_pts = 0; 
}