#include "kernel_data.h"
#include "ray_tracer.h"
#include "cuda_utils.h" 

__device__ void dfunc_add_pixel_radiance(const c_ray_chunk& ray_chunk, 
										const c_shading_points_array& shading_pts,
										const uint32 tid, 
										const float3 l_sample, 
										float4 *d_radiance)
{
	float3 scaled_l = make_float3(ray_chunk.d_weights_array[tid]) * l_sample;
	
	uint32 pixel_idx = shading_pts.d_pixels_buf[tid]; 
	
	float4 lo = d_radiance[pixel_idx]; 
	lo.x += scaled_l.x; 
	lo.y += scaled_l.y; 
	lo.z += scaled_l.z;
	d_radiance[pixel_idx] = lo; 
}

__global__ void kernel_add_direct_radiance(c_ray_chunk ray_chunk,
										c_shading_points_array shading_pts,
										uint32 *d_shadow_ray_result, 
										float4 *d_light_samples,
										float scale, 
										float4 *d_radiance)
{
	uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// We compacted the both ray chunk and shading points, so no invalid triangle indices.
	if (tid < ray_chunk.num_rays)
	{

		float3 l_segment = make_float3(0.0f, 0.0f, 0.0f); 
		l_segment *= scale; 
		
		dfunc_add_pixel_radiance(ray_chunk, shading_pts, tid, l_segment, d_radiance); 
	}
}


//////////////////////////////////////////////////////////////////////////


extern "C"
void ivk_krnl_solve_lte(const c_ray_chunk& ray_chunk, 
						const c_shading_points_array& shading_pts, 
						float4 *d_radiance)
{
	dim3 block_size2 = dim3(256, 1, 1); 
	dim3 grid_size2 = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size2.x), 1, 1); 

	kernel_add_direct_radiance<<<grid_size2, block_size2>>>(ray_chunk, shading_pts, NULL, NULL, 1.0f, d_radiance);
}