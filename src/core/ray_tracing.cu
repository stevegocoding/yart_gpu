#include "kernel_data.h"
#include "ray_tracer.h"
#include "cuda_utils.h" 
#include "math_utils.h"

#include "intersect_gpu.h"

/// \brief	Thread block size used for intersection search.
///	
///			For kernels calling dev_FindNextIntersectionKDWhileWhile().
#define INTERSECT_BLOCKSIZE	128

texture<float4, 1, cudaReadModeElementType> tex_tri_v0, tex_tri_v1, tex_tri_v2; 

__constant__ c_triangle_data const_tri_data; 

__constant__ float k_ray_epsilon = 1e-3f;


__device__ void dfunc_add_pixel_radiance(const c_ray_chunk& ray_chunk, 
										const c_shading_points_array& shading_pts,
										const uint32 tid, 
										const float3 l_sample, 
										float4 *d_radiance)
{
	float3 scaled_l = make_float3(ray_chunk.d_weights_array[tid]) * l_sample;
	
	uint32 pixel_idx = shading_pts.d_pixels_array[tid]; 
	
	float4 lo = d_radiance[pixel_idx]; 
	lo.x += scaled_l.x; 
	lo.y += scaled_l.y; 
	lo.z += scaled_l.z;
	d_radiance[pixel_idx] = lo; 
}

template <bool need_closest>
inline __device__ int d_find_next_intersection_naive(const float3 ray_o, 
													const float3 ray_d, 
													const float t_min, 
													float t_max, 
													float& out_lambda, 
													float2& out_hit_pt_bary)
{
	// Move some data into shared memory to save registers.
	// __shared__ float s_inv_dir[3][INTERSECT_BLOCKSIZE];
	__shared__ float s_idx_tri_isect[INTERSECT_BLOCKSIZE]; 

	// Precompute inverse ray direction.
	/*
	s_inv_dir[0][threadIdx.x] = 1.0f / ray_d.x;
	s_inv_dir[1][threadIdx.x] = 1.0f / ray_d.x; 
	s_inv_dir[2][threadIdx.x] = 1.0f / ray_d.x;
	*/

	s_idx_tri_isect[threadIdx.x] = -1; 

	uint32 num_tris = const_tri_data.num_tris; 
	for (uint32 i = 0; i < num_tris; ++i)
	{
		float3 v0 = make_float3(tex1Dfetch(tex_tri_v0, i)); 
		float3 v1 = make_float3(tex1Dfetch(tex_tri_v1, i)); 
		float3 v2 = make_float3(tex1Dfetch(tex_tri_v2, i));

		float bary_hit1, bary_hit2, lambda_hit; 
		bool is_hit = d_ray_tri_intersect(v0, v1, v2, ray_o, ray_d, lambda_hit, bary_hit1, bary_hit2);
		if (is_hit && lambda_hit > k_ray_epsilon && lambda_hit < out_lambda)
		{
			s_idx_tri_isect[threadIdx.x] = i; 
			out_hit_pt_bary = make_float2(bary_hit1, bary_hit2);
			out_lambda = lambda_hit; 
			if (!need_closest)
				return i;  
		}
	}
	
	return s_idx_tri_isect[threadIdx.x]; 
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

// ---------------------------------------------------------------------
/*
	Searches ray hit points for given ray chunk.
	
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_find_intersections(c_ray_chunk ray_chunk, c_shading_points_array shading_pts, uint32 *d_out_is_valid)
{
	uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < ray_chunk.num_rays)
	{
		const float3 ray_origin = make_float3(ray_chunk.d_origins_array[idx]);
		const float3 ray_dir = make_float3(ray_chunk.d_dirs_array[idx]);
		
		// NOTE: Currenlty we also trace rays with zero influence, if any.
		float lambda;
		float2 hit_pt_bary;
		int tri_idx = d_find_next_intersection_naive<true>(ray_origin, ray_dir, k_ray_epsilon, M_INFINITY, lambda, hit_pt_bary);

		shading_pts.d_pixels_array[idx] = ray_chunk.d_pixels_array[idx]; 
		shading_pts.d_tri_idx_array[idx] = tri_idx; 
		
		// Avoid branching, so just calculate
		shading_pts.d_isect_pts_array[idx] = make_float4(ray_origin + ray_dir * lambda);
		shading_pts.d_isect_bary_array[idx] = hit_pt_bary; 

		// Store if this result is valid.
		d_out_is_valid[idx] = ((tri_idx != -1)? 1 : 0);
	}
	
}

////////////////////////////////////////////////////////////////////////// 

extern "C"
void launch_kernel_trace_rays(const c_ray_chunk& ray_chunk, PARAM_OUT c_shading_points_array& out_isects, uint32 *d_out_is_valid)
{
	dim3 block_size = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size.x), 1, 1);

	kernel_find_intersections<<<grid_size, block_size>>>(ray_chunk, out_isects, d_out_is_valid);
	
	out_isects.num_pts = ray_chunk.num_rays; 
}


extern "C"
void launch_kernel_solve_lte(const c_ray_chunk& ray_chunk, 
						const c_shading_points_array& shading_pts, 
						float4 *d_radiance)
{
	dim3 block_size2 = dim3(256, 1, 1); 
	dim3 grid_size2 = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size2.x), 1, 1); 

	kernel_add_direct_radiance<<<grid_size2, block_size2>>>(ray_chunk, shading_pts, NULL, NULL, 1.0f, d_radiance);
}

