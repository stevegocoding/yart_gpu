#include "kernel_data.h"
#include "kdtree_kernel_data.h"
#include "ray_tracer.h"
#include "cuda_utils.h" 
#include "cuda_utils_device.h"
#include "math_utils.h"
#include "cuda_rng.h"
#include "cuda_mem_pool.h" 
#include "intersect_gpu.h"

/// \brief	Thread block size used for intersection search.
///	
///			For kernels calling dev_FindNextIntersectionKDWhileWhile().
#define INTERSECT_BLOCKSIZE	128

texture<float4, 1, cudaReadModeElementType> tex_tri_v0, tex_tri_v1, tex_tri_v2; 
texture<uint32, 1, cudaReadModeElementType> tex_tri_mat_idx; 
texture<uint32, 1, cudaReadModeElementType> tex_kdtree;

__constant__ c_light_data const_light_data; 
__constant__ c_material_desc const_mats_desc; 
__constant__ c_triangle_data const_tri_data;
__constant__ c_kdtree_data const_kdtree_data; 
__constant__ float k_ray_epsilon = 1e-3f;

//////////////////////////////////////////////////////////////////////////

__device__ float3 device_get_color_diffuse(uint32 idx_tri,
										uint32 idx_material, 
										char4 mat_flags, 
										float2 hit_barycoords)
{
	// @TODO: Add texture support 

	float3 diff_color; 
	diff_color = const_mats_desc.diff_color[idx_material]; 

	return diff_color; 
}

__device__ float3 device_sample_light(float3 pt, float3 pt_light_sample, float3 *out_wi)
{
	e_light_type light = const_light_data.type; 
	float3 l = const_light_data.emit_l; 

	// outW_i is incident direction (pointing from pt to light point).
	if (light == light_type_directional)
	{
		*out_wi = -const_light_data.direction; 
	}
	else if (light == light_type_area_disc)
	{
		
	}
	else if (light == light_type_area_quad)
	{

	}
	else if (light == light_type_point)
	{
		*out_wi = normalize(pt_light_sample - pt); 
		l /= device_distance_squared(pt_light_sample, pt); 
	}

	return l;
}

__device__ float3 device_calc_reflected_direct_light(float3 pt_eye, 
													float3 pt, 
													float3 n_geo, 
													float3 n_shading,
													float3 pt_light_sample,
													uint32 idx_tri, 
													uint32 idx_mat, 
													char4 mat_flags, 
													float2 hit_barycoords)
{
	// Get material's reflectance.
	float3 color_diff = device_get_color_diffuse(idx_tri, idx_mat, mat_flags, hit_barycoords); 
	
	float3 wi; 
	float3 li = device_sample_light(pt, pt_light_sample, &wi);

	float3 f = make_float3(0.0f); 
	float3 wo = normalize(pt_eye - pt); 
	
	// Evaluate only if w_o and w_i lie in the same hemisphere with respect to the 
	// geometric normal. This avoids light leaks and other problems resulting from the
	// use of shading normals. See PBR, p. 465.
	if (dot(wo, n_geo) * dot(wi, n_geo) > 0) 
		f = (color_diff) * M_INV_PI;
	
	return f * li * fabsf(dot(wi, n_shading)); 
}

__device__ void device_add_pixel_radiance(const c_ray_chunk& ray_chunk, 
										const c_shading_points& shading_pts,
										const uint32 tid, 
										const float3 l_sample, 
										float4 *d_radiance)
{
	float3 scaled_l = make_float3(ray_chunk.d_weights[tid]) * l_sample;
	
	uint32 pixel_idx = shading_pts.d_pixels[tid]; 

	printf("Intersection Not Found! \r\n"); 
	
	float4 lo = d_radiance[pixel_idx]; 
	lo.x += scaled_l.x; 
	lo.y += scaled_l.y; 
	lo.z += scaled_l.z;
	d_radiance[pixel_idx] = lo; 
}

template <bool need_closest>
inline __device__ int device_find_next_intersection_naive(const float3 ray_o, 
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
		bool is_hit = device_ray_tri_intersect(v0, v1, v2, ray_o, ray_d, lambda_hit, bary_hit1, bary_hit2);
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

template <bool need_closest> 
inline __device__ int device_find_next_intersection_kd_while_while(const float3 ray_origin, 
																const float3 ray_dir, 
																const float t_min_ray, 
																const float t_max_ray, 
																float& out_lambda, 
																float2& out_bary_hit)
{
	__shared__ float s_inv_dir[3][INTERSECT_BLOCKSIZE]; 
	__shared__ float s_idx_tri_inter[INTERSECT_BLOCKSIZE]; 

	// Precompute inverse ray direction
	s_inv_dir[0][threadIdx.x] = 1.0f / ray_dir.x; 
	s_inv_dir[1][threadIdx.x] = 1.0f / ray_dir.y; 
	s_inv_dir[2][threadIdx.x] = 1.0f / ray_dir.z; 
	s_idx_tri_inter[threadIdx.x] = -1; 

	// Compute the initial parametric range 
	float t_min_scene, t_max_scene; 
	const bool is_intersect_scene = 
		device_ray_box_intersect(const_kdtree_data.aabb_root_min, 
								const_kdtree_data.aabb_root_max, 
								ray_origin, 
								make_float3(s_inv_dir[0][threadIdx.x], s_inv_dir[1][threadIdx.x], s_inv_dir[2][threadIdx.x]),
								t_min_ray, 
								t_max_ray, 
								t_min_scene,
								t_max_scene); 
	
	if (!is_intersect_scene)
		return -1; 
	
	// Add epsilon to avoid floating point problems.
	t_max_scene = t_max_scene + k_ray_epsilon; 
	
	// Stack gets into local memory. Therefore access is *always* coalesced!
	// NOTE: Converting to vector leads to performance drops.
	uint32 todo_addr[KD_MAX_HEIGHT]; 
	float todo_tmin[KD_MAX_HEIGHT], todo_tmax[KD_MAX_HEIGHT]; 
	uint32 todo_pos = 0; 
	
	// Traverse kd-tree for ray
	int addr_node = 0;
	float t_min = t_min_scene; 
	float t_max = t_max_scene; 
	out_lambda = t_max; 
	
	do 
	{
		if (todo_pos > 0)
		{
			// Pop next node from stack
			todo_pos--;
			addr_node = todo_addr[todo_pos]; 
			t_min = todo_tmin[todo_pos]; 
			t_max = todo_tmax[todo_pos]; 
		}
		
		// Read node index + leaf info (MSB).
		// NOTE: We access preorder tree data directly without texture using Fermi's L1 cache.
		uint32 idx_node = const_kdtree_data.d_preorder_tree[addr_node]; 
		uint32 is_leaf = idx_node & 0x80000000; 
		idx_node &= 0x7FFFFFFF; 
		
		// Find next leaf 
		while (!is_leaf)
		{
			// Texture fetching probably results in a lot of serialization due to cache misses.
			uint32 addr_left = addr_node + 1 + 2; 
			uint2 parent_info = make_uint2(const_kdtree_data.d_preorder_tree[addr_node+1], 
											const_kdtree_data.d_preorder_tree[addr_node+2]); 
			uint32 addr_right = parent_info.x & 0x0FFFFFFF;
			uint32 split_axis = parent_info.x >> 30; 
			float split_pos = *(float*)&parent_info.y; 
			
			// Compute parametric distance along ray to split plane.
			float ray_origin_axis = ((float*)&ray_origin)[split_axis]; 
			float t_split = (split_pos - ray_origin_axis) * s_inv_dir[split_axis][threadIdx.x]; 

			uint32 addr_first;
			uint32 addr_sec;
			
			bool below_first = ray_origin_axis <= split_pos; 
			addr_first = (below_first ? addr_left : addr_right);
			addr_sec = (below_first ? addr_right : addr_left);	

			// Advance to next child node, possibly enqueue other child.
			// When the ray origin lies within the split plane (or very close to it), we have to
			// determine the child node using direction comparision. This case was discussed in
			// Havran's thesis "Heuristic Ray Shooting Algorithms", page 100.

			if (fabsf(ray_origin_axis - split_pos) < 1e-8f)
				// I use the inverse direction here instead of the actual value due to faster shared memory.
				addr_node = ((s_inv_dir[split_axis][threadIdx.x] > 0) ? addr_sec : addr_first); 
			
			// NOTE: The operators are very important! >=/<= leads to flickering in Sponza.
			else if (t_split > t_max || t_split < 0.0f)
				addr_node = addr_first; 
			else if (t_split < t_min)
				addr_node = addr_sec; 
			else
			{
				// Enqueue second child in todo list.
				todo_addr[todo_pos] = addr_sec; 
				todo_tmin[todo_pos] = t_split; 
				todo_tmax[todo_pos] = t_max; 
				todo_pos++; 
				
				addr_node = addr_first; 
				t_max = t_split; 
			}
			
			idx_node = const_kdtree_data.d_preorder_tree[addr_node]; 
			is_leaf = idx_node & 0x80000000;
			idx_node &= 0x7FFFFFFF; 
		}
		
		// Now we have a leaf. 
		// Check for intersections inside leaf node.
		uint32 num_tris = const_kdtree_data.d_preorder_tree[addr_node+1]; 
		for (uint32 t = 0; t < num_tris; ++t)
		{
			uint32 idx_tri = const_kdtree_data.d_preorder_tree[addr_node+2+t]; 
			
			// Texture fetching seems to be more efficient for my GTX 460. I tried to move the preorderTree
			// accesses to texture fetches and this to global memory accesses, but this hit the performance
			// drastically.
			float3 v0 = make_float3(tex1Dfetch(tex_tri_v0, idx_tri)); 
			float3 v1 = make_float3(tex1Dfetch(tex_tri_v1, idx_tri)); 
			float3 v2 = make_float3(tex1Dfetch(tex_tri_v2, idx_tri)); 

			float bary_hit1, bary_hit2, lambda_hit; 
			bool is_hit = device_ray_tri_intersect(v0, v1, v2, ray_origin, ray_dir, lambda_hit, bary_hit1, bary_hit2); 
			if (is_hit && lambda_hit > k_ray_epsilon && lambda_hit < out_lambda)
			{
				s_idx_tri_inter[threadIdx.x] = idx_tri;
				out_bary_hit = make_float2(bary_hit1, bary_hit2);
				out_lambda = lambda_hit; 
				
				if (!need_closest)
					return idx_tri; 
			}
		}
		if (out_lambda < t_max)
		{
			// Early exit!
			break; 
		}
	} while (todo_pos > 0); 

	return s_idx_tri_inter[threadIdx.x]; 
}

// ---------------------------------------------------------------------
/* 
/// \brief	Adds direct radiance from primary light source.
///
///			The function dev_GetReflectedDirectLight() is used to evaluate the direct radiance
///			reflected from each shading point into the direction given by the source ray.
///
/// \param	rayChunk					Source ray chunk. Compacted, so that rays hitting nothing
///										are removed.
/// \param	shadingPts					The shading points. Contains corresponding hits for ray
/// 									chunk. Compacted, so that invalid hits are removed. 
/// \param [in]		d_ShadowRayResult	The shadow ray result. Binary 0/1 buffer. Can be
/// 									generated using kernel_TraceShadowRaysArea() for area
/// 									lights and kernel_TraceShadowRaysDelta() for delta
/// 									lights. 
/// \param [in]		d_lightSamples		Sample point on area light sources for each shading
/// 									point. Set to \c NULL for delta light sources. 
/// \param	fScale						The scale factor. Radiance will be scaled by this factor,
/// 									before it is added to the accumulator. Can be used for
/// 									Monte-Carlo integration. 
/// \param [in,out]	d_ioRadiance		Radiance accumulator screen buffer, i.e. elements are
/// 									associated to screen's pixels. 
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_add_direct_radiance(c_ray_chunk ray_chunk,
										c_shading_points shading_pts,
										uint32 *d_shadow_ray_result, 
										float4 *d_light_samples,
										float scale, 
										float4 *d_io_radiance)
{
	uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	// We compacted the both ray chunk and shading points, so no invalid triangle indices.
	if (tid < ray_chunk.num_rays)
	{
		float3 ray_origin = make_float3(ray_chunk.d_origins[tid]); 
		float2 bary_hit = shading_pts.d_isect_barycoords[tid]; 

		// Get the intersection points 
		int idx_tri = shading_pts.d_tri_indices[tid]; 
		float3 isect_pt = make_float3(shading_pts.d_isect_pts[tid]); 
		uint32 idx_material = tex1Dfetch(tex_tri_mat_idx, idx_tri); 
		char4 mat_flags = make_char4(0, 0, 0, 0); 
		
		// Get shadow ray result 
		uint32 is_light_unoccluded; 
		if (d_shadow_ray_result)
			is_light_unoccluded = d_shadow_ray_result[tid]; 
		else 
			is_light_unoccluded = 1;
		
		// Get light sample
		float3 pt_light_sample = const_light_data.position; 
		if (d_light_samples)
		{
			float4 pt_l = d_light_samples[tid]; 
			pt_light_sample = make_float3(pt_l.x, pt_l.y, pt_l.z);
		}

		float3 l_segment = make_float3(0.0f, 0.0f, 0.0f);
		float3 n_shading = make_float3(shading_pts.d_geo_normals[tid]);
		float3 n_geo = make_float3(shading_pts.d_shading_normals[tid]); 
		if (is_light_unoccluded)
			l_segment = device_calc_reflected_direct_light(ray_origin, 
														isect_pt, 
														n_geo, 
														n_shading, 
														pt_light_sample, 
														idx_tri, 
														idx_material, 
														mat_flags,
														bary_hit); 
		l_segment *= scale; 

		l_segment = device_get_color_diffuse(idx_tri, idx_material, mat_flags, make_float2(0.0f, 0.0f)); 
		
		device_add_pixel_radiance(ray_chunk, shading_pts, tid, l_segment, d_io_radiance); 
	}
}

// ---------------------------------------------------------------------
/*
	Searches ray hit points for given ray chunk.
	
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_find_intersections(c_ray_chunk ray_chunk, 
										c_shading_points shading_pts, 
										uint32 *d_out_is_valid)
{
	uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (idx < ray_chunk.num_rays)
	{
		const float3 ray_origin = make_float3(ray_chunk.d_origins[idx]);
		const float3 ray_dir = make_float3(ray_chunk.d_dirs[idx]);
		
		// NOTE: Currenlty we also trace rays with zero influence, if any.
		float lambda;
		float2 hit_pt_bary;
		int tri_idx = device_find_next_intersection_naive<true>(ray_origin, ray_dir, k_ray_epsilon, M_INFINITY, lambda, hit_pt_bary);

		shading_pts.d_pixels[idx] = ray_chunk.d_pixels[idx]; 
		shading_pts.d_tri_indices[idx] = tri_idx; 
		
		// Avoid branching, so just calculate
		shading_pts.d_isect_pts[idx] = make_float4(ray_origin + ray_dir * lambda);
		shading_pts.d_isect_barycoords[idx] = hit_pt_bary; 

		// Store if this result is valid.
		d_out_is_valid[idx] = ((tri_idx != -1)? 1 : 0);
	}
	
}

__global__  void kernel_find_intersections_kd(c_ray_chunk ray_chunk, 
											c_shading_points shading_pts, 
											uint32 *d_out_is_valid)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Intersection Not Found! \r\n"); 
	
	if (idx < ray_chunk.num_rays)
	{
		const float3 ray_origin = make_float3(ray_chunk.d_origins[idx]); 
		const float3 ray_dir = make_float3(ray_chunk.d_dirs[idx]); 
		
		// NOTE: Currenlty we also trace rays with zero influence, if any.
		float lambda; 
		float2 bary_hit; 
		int idx_tri = device_find_next_intersection_kd_while_while<true>(ray_origin, 
																		ray_dir,
																		k_ray_epsilon, 
																		M_INFINITY, 
																		lambda, 
																		bary_hit); 
		shading_pts.d_pixels[idx] = ray_chunk.d_pixels[idx]; 
		shading_pts.d_tri_indices[idx] = idx_tri; 
		
		// Avoid branching, so just calculate
		shading_pts.d_isect_pts[idx] = make_float4(ray_origin + ray_dir * lambda); 
		shading_pts.d_isect_barycoords[idx] = bary_hit; 

		if (idx_tri != -1)
			printf("Intersection Found! tri_idx: %d \r\n", idx_tri); 
		else 
			printf("Intersection Not Found! \r\n"); 

		// Store if this result is valid.
		d_out_is_valid[idx] = ((idx_tri != -1)? 1 : 0); 
	}
}

__global__ void kernel_add_emitted_indirect(float4 *d_indirect_illum, 
											c_ray_chunk ray_chunk, 
											c_shading_points shading_pts, 
											float4 *d_io_radiance)
{
	uint32 tid = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (tid < ray_chunk.num_rays)
	{
		float3 l_add = make_float3(0.0f, 0.0f, 0.0f);
		
		// Get the intersection point 
		int idx_tri = shading_pts.d_tri_indices[tid];
		
		
		float4 l_indirect = d_indirect_illum[tid]; 
		l_add.x += l_indirect.x; 
		l_add.y += l_indirect.y; 
		l_add.z += l_indirect.z;

		device_add_pixel_radiance(ray_chunk, shading_pts, tid, l_add, d_io_radiance); 
	}
}

////////////////////////////////////////////////////////////////////////// 

extern "C"
void kernel_wrapper_trace_rays(const c_ray_chunk& ray_chunk, PARAM_OUT c_shading_points& out_isects, uint32 *d_out_is_valid)
{
	dim3 block_size = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size.x), 1, 1);

	kernel_find_intersections_kd<<<grid_size, block_size>>>(ray_chunk, out_isects, d_out_is_valid);
	CUDA_CHECKERROR;

	out_isects.num_pts = ray_chunk.num_rays; 
}


extern "C"
void kernel_wrapper_solve_lte(const c_ray_chunk& ray_chunk, 
							const c_shading_points& shading_pts, 
							const c_light_data& lights, 
							float4 *d_radiance_indirect, 
							bool is_direct_rt, 
							bool is_shadow_rays, 
							uint2 area_light_samples, 
							float4 *d_io_radiance)
{
	dim3 block_size = dim3(INTERSECT_BLOCKSIZE, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size.x), 1, 1); 
	dim3 block_size2 = dim3(256, 1, 1); 
	dim3 grid_size2 = dim3(CUDA_DIVUP(ray_chunk.num_rays, block_size2.x), 1, 1); 

	if (is_direct_rt)
	{
		c_cuda_memory<uint32> d_shadow_rays_result(shading_pts.num_pts); 
		
		// For shadow rays if required
		c_cuda_rng& rng = c_cuda_rng::get_instance(); 
		uint32 seed = 3494; 
		srand(seed); 
		
		if (lights.type != light_type_area_disc && lights.type != light_type_area_quad)
		{
			if (!is_shadow_rays)
				cuda_init_constant(d_shadow_rays_result.buf_ptr(), shading_pts.num_pts, (uint32)1); 
			else 
			{
				// Trace shadow rays
			}
			CUDA_CHECKERROR; 

			kernel_add_direct_radiance<<<grid_size2, block_size2>>>(ray_chunk, 
																	shading_pts, 
																	d_shadow_rays_result.buf_ptr(), 
																	NULL, 
																	1.0f, 
																	d_io_radiance);
		}
		else	// For area light 
		{
			
		}
	}

	// Add emitted and in indirect illumination from photon maps.
	kernel_add_emitted_indirect<<<grid_size2, block_size2>>>(d_radiance_indirect, 
															ray_chunk, 
															shading_pts, 
															d_io_radiance); 
}



extern "C"
void update_raytracing_kernel_data(const c_light_data& lights, 
									const c_triangle_data& tris,
									const c_material_data& mats, 
									const c_kdtree_data& kdtree,
									float ray_epsilon)
{
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("const_light_data", &lights, sizeof(c_light_data)));
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("const_mats_desc", &mats, sizeof(c_material_desc))); 
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("const_kdtree_data", &kdtree, sizeof(c_kdtree_data))); 
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("const_tri_data", &tris, sizeof(c_triangle_data))); 
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("k_ray_epsilon", &ray_epsilon, sizeof(float)));

	// Channel format
	cudaChannelFormatDesc cd_float4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); 
	cudaChannelFormatDesc cd_uint = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	// Bind triangle data to texture reference

	// Vertex
	tex_tri_v0.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_v0, tris.d_verts[0], cd_float4, tris.num_tris*sizeof(float4)));
	tex_tri_v1.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_v1, tris.d_verts[1], cd_float4, tris.num_tris*sizeof(float4)));
	tex_tri_v2.normalized = false;
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_v2, tris.d_verts[2], cd_float4, tris.num_tris*sizeof(float4)));

	// Materials
	tex_tri_mat_idx.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_mat_idx, tris.d_material_idx, cd_uint, tris.num_tris*sizeof(uint32))); 

	// KD-Tree 
	tex_kdtree.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_kdtree, kdtree.d_preorder_tree, cd_uint, kdtree.size_tree*sizeof(uint32)));
};

extern "C"
void kernel_wrapper_cleanup_kernel_data()
{	
	// Triangle data 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v0)); 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v1)); 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v2));

	// Triangle material index 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_mat_idx)); 
	
	// KD-Tree data 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_kdtree));  	
}

extern "C"
void init_raytracing_kernels()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	
	// We need no shared memory for these kernels. So prefer L1 caching.
	cuda_safe_call_no_sync(cudaFuncSetCacheConfig(kernel_find_intersections, cudaFuncCachePreferL1)); 
	cuda_safe_call_no_sync(cudaFuncSetCacheConfig(kernel_find_intersections_kd, cudaFuncCachePreferL1)); 
}

extern "C"
void cleanup_raytracing_kernels()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 

	// Triangle Data 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v0));
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v1));
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v2)); 

	// KD-Tree Data 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_kdtree)); 

	// Material Data 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_mat_idx));
}