#include "kernel_data.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

texture<float4, 1, cudaReadModeElementType> tex_tri_v0, tex_tri_v1, tex_tri_v2; 
texture<float4, 1, cudaReadModeElementType> tex_tri_n0, tex_tri_n1, tex_tri_n2; 

texture<uint32, 1, cudaReadModeElementType> tex_tri_mat_idx; 

__global__ void kernel_get_normal_hit_pt(uint32 count, 
										int *d_tri_hit_indices,
										float2 *d_hit_barycoords, 
										float4 *d_out_normals_geo, 
										float4 *d_out_nromals_shading)
{
	uint32 tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < count)
	{
		int idx_tri = d_tri_hit_indices[tid]; 
		float3 n_geo = make_float3(1.0f, 0.0f, 0.0f); 
		float3 n_shading = make_float3(1.0f, 0.0f, 0.0f);
		
		if (idx_tri != -1)
		{		
			float2 bary_hit = d_hit_barycoords[tid]; 
			
			// Get geometry normal from vertices 
			float3 p[3];
			p[0] = make_float3(tex1Dfetch(tex_tri_v0, idx_tri)); 
			p[1] = make_float3(tex1Dfetch(tex_tri_v1, idx_tri));
			p[2] = make_float3(tex1Dfetch(tex_tri_v2, idx_tri));

			n_geo = normalize(cross(p[1] - p[0], p[2] - p[0]));
			
			// @TODO: Check if we have a bump map

			float3 n0 = make_float3(tex1Dfetch(tex_tri_n0, idx_tri));
			float3 n1 = make_float3(tex1Dfetch(tex_tri_n1, idx_tri));
			float3 n2 = make_float3(tex1Dfetch(tex_tri_n2, idx_tri));
			float3 bary = make_float3(1.0f - bary_hit.x - bary_hit.y, bary_hit.x, bary_hit.y); 
			n_shading = n0 * bary.x + n1 * bary.y + n2 * bary.z;

			if (dot(n_geo, n_shading) < 0.0f)
				n_geo *= -1.0f; 
		}

		d_out_normals_geo[tid] = make_float4(n_geo.x, n_geo.y, n_geo.z, 0.0f); 
		d_out_nromals_shading[tid] = make_float4(n_shading.x, n_shading.y, n_shading.z, 0.0f);
	}
}

//////////////////////////////////////////////////////////////////////////

extern "C"
void kernel_wrapper_bsdf_get_normal_hit_pt(uint32 count, 
										int *d_tri_hit_indices, 
										float2 *d_hit_barycoords, 
										float4 *d_out_normals_geo, 
										float4 *d_out_normals_shading)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(count, block_size.x), 1, 1);
	
	kernel_get_normal_hit_pt<<<grid_size, block_size>>>(count, 
														d_tri_hit_indices, 
														d_hit_barycoords,
														d_out_normals_geo,
														d_out_normals_shading); 

	CUDA_CHECKERROR; 
}