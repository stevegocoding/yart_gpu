#include "cuda_defs.h"
#include "cuda_utils.h"
#include "cuda_rng.h"
#include "cuda_mem_pool.h"
#include "camera.h" 
#include "render_target.h"
#include "ray_tracer.h"
#include <stdio.h>

struct float4x4_t
{
	float elems[4][4];
};

// Constant memory data
__constant__ float4x4_t con_mat_cam2world; 
__constant__ float4x4_t con_mat_raster2cam; 

// ---------------------------------------------------------------------
/*
	Device Functions
*/ 
// ---------------------------------------------------------------------
__device__ float3 dfunc_transform_pt(float trans[4][4], float3 p)
{
	float3 ret; 
	
	// The homogeneous representation for points is [x, y, z, 1]^T.
	ret.x   = trans[0][0]*p.x + trans[0][1]*p.y + trans[0][2]*p.z + trans[0][3];
	ret.y   = trans[1][0]*p.x + trans[1][1]*p.y + trans[1][2]*p.z + trans[1][3];
	ret.z   = trans[2][0]*p.x + trans[2][1]*p.y + trans[2][2]*p.z + trans[2][3];
	float w = trans[3][0]*p.x + trans[3][1]*p.y + trans[3][2]*p.z + trans[3][3];

	if (w != 1.0f)
		ret /= w; 
	
	return ret;
}

__device__ float3 dfunc_transform_vec(float trans[4][4], float3 v)
{
	float3 ret;

	// Note: The homogeneous coords for v are [x, y, z, 0]^T.
	ret.x = trans[0][0]*v.x + trans[0][1]*v.y + trans[0][2]*v.z;
	ret.y = trans[1][0]*v.x + trans[1][1]*v.y + trans[1][2]*v.z;
	ret.z = trans[2][0]*v.x + trans[2][1]*v.y + trans[2][2]*v.z;

	return ret;
}

// ---------------------------------------------------------------------
/*
	Kernels
*/ 
// ---------------------------------------------------------------------


__global__ void kernel_gen_primary_rays(uint32 res_x, uint32 res_y,
										float sample_idx_x, float inv_spp_x, 
										float sample_idx_y, float inv_spp_y, 
										float z_near, float z_far, 
										float *d_rands1, float *d_rands2,
										c_ray_chunk out_chunk)
{
	uint32 pixel_idx = blockIdx.x * blockDim.x + threadIdx.x; 
	// printf("Pixel index: %u | Block index: %u | Thread index: %u \r\n", pixel_idx, blockIdx.x, threadIdx.x);
	// printf("width: %u | height: %u \r\n", res_x, res_y);

	if (pixel_idx < res_x * res_y)
	{
		// Assign rays following the Morton order (Z-curve). This was proposed by Aila2009.
		// See http://en.wikipedia.org/wiki/Z-order_%28curve%29

		// Extract even bits for x and odd bits for y raster coordinate.
		uint32 x = 0, y = 0;
		uint32 scr_pos = 0; // Starting with lsb bit 0.
		uint32 target_pos = 0;
		uint32 mask = 1;

		// Get raster coordinates for this thread.
		while(mask <= pixel_idx)
		{
			bool isOdd = scr_pos & 1;
			if(!isOdd && (mask & pixel_idx)) // even bit set?
				x |= 1 << target_pos;
			if( isOdd && (mask & pixel_idx)) // odd bit set?
				y |= 1 << target_pos;
			
			// Update mask.
			mask <<= 1;
			scr_pos++;

			// Increase target position in case we are done with the odd bit.
			if(isOdd)
				target_pos++;
		}
		
		// Stratified sampling
		float rnd1 = d_rands1[pixel_idx];
		float rnd2 = d_rands2[pixel_idx]; 
		
		rnd1 = (sample_idx_x + rnd1) * inv_spp_x; 
		rnd2 = (sample_idx_y + rnd2) * inv_spp_y; 
		
		// Generate rays 
		float3 pt_raster; 
		
		if (inv_spp_x * inv_spp_y < 1.0f)			// if multi-sampled per pixel
			pt_raster = make_float3((float)x + rnd1, (float)y + rnd2, 0.0f); 
		else 
			pt_raster = make_float3((float)x + 0.5f, (float)y + 0.5f, 0.0f);

		float3 pt_raster_cam = dfunc_transform_pt(con_mat_raster2cam.elems, pt_raster);
		float3 pt_raster_world = dfunc_transform_pt(con_mat_cam2world.elems, pt_raster_cam); 
		
		float3 origin_world = pt_raster_world; 
		float3 dir_cam = normalize(pt_raster_cam); 
		float3 dir_world = dfunc_transform_vec(con_mat_cam2world.elems, dir_cam); 
		dir_world = normalize(dir_world);

		printf("Origin world: %f, %f, %f \r\n", pt_raster.x, pt_raster.y, pt_raster.z);
		
		out_chunk.d_origins_array[pixel_idx] = make_float4(origin_world); 
		out_chunk.d_dirs_array[pixel_idx] = make_float4(dir_world);
		out_chunk.d_weights_array[pixel_idx] = make_float4(1.0f);
		out_chunk.d_pixels_array[pixel_idx] = y * res_x + x; 
	}
}

// ---------------------------------------------------------------------
/*
	Wrappers 
*/ 
// ---------------------------------------------------------------------
extern "C"
void ivk_krnl_gen_primary_rays(const c_perspective_camera *camera, 
							uint32 sample_idx_x, 
							uint32 num_samples_x, 
							uint32 sample_idx_y, 
							uint32 num_samples_y,
							PARAM_OUT c_ray_chunk& out_chunk)
{	
	uint32 _res_x = camera->res_x(); 
	uint32 _res_y = camera->res_y();

	c_transform cam_to_world = camera->get_cam_to_world(); 
	c_transform raster_to_cam = camera->get_raster_to_cam(); 

	// Copy matrix to constant memory
	float4x4_t mat_cam2world; 
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			mat_cam2world.elems[i][j] = cam_to_world.get_matrix().get_elem(i, j); 
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("con_mat_cam2world", &mat_cam2world, sizeof(float4x4_t)));
	
	float4x4_t mat_raster2cam; 
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			mat_raster2cam.elems[i][j] = raster_to_cam.get_matrix().get_elem(i, j); 
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("con_mat_raster2cam", &mat_raster2cam, sizeof(float4x4_t)));

	uint32 num_pixels = _res_x * _res_y; 
	
	dim3 block_size = dim3(16, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(num_pixels, block_size.x), 1); 
	
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	uint32 num_rand = rng.get_aligned_cnt(num_pixels); 
	c_cuda_memory<float> d_rands(2 * num_rand);
	
	rng.seed(rand()); 
	rng.gen_rand(d_rands.buf_ptr(), 2*num_rand); 

	float inv_num_spp_x = ((num_samples_x > 1) ? 1.0f / float(num_samples_x) : 1.0f); 
	float inv_num_spp_y = ((num_samples_y > 1) ? 1.0f / float(num_samples_y) : 1.0f);

	kernel_gen_primary_rays<<<grid_size, block_size>>>(_res_x, _res_y, 
													(float)sample_idx_x, inv_num_spp_x, (float)sample_idx_y, inv_num_spp_y,
													1.0f, 1000.0f, d_rands.buf_ptr(), d_rands.buf_ptr()+num_rand,  out_chunk); 

	out_chunk.depth = 0; 
	out_chunk.num_rays = num_pixels; 
} 