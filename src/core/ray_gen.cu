
#include "cuda_defs.h"
#include "cuda_utils.h"
#include "cuda_rng.h"
#include "camera.h" 
#include "render_target.h"
#include "ray_tracer.h"

struct float4x4_t
{
	float elems[4][4];
};

// Constant memory data
__constant__ float4x4_t con_mat_cam2world; 
__constant__ float4x4_t con_mat_raster2cam; 

extern "C"
void ivk_krnl_gen_primary_rays(const c_perspective_camera *camera, 
							uint32 sample_idx_x, 
							uint32 num_samples_x, 
							uint32 sample_idx_y, 
							uint32 num_samples_y,
							PARAM_OUT c_ray_chunk *out_chunk)
{
	/*
	uint32 res_x = camera->res_x(); 
	uint32 res_y = camera->res_y();

	c_transform cam_to_world = camera->get_cam_to_world(); 
	c_transform raster_to_cam = camera->get_raster_to_cam(); 

	// Copy matrix to constant memory
	float4x4_t mat_cam2world; 
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			mat_cam2world.elems[i][j] = cam_to_world.get_matrix().get_elem(i, j); 
	cudaMemcpyToSymbol("con_mat_cam2world", &mat_cam2world, sizeof(float4x4_t)); 
	
	float4x4_t mat_raster2cam; 
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 4; ++j)
			mat_raster2cam.elems[i][j] = raster_to_cam.get_matrix().get_elem(i, j); 
	cudaMemcpyToSymbol("con_mat_raster2cam", &mat_raster2cam, sizeof(float4x4_t)); 

	uint32 num_pixels = res_x * res_y; 
	
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(cuda_div_up(num_pixels, block_size.x), 1); 
	
	c_cuda_rng& rng = get_rng_instance();
	uint32 num_rand = rng.get_aligned_cnt(num_pixels);  
	*/


}