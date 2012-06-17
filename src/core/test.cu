
#include "cuda_defs.h"
#include "camera.h" 
#include "render_target.h"
#include "ray_gen.h"


extern "C"
void ivk_krnl_gen_primary_rays(const c_camera *camera, 
							uint32 sample_idx_x, 
							uint32 num_samples_x, 
							uint32 sample_idx_y, 
							uint32 num_samples_y,
							c_ray_chunk *out_chunk)
{
	uint32 res_x = camera->get_render_target()->res_x(); 
	

}

