#include "obj_makers.h"
#include "camera.h"
#include "ray_tracer.h"

perspective_cam_ptr make_perspective_cam(const c_transform& cam_to_world, 
										const float screen_wnd[4], 
										float lensr, 
										float focal_d, 
										float fov, 
										uint32 res_x, 
										uint32 res_y)
{
	perspective_cam_ptr cam;
	cam.reset(new c_perspective_camera(cam_to_world, screen_wnd, lensr, focal_d, fov, res_x, res_y));
	return cam; 
}

ray_pool_ptr make_ray_pool(uint32 max_rays_per_chunk, uint32 spp_x, uint32 spp_y)
{
	ray_pool_ptr ray_pool; 
	ray_pool.reset(new c_ray_pool(max_rays_per_chunk, spp_x, spp_y));
	return ray_pool; 
}