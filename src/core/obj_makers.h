#ifndef __obj_makers_h__
#define __obj_makers_h__

#pragma once 

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include "prerequisites.h"
#include "cuda_defs.h"

typedef boost::shared_ptr<c_ray_pool> ray_pool_ptr; 
typedef boost::shared_ptr<c_perspective_camera> perspective_cam_ptr; 

perspective_cam_ptr make_perspective_cam(const c_transform& cam_to_world, 
										const float screen_wnd[4], 
										float lensr, 
										float focal_d, 
										float fov, 
										uint32 res_x, 
										uint32 res_y);



ray_pool_ptr make_ray_pool(uint32 max_rays_per_chunk, uint32 spp_x, uint32 spp_y);


#endif // __obj_makers_h__
