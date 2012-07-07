#ifndef __renderer_h__
#define __renderer_h__

#pragma once 

#include <assert.h>
#include <vector_types.h> 
#include "prerequisites.h"
#include "kernel_data.h"
#include "obj_makers.h"

struct c_ray_chunk; 
class c_renderer
{
public:
	c_renderer(ray_pool_ptr ray_pool, perspective_cam_ptr camera);
	virtual ~c_renderer(); 

	bool render_scene(uchar4 *d_screen_buf); 
	
private: 
	
	void initialise(uint32 screen_size); 
	void destroy();
	bool render_to_buf(PARAM_OUT float4 *d_radiance);

	uint32 trace_rays(c_ray_chunk *ray_chunk, c_shading_points_array *sp_shading, PARAM_OUT uint32 *d_src_addr = NULL);
	
	ray_pool_ptr m_ray_pool; 
	perspective_cam_ptr m_camera;
	c_shading_points_array m_sp_shading;
	
};

#endif // __renderer_h__
