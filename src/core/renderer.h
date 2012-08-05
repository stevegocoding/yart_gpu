#ifndef __renderer_h__
#define __renderer_h__

#pragma once 

#include <assert.h>
#include <vector_types.h> 
#include "prerequisites.h"
#include "kernel_data.h"
#include "obj_makers.h"

struct c_ray_chunk; 

class c_scene;
typedef boost::shared_ptr<c_scene> scene_ptr; 

class c_kdtree_triangle; 
typedef boost::shared_ptr<c_kdtree_triangle> kdtree_tri_ptr; 

class c_renderer
{
public:
	c_renderer(scene_ptr scene);
	virtual ~c_renderer(); 

	void initialise(); 
	void destroy();

	bool render_scene(uchar4 *d_screen_buf); 

private: 
	
	bool rebuild_obj_kdtree(); 

	bool render_to_buf(PARAM_OUT float4 *d_radiance); 
	uint32 trace_rays(c_ray_chunk *ray_chunk, c_shading_points *sp_shading, uint32 *d_src_addr = NULL);
	
	ray_pool_ptr m_ray_pool; 
	scene_ptr m_scene;
	kdtree_tri_ptr m_kdtree_tri; 
	c_shading_points m_sp_shading; 

	// ---------------------------------------------------------------------
	/*
		Renderer Settings 
	*/ 
	// ---------------------------------------------------------------------
 
	uint32 m_spp_x, m_spp_y; 
	bool m_enable_direct_rt; 
	bool m_enable_shadow_rays;
	bool m_is_dynamic_scene;
	uint32 m_area_light_samples_x, m_area_light_samples_y;
	float m_ray_epsilon; 
};

#endif // __renderer_h__
