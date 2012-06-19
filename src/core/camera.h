#ifndef __CAMERA_H__
#define __CAMERA_H__

#pragma once

#include "cuda_defs.h"
#include "transform.h"

class c_camera
{
public:
	explicit c_camera(const c_transform& cam_to_world, uint32 res_x, uint32 res_y)
		: m_camera_to_world(cam_to_world)
		, m_res_x(res_x)
		, m_res_y(res_y)
	{
	}

	virtual ~c_camera() {} 

	uint32 res_x() const { return m_res_x; }
	uint32 res_y() const { return m_res_y; }

protected:

	uint32 m_res_x, m_res_y; 

	c_transform m_camera_to_world; 

};

//////////////////////////////////////////////////////////////////////////

class c_projective_camera : public c_camera
{
	typedef c_camera super;
public:
	c_projective_camera(const c_transform& cam_to_world, const c_transform& proj, const float screen_wnd[4], float lensr, float focal_d, uint32 res_x, uint32 res_y);

protected:
	// Projection Transformation
	c_transform m_cam_to_screen;
	c_transform m_screen_to_raster, m_raster_to_screen, m_raster_to_camera; 
	float m_lens_radius;
	float m_focal_distance; 
};

//////////////////////////////////////////////////////////////////////////

class c_perspective_camera : public c_projective_camera
{
	typedef c_projective_camera super; 

public:
	c_perspective_camera(const c_transform& cam_to_world, const float screen_wnd[4], float lensr, float focal_d, float fov, uint32 res_x, uint32 res_y);
	// virtual float generate_ray(const c_camera_sample& cam_sample, c_ray *ray) const; 

	c_transform get_cam_to_world() const { return m_camera_to_world; }
	c_transform get_raster_to_cam() const { return m_raster_to_camera; }
	
private:
	vector3f m_dx, m_dy;
};

#endif