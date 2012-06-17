#ifndef __CAMERA_H__
#define __CAMERA_H__

#pragma once

#include <boost/shared_ptr.hpp>
#include "transform.h"

typedef boost::shared_ptr<c_render_target> render_target_ptr; 

class c_camera
{
public:
	explicit c_camera(const c_transform& cam_to_world, render_target_ptr film)
		: m_render_target(film)
		, m_camera_to_world(cam_to_world)
	{
	}

	virtual ~c_camera() {} 

	render_target_ptr get_render_target() const { return m_render_target; }


protected:
	render_target_ptr m_render_target;
	c_transform m_camera_to_world; 

};

//////////////////////////////////////////////////////////////////////////

class c_projective_camera : public c_camera
{
	typedef c_camera super;
public:
	c_projective_camera(const c_transform& cam_to_world, const c_transform& proj, const float screen_wnd[4], float lensr, float focal_d, render_target_ptr& film);

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
	c_perspective_camera(const c_transform& cam_to_world, const float screen_wnd[4], float lensr, float focal_d, float fov, render_target_ptr& film);
	// virtual float generate_ray(const c_camera_sample& cam_sample, c_ray *ray) const; 

	c_transform get_cam_to_world() const { return m_camera_to_world; }
	c_transform get_raster_to_cam() const { return m_raster_to_camera; }
	
private:
	vector3f m_dx, m_dy;
};

#endif