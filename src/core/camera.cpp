#include "camera.h"
#include "render_target.h"
#include "sampler.h"

c_projective_camera::c_projective_camera(
	const c_transform& cam_to_world, 
	const c_transform& proj, 
	const float screen_wnd[4], 
	float lensr, 
	float focal_d,
	uint32 res_x, 
	uint32 res_y)
	: super(cam_to_world, res_x, res_y)
	, m_cam_to_screen(proj)
	, m_lens_radius(lensr)
	, m_focal_distance(focal_d)
{
	m_screen_to_raster = make_scale((float)m_res_x, 
		(float)m_res_y, 
		1.0f) * 
		make_scale(1.0f / (screen_wnd[1] - screen_wnd[0]), 1.0f / (screen_wnd[2] - screen_wnd[3]), 1.0f) * 
		make_translate(vector3f(-screen_wnd[0], -screen_wnd[3], 0)); 

	m_raster_to_screen = inverse_transform(m_screen_to_raster);
	m_raster_to_camera = inverse_transform(m_cam_to_screen) * m_raster_to_screen;	
}

//////////////////////////////////////////////////////////////////////////

c_perspective_camera::c_perspective_camera(
	const c_transform& cam_to_world, 
	const float screen_wnd[4], 
	float lensr, 
	float focal_d, 
	float fov, 
	uint32 res_x, 
	uint32 res_y)
	: super(cam_to_world, make_perspective_proj(fov, 1e-2f, 1000.0f), screen_wnd, lensr, focal_d, res_x, res_y)
{
	//vector3f right = m_raster_to_camera.transform_pt(vector3f(1,0,0));
	//vector3f left = m_raster_to_camera.transform_pt(vector3f());
} 

/*
float c_perspective_camera::generate_ray(const c_camera_sample& cam_sample, c_ray *ray) const 
{
	vector3f pt_sample(cam_sample.image_x, cam_sample.image_y, 0); 
	vector3f pt_sample_cam = m_raster_to_camera.transform_pt(pt_sample); 

	// Create the ray according to the sample
	*ray = c_ray(vector3f(0,0,0), normalize(pt_sample_cam));

	// @TODO: Modify the ray for depth of field

	// Transform the ray from camera space to world space
	*ray = m_camera_to_world.transform_ray(*ray); 

	return 1.0f;
}
*/ 

