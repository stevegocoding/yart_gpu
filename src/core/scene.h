#ifndef __scene_h__
#define __scene_h__

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>

#include "bounding_box.h"
#include "kernel_data.h"
#include "camera.h"

class c_triangle_mesh; 
typedef boost::shared_ptr<c_triangle_mesh> triangle_mesh_ptr; 
typedef std::vector<triangle_mesh_ptr> triangle_meshes_array; 

class c_triangle_mesh2; 
typedef boost::shared_ptr<c_triangle_mesh2> triangle_mesh2_ptr; 
typedef std::vector<triangle_mesh2_ptr> triangle_meshes2_array; 

class c_perspective_camera;
typedef boost::shared_ptr<c_perspective_camera> perspective_cam_ptr; 

struct scene_material
{
	scene_material(const std::string& _name, const c_vector3f& diff_color, const c_vector3f& spec_color, float _spec_exp, bool area_light = false)
		: is_area_light(area_light)
		, diffuse_color(diff_color)
		, specular_color(spec_color)
		, spec_exp(_spec_exp) 
		, name(_name)
	{} 
	
	bool is_area_light; 
	c_vector3f diffuse_color; 
	c_vector3f specular_color; 
	float spec_exp;
	std::string name;
};

struct scene_light
{
	scene_light(e_light_type type, 
				const c_vector3f& _pos, 
				const c_vector3f& _dir, 
				const c_vector3f& _emit, 
				const c_vector3f& _area_v1 = c_vector3f(0.f, 0.f, 0.f), 
				const c_vector3f& _area_v2 = c_vector3f(0.f, 0.f, 0.f), 
				const float _area_radius = 0.f)
				: pos(_pos)
				, direction(_dir)
				, emit_l(_emit)
				, area_v1(_area_v1)
				, area_v2(_area_v2)
				, area_radius(_area_radius)
	{	
	}

	e_light_type light_type; 
	c_vector3f pos; 
	c_vector3f direction; 
	c_vector3f emit_l; 
	c_vector3f area_v1; 
	c_vector3f area_v2; 
	float area_radius; 
};

static scene_light make_point_light(const c_vector3f& pos, const c_vector3f& intense)
{
	return scene_light(light_type_point, pos, c_vector3f(0.f, 0.f, 0.f), intense); 
}

typedef std::vector<scene_light> scene_light_array;
typedef std::vector<scene_material> scene_material_array; 

class c_scene 
{
public:
	c_scene(triangle_meshes2_array& meshes, 
			const c_aabb& bounds, 
			perspective_cam_ptr cam, 
			scene_light_array& lights,
			scene_material_array& mats)
		: m_meshes(meshes) 
		, m_bounds(bounds)
		, m_perspective_cam(cam)
	{
		init_light_data(lights);
		init_materials_data(mats);
		init_triangle_data(meshes, bounds);
	} 

	virtual ~c_scene(); 

	size_t get_num_meshes() const 
	{
		return m_meshes.size();
	}

	triangle_mesh2_ptr get_triangle_mesh(unsigned int idx) 
	{ 
		assert(idx < m_meshes.size());
		return m_meshes[idx];
	}

	size_t get_num_tri_total() const; 
	const c_aabb& get_bounds() const { return m_bounds; }
	perspective_cam_ptr get_cam() { return m_perspective_cam; }
	c_light_data& get_light_data() { return m_light_data; }
	c_triangle_data& get_triangle_data() { return m_tri_data; }
	c_material_data& get_material_data() { return m_material_data; } 
	
private:
	void init_triangle_data(triangle_meshes2_array& meshes, const c_aabb& scene_bounds);
	void release_triangle_data();  
	
	void init_light_data(scene_light_array& lights)
	{
		for (size_t i = 0; i < lights.size(); ++i)
		{
			m_light_data.type = lights[i].light_type; 
			m_light_data.position = *(float3*)&(lights[i].pos); 
			m_light_data.direction = *(float3*)&(lights[i].direction);
			m_light_data.emit_l = *(float3*)&(lights[i].emit_l);
			m_light_data.area_v1 = *(float3*)&(lights[i].area_v1);
			m_light_data.area_v2 = *(float3*)&(lights[i].area_v2); 
			m_light_data.area_radius = lights[i].area_radius; 
		}
	}
	
	void init_materials_data(scene_material_array& mats)
	{
		assert(mats.size() < MAX_MATERIALS); 
		
		for (size_t i = 0; i < mats.size(); ++i)
		{
			m_material_data.mats_desc.diff_color[i] = *(float3*)&(mats[i].diffuse_color);
			m_material_data.mats_desc.spec_color[i] = *(float3*)&(mats[i].specular_color); 
			m_material_data.mats_desc.spec_exp[i] = mats[i].spec_exp; 
			m_material_data.num_materials ++;
		}
	}
	
	triangle_meshes2_array m_meshes;  
	c_aabb m_bounds; 
	std::vector<scene_material> m_scene_materials; 
	
	// ---------------------------------------------------------------------
	/* GPU Data 
	*/ 
	// ---------------------------------------------------------------------
	c_triangle_data m_tri_data; 
	c_light_data m_light_data; 
	c_material_data m_material_data;
	
	// ---------------------------------------------------------------------
	/* Camera
	*/ 
	// ---------------------------------------------------------------------
	perspective_cam_ptr m_perspective_cam; 
};

typedef boost::shared_ptr<c_scene> scene_ptr; 

#endif // __scene_h__
