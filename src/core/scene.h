#ifndef __scene_h__
#define __scene_h__

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>

#include "bounding_box.h"

class c_triangle_mesh; 
typedef boost::shared_ptr<c_triangle_mesh> triangle_mesh_ptr; 
typedef std::vector<triangle_mesh_ptr> triangle_meshes_array; 

class c_triangle_mesh2; 
typedef boost::shared_ptr<c_triangle_mesh2> triangle_mesh2_ptr; 
typedef std::vector<triangle_mesh2_ptr> triangle_meshes2_array; 

class c_scene 
{
	
public:
	c_scene(triangle_meshes2_array& meshes, const c_aabb& bounds)
		: m_meshes(meshes) 
		, m_bounds(bounds)
	{} 

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

private:
	triangle_meshes2_array m_meshes; 

	c_aabb m_bounds; 
	
};

typedef boost::shared_ptr<c_scene> scene_ptr; 

#endif // __scene_h__
