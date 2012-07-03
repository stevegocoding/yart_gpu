#ifndef __scene_h__
#define __scene_h__

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>

class c_triangle_mesh; 
typedef boost::shared_ptr<c_triangle_mesh> triangle_mesh_ptr; 
typedef std::vector<triangle_mesh_ptr> triangle_meshes_array; 

class c_triangle_mesh2; 
typedef boost::shared_ptr<c_triangle_mesh2> triangle_mesh2_ptr; 
typedef std::vector<triangle_mesh2_ptr> triangle_meshes2_array; 

class c_scene 
{
	
public:
	c_scene(triangle_meshes2_array& meshes)
		: m_meshes(meshes) 
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

private:
	triangle_meshes2_array m_meshes; 
	
};

typedef boost::shared_ptr<c_scene> scene_ptr; 

#endif // __scene_h__
