#ifndef __triangle_mesh_h__
#define __triangle_mesh_h__

#pragma once 

#include <vector>
#include <iterator>
#include <boost/shared_array.hpp>
#include "vector3f.h"
#include "point3f.h"

struct uv
{
	uv(float u, float v)
	{uvs[0] = u; uvs[1] = v;}
	float uvs[2];
};

struct tri_indices
{
	tri_indices(unsigned int i1, unsigned int i2, unsigned int i3)
	{ indices[0] = i1; indices[1] = i2; indices[2]; }
	unsigned int indices[3]; 
};

typedef std::vector<point3f> vertices_array; 
typedef std::vector<vector3f> normals_array; 
typedef std::vector<vector3f> tangents_array;
typedef std::vector<uv> uvs_array; 
typedef std::vector<tri_indices> face_indices_array;

class c_triangle_mesh
{
public: 
	c_triangle_mesh(const vertices_array& verts, 
		const normals_array& normals, 
		const tangents_array& tangents,
		const uvs_array& uvs,
		const face_indices_array& indices); 

	size_t get_num_verts() const 
	{ 
		return m_num_verts;
	}
	
	size_t get_num_faces() const
	{
		return m_num_faces; 
	}

	bool has_normal() const 
	{
		return m_has_normal; 
	}

	bool has_tangent() const
	{
		return m_has_tangent; 
	}

	bool has_uvs() const 
	{
		return m_has_uv;
	}
	
private: 
	vertices_array m_verts;
	normals_array m_normals;
	tangents_array m_tangents; 
	uvs_array m_uvs; 
	face_indices_array m_faces_indices; 

	size_t m_num_verts;
	size_t m_num_faces; 

	bool m_has_normal;
	bool m_has_tangent; 
	bool m_has_uv;

};

//////////////////////////////////////////////////////////////////////////

typedef boost::shared_array<point3f> verts_pos_array;
typedef boost::shared_array<normal3f> verts_normal_array;
typedef boost::shared_array<vector3f> verts_uv_array;

class c_triangle_mesh2
{
	
public:
	c_triangle_mesh2(const verts_pos_array pos[3],
					 const verts_normal_array normals[3], 
					 const verts_uv_array uvs[3],
					 size_t num_faces,
					 size_t num_verts);

	size_t get_num_faces() const { return m_num_faces; }

	// Return an array of points, where each point is the i-th triangle vertex for some triangle.
	point3f* get_verts(size_t i) { assert(i < 3); return m_verts[i].get(); }

	// Return an array of normals, where each normal is the i-th triangle normal for some triangle. 
	normal3f* get_normals(size_t i) { assert(i < 3); return m_normals[i].get(); }

	vector3f* get_texcoords(size_t i) { assert(i < 3); return m_uvs[i].get(); }

	bool has_normal() const 
	{
		return m_has_normal; 
	}

	bool has_uvs() const 
	{
		return m_has_uv;
	}

private:
	
	// Triangle vertex arrays, stored as three arrays of #m_numTris points each.
	verts_pos_array m_verts[3];

	// Triangle normal arrays, stored as three arrays of #m_numTris normals each.
	verts_normal_array m_normals[3]; 

	// Texture coordinate arrays, stored as three arrays of #m_numTris UVW-vectors each.
	// So at most three-dimensional textures are supported. 
	verts_uv_array m_uvs[3]; 

	size_t m_num_verts;
	size_t m_num_faces; 

	bool m_has_normal;
	bool m_has_tangent; 
	bool m_has_uv;	
};


#endif // __triangle_mesh_h__
