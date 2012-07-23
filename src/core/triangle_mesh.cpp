#include "triangle_mesh.h"

c_triangle_mesh::c_triangle_mesh(const vertices_array& verts, 
	const normals_array& normals, 
	const tangents_array& tangents, 
	const uvs_array& uvs, 
	const face_indices_array& indices) 
{
	assert(verts.size() > 0); 
	assert(indices.size() > 0);

	m_num_faces = indices.size(); 
	m_num_verts = verts.size(); 

	m_has_normal = (normals.size() > 0) ? true : false; 
	m_has_tangent = (tangents.size() > 0) ? true : false; 
	m_has_uv = (uvs.size() > 0) ? true : false; 

	m_verts.reserve(m_num_verts); 
	m_faces_indices.reserve(indices.size());
	std::copy(verts.begin(), verts.end(), back_inserter(m_verts)); 
	std::copy(indices.begin(), indices.end(), back_inserter(m_faces_indices));

	if (m_has_normal)
	{
		m_normals.reserve(m_num_verts); 
		std::copy(normals.begin(), normals.end(), back_inserter(m_normals)); 
	}

	if (m_has_tangent)
	{
		m_tangents.reserve(m_num_verts); 
		std::copy(tangents.begin(), tangents.end(), back_inserter(m_tangents));
	}

	if (m_has_uv)
	{
		m_uvs.reserve(m_num_verts);
		std::copy(uvs.begin(), uvs.end(), back_inserter(m_uvs)); 
	}
}

//////////////////////////////////////////////////////////////////////////

c_triangle_mesh2::c_triangle_mesh2(const verts_pos_array pos[3], 
	const verts_normal_array normals[3], 
	const verts_uv_array uvs[3], 
	size_t num_faces,
	size_t num_verts)
	: m_num_faces(num_faces)
	, m_num_verts(num_verts)
{
	m_has_normal = (normals[0]) ? true : false; 
	m_has_uv = (uvs[0]) ? true : false; 

	for (int i = 0; i < 3; ++i)
	{
		m_verts[i] = pos[i]; 
		m_normals[i] = normals[i];
		m_uvs[i] = uvs[i]; 
	}
}

