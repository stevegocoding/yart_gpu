#include "scene.h"
#include "triangle_mesh.h"

size_t c_scene::get_num_tri_total() const
{
	size_t total = 0; 
	for (size_t i = 0; i < m_meshes.size(); ++i)
	{
		total += m_meshes[i]->get_num_faces();
	}
	return total; 
}