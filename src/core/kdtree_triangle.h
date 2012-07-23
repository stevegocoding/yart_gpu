#ifndef __kdtree_triangle_h__
#define __kdtree_triangle_h__

#pragma once

#include "kdtree_gpu.h"

struct c_triangle_data;
class c_kdtree_triangle : public c_kdtree_gpu
{
public:
	c_kdtree_triangle(const c_triangle_data& tri_data); 

	// For DEBUG 
	c_kd_chunk_list* get_chunks_list() const { return m_chunk_list; }
	c_kd_node_list* get_active_list() const { return m_active_node_list; } 

protected:
	
	// ---------------------------------------------------------------------
	/*
	/// \brief	Adds the root node to the given node list.
	///
	///			This method uses the provided TriangleData object (see KDTreeTri()) to copy the required
	///			information for the root node from host to device memory. Furthermore the element points
	///			are set by computing AABBs for all triangles. 
	*/ 
	// ---------------------------------------------------------------------
	virtual void add_root_node(c_kd_node_list *node_list);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Performs split clipping for large nodes.
	/// 		
	/// 		For triangles, split clipping can be used to reduce the actual triangle AABB within
	/// 		child nodes after node splitting. This was suggested by Havran, "Heuristic Ray
	/// 		Shooting Algorithms", 2000. To parallelize the process, this method computes a chunk
	/// 		list for child node list.  
	*/ 
	// ---------------------------------------------------------------------
	virtual void perform_split_clipping(c_kd_node_list *parent_list, c_kd_node_list *child_list);

private: 
	const c_triangle_data *m_tri_data;
};

#endif // __kdtree_triangle_h__
