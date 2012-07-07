#ifndef __kdtree_triangle_h__
#define __kdtree_triangle_h__

#pragma once

#include "kdtree_gpu.h"



struct c_triangle_data;
class c_kdtree_triangle : public c_kdtree_gpu
{
	

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

private:
	
	c_triangle_data *m_tri_data;
};

#endif // __kdtree_triangle_h__
