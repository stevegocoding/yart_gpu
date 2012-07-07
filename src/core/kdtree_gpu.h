#ifndef __kdtree_gpu_h__
#define __kdtree_gpu_h__

#pragma once 

#include "kdtree_kernel_data.h"
#include "cuda_mem_pool.h"

// ---------------------------------------------------------------------
/*
/// \brief	GPU-based kd-tree implementation abstract base class.
/// 		
/// 		This class provides a GPU-based kd-tree construction algorithm that can be adapted
/// 		for different types of kd-trees (e.g. points or triangles as objects) within
/// 		subclasses. However, this extension possibility is somewhat limited due to problems
/// 		with the parallel implementation and might need some reworking to allow more types of
/// 		primitives.
/// 		
/// 		The implementation is based on the work of \ref lit_zhou "[Zhou et al. 2008]".
/// 		
/// 		This class uses pointers to the concrete elements to store in the kd-tree. These
/// 		pointers are indices for the array of elements. Within the construction algorithm,
/// 		the elements are in most cases hidden behind their AABBs. But at some places, e.g.
/// 		for splitting the nodes, there has to be special treatment depending on the primitive
/// 		type. Opposed to \ref lit_zhou "[Zhou et al. 2008]" I limit this special treatment
/// 		quite a bit to allow both point kd-trees and triangle kd-trees to be a subclass of
/// 		this class. This was done by introducing the number of \em element \em points, which
/// 		can be 1 or 2, depending on the primitive type. It is used for construction puropses
/// 		only.
/// 		
/// 		If the number of element points is 1, only one at most four-dimensional point can be
/// 		used per element. This is enough for a point kd-tree, where each element is just a
/// 		three-dimensional point. Similarly two four-dimensional points can be stored per
/// 		element, when the number of element points is set to 2. This is enough for triangle
/// 		kd-trees as the construction process mainly requires the AABB of the triangles.
/// 		Special handling like split clipping can be sourced out to the corresponding
/// 		subclass. This might be extended to other primtive objects as long as they can be
/// 		represented with at most two element points. 
/// 
*/ 
// ---------------------------------------------------------------------

class c_kdtree_gpu
{
	
public:

	c_kdtree_gpu(size_t num_input_elems, uint32 num_elems_points, float3 root_aabb_min, float3 root_aabb_max);
	


	// ---------------------------------------------------------------------
	/*
	/// \brief	Constructs the kd-tree for the elements supplied by subclasses.
	/// 		
	/// 		The process starts by requesting auxiliary structures (in most cases lists of nodes).
	/// 		After that, the root node is created using AddRootNode(). This step has to be defined
	/// 		by subclasses. Hence the concrete elements have to be supplied by the subclasses.
	/// 		Subsequently, large nodes are processed within the large node stage until no more
	/// 		large nodes are available. Then, within the small node stage, all small nodes are
	/// 		processed.
	/// 		
	/// 		In both stages, all final nodes are added to a final node list that represents the
	/// 		final kd-tree information. As node ordering in this final node list is somewhat
	/// 		chaotic, a final step is added to layout the tree in a new, more cache friendly way.
	/// 		This is done using a two tree traversals. To avoid high memory consumption, all
	/// 		auxiliary structures used for construction, e.g. lists of intermediate nodes or the
	/// 		chunk list, are destroyed before this method returns.  
	*/ 
	// ---------------------------------------------------------------------
	bool build_tree();

	

protected: 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Adds the root node to the given node list.
	///
	///			This method has to be specified by subclasses. It's responsible for creating and
	///			adding the kd-tree root node to the given node list. The concrete implementation
	///			has to copy all required data (see KDNodeList) for the root node into the first
	///			node list entry. This is usually a host to device copy operation.
	///
	///			Regarding AABBs: It is assumed that the subclass will only initialize the inherited
	///			root node bounds. Furthermore the subclass is responsible for calculating and
	///			initializing the element points for all root node elements.
	*/ 
	// ---------------------------------------------------------------------
	virtual void add_root_node(c_kd_node_list *node_list) = 0; 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Initializes auxiliary structures and data.
	///
	///			This method is called right before BuildTree() performs the actual construction. It's
	///			default implementation is important and subclasses should not forget to call it. 
	*/ 
	// ---------------------------------------------------------------------
	virtual void pre_build();

	virtual void post_build(); 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Fills chunk list #m_pChunkList with chunks for the given node list.
	///
	///			The creation is only performed when the current chunk list #m_pChunkList is not
	///			created for the given node list \a pList. Structurally it might be better to source
	///			this out into the KDChunkList type and to allow multiple chunk lists. However, I opted
	///			to use a single chunk list as I wanted to reduce memory and time requirements. 
	*/ 
	// ---------------------------------------------------------------------
	void create_chunk_list(c_kd_node_list *node_list);
	
	// Currently active nodes list.
	c_kd_node_list *m_active_node_list; 

	// Node list for next pass of algorithm.
	c_kd_node_list *m_next_node_list;
	
	// Chunk list used for node AABB calculation and element counting.
	c_kd_chunk_list *m_chunk_list;

	// Root node bounding box minimum supplied by constructor. Not neccessarily tight.
	float3 m_root_aabb_min;
	// Root node bounding box maximum supplied by constructor. Not neccessarily tight.
	float3 m_root_aabb_max;


private:

	// Implements the large node stage as described in the paper.
	void large_node_stage();

	// Implements the small node stage as described in the paper.
	void small_node_stage(); 

	// Processes large nodes in the current active list.
	void process_large_nodes(uint32 *d_final_list_idx_active);

	// Performs preorder traversal of the final tree to generate a final node list. 
	void preorder_traversal();


	// The final node list
	c_kdtree_data *m_kd_data; 

	// Input element count
	size_t m_num_input_elements;
	size_t m_num_element_points; 

	// Final internel node list 
	c_kd_final_node_list *m_final_list; 

	// Small node list
	c_kd_node_list *m_small_node_list; 

	// List the chunk list was build for. Avoids useless rebuilding.
	c_kd_node_list *m_current_chunk_list_src; 

	// Split candidate list. Stores split candidates for root small nodes.
	c_kd_split_list *m_split_list;

	// Small root parents stored here as m_pListNode indices. This is used to
	// update the parent indices *when* the small roots are added to m_pListNode.
	// Without this there is no connection between the small roots and their
	// parents. The MSB of each value is used to distinguish between left (0) and
	// right (1) child.
	uint32 *d_small_root_parents;

	// Single value buffer used for return values and other stuff.
	c_cuda_memory<uint32> d_temp_val;

	
	// Settings
	float m_empty_scene_ratio;
	uint32 m_small_nodes_max;
	// Conservative global estimate for KNN query radius. All queries will have at most this query radius.
	float m_max_query_radius;
	// std::vector<KDTreeListener*> m_vecListeners; 
	 
	
};

#endif // __kdtree_gpu_h__
