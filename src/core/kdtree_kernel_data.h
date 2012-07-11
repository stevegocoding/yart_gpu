

#ifndef __kdtree_kernel_data_h__
#define __kdtree_kernel_data_h__

#pragma once

#include <assert.h>
#include "cuda_defs.h"

/// Maximum chunk size used for chunk lists. Should be power of 2.
#define KD_CHUNKSIZE		256


// Element mask type used for element bit masks in small node stage. Currently up to 64 bit.
typedef unsigned long long elem_mask_t;


// ---------------------------------------------------------------------
/*
/// \brief	kd-tree node list data structure for intermediate node lists.
/// 		
/// 		This structure is used during kd-tree construction and is quite memory consuming. It
/// 		stores all required information all steps of the construction process. For node
/// 		bounds we distinguish tight and inherited bounds. Tight bounds result from computing
/// 		the exact AABB for all elements contained. Inherited bounds are generated when
/// 		splitting into child nodes. They are usually less tight. This distinction enables
/// 		empty space cutting. 
*/ 
// ---------------------------------------------------------------------

struct c_kd_node_list
{
#ifdef __cplusplus 
public:
	
	// ---------------------------------------------------------------------
	/*
	/// \brief	Initializes device memory.
	/// 		
	/// 		Provided maximum numbers should not be too low to avoid multiple resizes of the
	/// 		corresponding buffers. 
	/// 
	///
	/// \param	_maxNodes			The initial maximum number of nodes. 
	/// \param	_maxElems			The initial maximum number of elements. 
	/// \param	_numElementPoints	Number of element points. See KDTreeGPU for a description of
	///								this parameter. 
	*/ 
	// ---------------------------------------------------------------------
	void initialize(uint32 _max_nodes, uint32 _max_elems, uint32 _num_elem_points = 2);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Appends other node list to this node list.
	/// 		
	/// 		Copies data from given node list and resizes buffers if required. 
	*/ 
	// ---------------------------------------------------------------------
	void append_list(c_kd_node_list *nodes_list, bool append_data);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Resize node related device memory.
	///
	///			To prevent frequently resizes, the new maximum #maxNodes is chosen to be at least
	///			twice as large as the previous #maxNodes.
	*/ 
	// ---------------------------------------------------------------------
	void resize_node_data(uint32 required);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Resize element related device memory.
	/// 		
	/// 		To prevent frequently resizes, the new maximum #maxElems is chosen to be at least
	/// 		twice as large as the previous #maxElems. 
	*/ 
	// ---------------------------------------------------------------------
	void resize_elem_data(uint32 required);

	bool is_empty() const;

	void clear();

	void free_memory();

#endif

	// Number of nodes in this list.
	uint32 num_nodes;
	// Maximum number of nodes that can be stored.
	uint32 max_nodes;
	// Next free spot for new element data in ENA list. Aligned to allow coalesced access.
	uint32 next_free_pos;
	// Maximum number of elements that can be stored.
	uint32 max_elems; 
	// Number of element specific points in this list. Can be 1 or 2. See KDTreeGPU. 
	uint32 num_elem_points;

	// First element index address in ENA for each node (device memory).
	uint32 *d_first_elem_idx; 
	// Number of elements for each node (device memory).
	uint32 *d_num_elems_array; 
	// Tight AABB minimum for each node (device memory).
	float4 *d_aabb_tight_min; 
	// Tight AABB maximum for each node (device memory).
	float4 *d_aabb_tight_max; 
	// Inherited AABB minimum for each node (device memory).
	float4 *d_aabb_inherit_min; 
	// Inherited AABB maximum for each node (device memory).
	float4 *d_aabb_inherit_max;
	// Node levels (device memory). Starting with 0 for root.
	uint32 *d_node_level;

	// Split information
	
	uint32 *d_split_axis;
	
	float *d_split_pos; 

	uint32 *d_child_left;
	
	uint32 *d_child_right; 

	// Element information
	
	// ---------------------------------------------------------------------
	/*
	/// \brief	Element node association (ENA) list.
	///			
	///			Keeps track of which elements are assigned to each node. Element indices are stored
	///			contiguously for each node. The first element index address for node i is given by
	///			#d_idxFirstElem[i]. There can be holes between adjacent node element indices
	///			as the first element index address is aligned to improve performance.
	///
	/// \note	This list usually contains a single element (e.g. triangle) multiple times. 
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_node_elems_list; 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Element specific points 1.  Same order as ENA.
	///
	///			Allowed are at most two points per element. E.g. for triangle kd-trees, this would
	///			be the minimum of the triangle's bounding box.
	/// \see	#numElementPoints, KDTreeGPU
	*/ 
	// ---------------------------------------------------------------------
	float4 *d_elem_point1;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Element specific points 2. Same order as ENA.
	///
	///			Allowed are at most two points per element. E.g. for triangle kd-trees, this would
	///			be the maximum of the triangle's bounding box.
	/// \see	#numElementPoints, KDTreeGPU
	*/ 
	// ---------------------------------------------------------------------
	float4 *d_elem_point2;

	
	// Small node only information. This is *not* appended to final node list and
	// only valid in small node stage. 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Small root node index in small node list for each node.
	///
	///			Used to remember the corresponding small root parent for a given
	///			small node. The small root parent is the original small node list
	///			node. For each such node we precompute element masks and possible
	///			splits. Using this array we can combine that information with
	///			node specific data. See #d_elemMask.
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_small_root_idx; 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Element set stored as bit mask for each node (relative to small root node).
	///
	///			For each small root node, that is for each initial small node, this
	///			element mask is precomputed by mapping all node elements to the bits of
	///			the bit mask. As the number of elements per small node is restricted by
	///			::KD_SMALLNODEMAX, this mapping is possible. So initially, all #d_numElems[i]
	///			bits starting from bit 0 are set for the i-th small node.
	///
	///			For subsequent small nodes, that is for each direct or indirect child j of an initial
	///			small node i, #d_elemMask[j] is relative to the corresponding #d_elemMask[i] of the
	///			initial small node list. So if any of the bits set in #d_elemMask[i] is unset in
	///			#d_elemMask[j], the corresponding element is not contained in the child j.
	*/ 
	// ---------------------------------------------------------------------
	elem_mask_t *d_elem_mask; 
	
};

//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
/// \brief	Structure to hold all final nodes generated during kd-tree construction.
///
///			Separated from the default node list structure KDNodeList to save memory since only a 
///			subset of the node list data is required for the generation of the final kd-tree
///			layout. Note that this is a structure of arrays and not an array of structures.
*/ 
// ---------------------------------------------------------------------

struct c_kd_final_node_list
{
#ifdef __cplusplus
public:

	// ---------------------------------------------------------------------
	/*
	/// \brief	Initializes device memory. 
	///
	///			Provided maximum numbers should not be too low to avoid multiple resizes of
	///			the corresponding buffers.
	*/ 
	// ---------------------------------------------------------------------
	void initialize(uint32 _max_nodes, uint32 _max_elems);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Appends a node list to this final node list.
	///
	///			Copies data from given node list and resizes buffers if required.
	*/ 
	// ---------------------------------------------------------------------
	void append_list(c_kd_node_list *node_list, bool append_ena, bool has_inherited_bounds);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Resize node related device memory.
	///
	///			To prevent frequently resizes, the new maximum #maxNodes is chosen to be at least
	///			twice as large as the previous #maxNodes.
	*/ 
	// ---------------------------------------------------------------------
	void resize_node_data(uint32 required); 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Resize element related device memory.
	/// 		
	/// 		To prevent frequently resizes, the new maximum #maxElems is chosen to be at least
	/// 		twice as large as the previous #maxElems.  
	*/ 
	// ---------------------------------------------------------------------
	void resize_elem_data(uint32 required);

	
	bool is_empty() const { return (num_nodes == 0); }
	
	void clear();

	void free_memory(); 

#endif

	/// Current number of nodes.
	uint32 num_nodes;
	/// Maximum number of nodes that can be stored.
	uint32 max_nodes;
	/// Next free element position. Aligned to allow coalesced access.
	uint32 next_free_pos;
	/// Maximum number of elements that can be stored.
	uint32 max_elems;

	/// First element index address in ENA for each node (device memory).
	uint32* d_first_elem_idx;
	/// Number of elements for each node (device memory).
	uint32* d_num_elems;
	/// Node bounds minimum for radius/center calculation (device memory). Might not be tight.
	float4 *d_aabb_min;
	/// Node bounds minimum for radius/center calculation (device memory). Might not be tight.
	float4 *d_aabb_max;
	/// Node levels (device memory). Starting with 0 for root.
	uint32* d_node_level;

	/// Split axis for each node (device memory).  Can be 0, 1 or 2. However, this is only valid for inner nodes.
	uint32* d_split_axis;
	/// Split position for each node (device memory). However, this is only valid for inner nodes.
	float* d_split_pos;
	/// Left (below) child node index for each node (device memory). Only valid for inner nodes.
	uint32* d_child_left;
	/// Right (above) child node index for each node (device memory). Only valid for inner nodes.
	uint32* d_child_right;

	/// \brief	Element node association (ENA) list.
	///			
	///			Keeps track of which elements are assigned to each node. Element indices are stored
	///			contiguously for each node. The first element index address for node i is given by
	///			#d_idxFirstElem[i]. There can be holes between adjacent node element indices
	///			as the first element index address is aligned to improve performance.
	uint32* d_elem_node_assoc; 

	
};

//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
/// \brief	Chunk list representation used for kd-tree construction.
/// 		
/// 		This list is used for node bounding box computation and node element counting.
///			The chunks partition the elements associated to the nodes of a given node list.
///			A chunk can only be assigned to a single node. Each node might be represented
///			by multiple chunks, depending on the element count of the node. That's because
///			each chunk can contain at most ::KD_CHUNKSIZE elements.
///
///			The chunks are mapped to thread blocks of CUDA threads. At the beginning,
///			chunk information like node index and first element index address are read
///			into shared memory. Then all threads can read this data and use it to
///			process it's element, e.g. for element AABB computation. 
*/ 
// ---------------------------------------------------------------------

struct c_kd_chunk_list
{
#ifdef __cplusplus

public:
	
	void initialize(uint32 _max_chunks);

	bool is_empty() const { return (num_chunks == 0); }

	void clear();

	void free_memory(); 
	
#endif // __cplusplus

	uint32 num_chunks;

	uint32 max_chunks;

	uint32 *d_node_idx; 

	uint32 *d_first_elem_idx; 

	uint32 *d_num_elems; 

	float4 *d_aabb_min; 

	float4 *d_aabb_max; 

};

////////////////////////////////////////////////////////////////////////// 

// ---------------------------------------------------------------------
/*
/// \brief	Stores split information used in small node stage.
/// 		
/// 		Specifically all split candidates for the small root nodes are stored. Split axis is
/// 		not stored. Instead it is calculated from the split index. This is possible as we
/// 		know the order the splits are stored. Note that the number of small root nodes is not
/// 		stored explicitly as it is equivalent to the number of nodes in the small node list. 
///
///			For each split i we precalculate element masks #d_maskLeft[i] and #d_maskRight[i] that
///			show which elements of the corresponding small root node would get into the left and
///			right child respectively. These element masks can be combined with the element mask
///			of the small root node (or more general: a small parent node) using a Booealn AND
///			operation to get the element masks for the child nodes.
*/ 
// ---------------------------------------------------------------------

struct c_kd_split_list
{
#ifdef __cplusplus
public:

	// ---------------------------------------------------------------------
	/*
	@param: List of small root nodes. These is the list of small node just
		after finishing the large node stage.  
	*/ 
	// ---------------------------------------------------------------------
	void initialize(c_kd_node_list *small_roots);

	void free_memory();
	
	
#endif // __cplusplus

	uint32 *d_first_split_idx;

	uint32 *d_num_splits;
	
	float *d_split_pos_array;  

	// ---------------------------------------------------------------------
	/*
	/// \brief	Split information for each split (device memory).
	///
	///			The tree least significant bits are used. Bits 0 and 1 store the split
	///			axis and bit 3 stores whether this is a minimum or maximum split. The
	///			distinction is important for non-degenerating element AABBs, e.g. for
	///			triangle kd-trees. It determines on which side the split triangle is
	///			placed.
	/// \see	::kernel_InitSplitMasks() 
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_split_info_array;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Element set on the left of the splitting plane for each split (device memory).
	///
	///			A set bit signalizes that the corresponding element would get into the left
	///			child node when splitting according to this split.
	*/ 
	// ---------------------------------------------------------------------
	elem_mask_t *d_mask_left_array; 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Element set on the right of the splitting plane for each split (device memory).
	///
	///			A set bit signalizes that the corresponding element would get into the right
	///			child node when splitting according to this split.
	*/ 
	// --------------------------------------------------------------------- 
	elem_mask_t *d_mask_right_array;
	
};

//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
/// \brief	Final representation of kd-tree after construction.
///
///			The KDFinalNodeList representation that is generated during the construction process
///			is not sufficient for the needs of fast traversal algorithms. It contains too much
///			information without any real compression. Furthermore it does nothing
///			to improve cache performance.
///
///			\par Reorganizing Traversal Information
///			This structure by contrast avoids storing useless data and compresses traversal
///			related information to reduce memory accesses. To allow better cache performance,
///			traversal related information is reorganized in #d_preorderTree using the
///			following format: 
///
///			\code root_idx | root_info | left_idx | left_info | left_subtree | right_idx | right_info | right_subtree \endcode
///
///			where
///			\li \c root_idx:		Root node index.
///			\li \c root_info:		Parent information for root, see below. Takes two \c uint.
///			\li \c left_idx:		Index of the left child node.
///			\li	\c left_info:		Parent information for left child node, see below. Takes two \c uint.
///			\li \c left_subtree:	All further nodes in the left subtree.
///
///			It is important to note that the whole left subtree is stored before the right child 
///			and its subtree. This helps improving cache performance. Indices are relative to the
///			order of the other arrays of this structure, e.g. #d_nodeExtent. They are enriched
///			with leaf information: The MSB is set, if the node is a leaf. Else the MSB is not set.
///			This improves leaf detection performance because it avoids reading child information.
///
///			The above format applies only for inner nodes. For leafs, instead of the parent
///			information, element count and element indices are stored. The element indices are
///			relative to the underlying data, e.g. triangle data or point data. Hence we have
///			the subsequent format for leafs:
///
///			\code leaf_idx | leaf_count | leaf_elmIndices \endcode
///
///			\par Parent information
///			As noted above, inner node representation contains a \em parent \em information. This is
///			a compressed form of what is needed during traversal. It takes two \c uint, hence two
///			elements of #d_preorderTree. It is organized the following way:
///
///			\li First Entry: [MSB] Split axis (2 bits) + Custom bits (2 bits) + Child above address (28 bits) [LSB]
///			\li Second Entry: Split position (\c float stored as \c uint).
///
///			Child above address is 0 for leaf nodes. Else it is the address of the right child
///			node in #d_preorderTree. The left child node address is not stored explicitly as it 
///			can be computed. Split axis takes 2 most significant bits. 0 stands for x-axis, 
///			1 for y-axis and 2 for z-axis. The custom bits are used to mark nodes. See
///			KDTreeGPU::SetCustomBits().
*/ 
// ---------------------------------------------------------------------

struct c_kdtree_data
{
#ifdef __cplusplus
	
	// ---------------------------------------------------------------------
	/*
	/// \brief	Initializes device memory and reads out some information from given node list.
	///
	///			Currently, only element counts (#d_numElems) and left/right child indices (#d_childLeft, 
	///			#d_childRight) are transfered. Root node AABB information is required to avoid
	///			reading it back from GPU memory. 
	*/ 
	// ---------------------------------------------------------------------
	void initialize(c_kd_final_node_list *list, float3 aabb_min, float3 aabb_max);

	bool is_empty() const { return num_nodes == 0; }

	void free_memory();  
	
#endif

	uint32 num_nodes;
	/// Root node bounding box minimum.
	float3 aabb_root_min; 
	/// Root node bounding box maximum
	float3 aabb_root_max;

	// Number of elements (e.g. triangles) in each node (device memory).
	uint32 *d_num_elems;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Address of each nodes in #d_preorderTree.
	///
	///			Stored in the same order as #d_numElems or #d_childLeft. Associates these "unordered"
	///			arrays with the "ordered" array #d_preorderTree, so that we can modify the latter when
	///			only having data in "unordered" array, e.g. when setting custom bits. 
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_node_addresses_array;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Node extents (device memory).
	///
	///			\li \c xyz: Node center.
	///			\li \c w:	Node "radius", that is half the diagonal of the node's bounding box.
	*/ 
	// ---------------------------------------------------------------------
	float4 *d_node_extent;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Left child node indices (device memory). 
	/// 
	///			Stored due to the fact that we cannot get child information when iterating
	///			over the nodes instead of traversal using #d_preorderTree (e.g. for query radius estimation).
	///			Using #d_nodeAddresses, it might be possible to remove it. I still use it for convenience.
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_child_left;

	// ---------------------------------------------------------------------
	/*
	/// \brief	Right child node indices (device memory). 
	/// 
	///			Stored due to the fact that we cannot get child information when iterating
	///			over the nodes instead of traversal using #d_preorderTree (e.g. for query radius estimation).
	///			Using #d_nodeAddresses, it might be possible to remove it. I still use it for convenience.
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_child_right;

	/// Size of the preorder tree representation #d_preorderTree in bytes.
	uint32 size_tree; 

	// ---------------------------------------------------------------------
	/*
	/// \brief	Preorder tree representation. 
	///
	///			The format description can be found in the detailed description of this structure. 
	*/ 
	// ---------------------------------------------------------------------
	uint32 *d_preorder_tree;
	

};

#endif // __kdtree_kernel_data_h__