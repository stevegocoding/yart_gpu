#include "kernel_data.h"
#include "kdtree_kernel_data.h" 
#include "cuda_utils_device.h"
#include "cuda_mem_pool.h"
#include "functor_device.h"
#include "cuda_rng.h"

#include "kdtree_debug.h"


extern std::ofstream chunk_ofs; 
extern std::ofstream active_ofs; 
extern std::ofstream others_ofs; 

/// \brief	kd-tree traverse cost. Used to stop the splitting at some point where traversal would
///			lead to a higher cost than splitting a given node.
#define KD_COST_TRAVERSE	3.0f

cudaDeviceProp device_props;

__constant__ float k_empty_space_ratio; 
__constant__ uint32 k_small_node_max; 
__constant__ float k_max_query_radius;

// ---------------------------------------------------------------------
/*
/// \brief	Slim version of KDNodeList for node AABB related tasks.
///
///			To avoid parameter space overflows.
*/ 
// ---------------------------------------------------------------------
struct kd_node_list_aabb
{
#ifdef __cplusplus
	kd_node_list_aabb(const c_kd_node_list& src)
		: num_nodes(src.num_nodes)
		, d_first_elem_idx(src.d_first_elem_idx)
		, d_num_elems(src.d_num_elems_array)
		, d_node_level(src.d_node_level)
		, d_aabb_tight_min(src.d_aabb_tight_min)
		, d_aabb_tight_max(src.d_aabb_tight_max)
		, d_aabb_inherit_min(src.d_aabb_inherit_min)
		, d_aabb_inherit_max(src.d_aabb_inherit_max)
		, d_elem_node_assocs(src.d_node_elems_list) 
	{

	}
#endif

	// Number of nodes in this list.
	uint32 num_nodes;

	// First element index address in ENA for each node (device memory).
	uint32 *d_first_elem_idx; 
	// Number of elements for each node (device memory).
	uint32 *d_num_elems;
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

	uint32 *d_elem_node_assocs;
	
};

struct kd_small_node_list
{ 
	uint32 num_nodes; 

	uint32 *d_num_elems;

	uint32 *d_node_level;
	
	// Tight AABB minimum for each node (device memory).
	float4 *d_aabb_tight_min; 
	// Tight AABB maximum for each node (device memory).
	float4 *d_aabb_tight_max; 

	/// See KDNodeList::d_splitAxis.
	uint32 *d_split_axis;
	/// See KDNodeList::d_splitPos.
	float *d_split_pos;

	/// See KDNodeList::d_idxSmallRoot.
	uint32 *d_idx_small_root; 
	/// See KDNodeList::d_elemMask.
	elem_mask_t *d_elem_mask;
	
};

struct kd_ena_node_list
{
	uint32 num_nodes;
	
	uint32 *d_num_elems;

	uint32 *d_idx_first_elem;

	uint32 *d_idx_small_root; 

	elem_mask_t *d_elem_marks;
	
	uint32 *d_elem_node_assoc; 
}; 

// ---------------------------------------------------------------------
/*
/// \brief	Creates an empty leaf node in the final node list.
/// 		
/// 		Sets all required information for the empty node, that is node level, number of
/// 		elements, child addresses (both 0) and node bounding box. 
*/ 
// ---------------------------------------------------------------------
inline __device__ void device_create_empty_leaf(c_kd_final_node_list& final_list, uint32 idx_new, float aabb_min[3], float aabb_max[3], uint32 node_level)
{
	// d_idxFirstElem can stay undefined as we have no elements.
	final_list.d_num_elems[idx_new] = 0;
	final_list.d_node_level[idx_new] = node_level;
	// d_splitAxis, d_splitPos are undefined for leafs.
	final_list.d_child_left[idx_new] = 0;
	final_list.d_child_right[idx_new] = 0;
	final_list.d_aabb_min[idx_new] = make_float4(aabb_min[0], aabb_min[1], aabb_min[2], 0.f);
	final_list.d_aabb_max[idx_new] = make_float4(aabb_max[0], aabb_max[1], aabb_max[2], 0.f);
	// d_elemNodeAssoc: no changes required.	
}

// ---------------------------------------------------------------------
/*
/// \brief	Copies a final node list node.
/// 		
/// 		Used for empty space cutting to generate a copy of the actual node. This copy
/// 		represents the new, non-empty node. The empty node is generated using
/// 		dev_CreateEmptyLeaf(). 
*/ 
// ---------------------------------------------------------------------
inline __device__ void device_create_final_node_copy(c_kd_final_node_list& final_list, 
													uint32 idx_old, 
													uint32 idx_new, 
													float aabb_min[3], float aabb_max[3],
													uint32 node_level)
{
	// Both nodes use the same elements.
	final_list.d_first_elem_idx[idx_new] = final_list.d_first_elem_idx[idx_old];
	final_list.d_num_elems[idx_new] = final_list.d_num_elems[idx_old];
	final_list.d_node_level[idx_new] = node_level;
	// d_splitAxis, d_splitPos, d_childLeft, d_childRight are not yet known.
	final_list.d_aabb_min[idx_new] = make_float4(aabb_min[0], aabb_min[1], aabb_min[2], 0.f);
	final_list.d_aabb_max[idx_new] = make_float4(aabb_max[0], aabb_max[1], aabb_max[2], 0.f); 
}

inline __device__ void device_create_child(kd_node_list_aabb& next_list, uint32 idx_new, 
											float aabb_min[3], float aabb_max[3], uint32 node_level)
{
	next_list.d_node_level[idx_new] = node_level;
	next_list.d_aabb_inherit_min[idx_new] = make_float4(aabb_min[0], aabb_min[1], aabb_min[2], 0.0f);
	next_list.d_aabb_inherit_max[idx_new] = make_float4(aabb_max[0], aabb_max[1], aabb_max[2], 0.0f); 
}

// ---------------------------------------------------------------------
/*
/// \brief	Generates AABBs for chunks using parallel reduction.
/// 		
/// 		Chunk AABBs are generated by performing parallel reductions on the element AABBs
///			given in the node list.
///
/// \note	Required shared memory per thread block of size N: 8 * N bytes.
*/ 
// ---------------------------------------------------------------------
template <uint32 num_elems_pts>
__global__ void kernel_gen_chunk_aabb(c_kd_node_list node_list, c_kd_chunk_list chunk_list)
{
	uint32 chk = CUDA_GRID2DINDEX;		// Global block index

	__shared__ uint32 s_num_elems;
	__shared__ uint32 s_first_elem_idx; 
	if (threadIdx.x == 0)
	{
		s_num_elems = chunk_list.d_num_elems[chk];
		s_first_elem_idx = chunk_list.d_first_elem_idx[chk];
	}

	__syncthreads();

	// Copy values into shared memory.
	__shared__ float s_mem[KD_CHUNKSIZE];
	float3 aabb_min, aabb_max; 

	// Manual unrolling since automatic did not work.
	float v1, v2;
	
	if (num_elems_pts == 1)
	{
		// Use second shared buffer to avoid rereading.
		__shared__ float s_mem2[KD_CHUNKSIZE];
		
		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].x;
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].x;
		s_mem[threadIdx.x] = fminf(v1, v2);
		if (threadIdx.x >= s_num_elems)
			v1 = -M_INFINITY;
		if (threadIdx.x + blockDim.x >= s_num_elems)
			v2 = -M_INFINITY;
		s_mem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads();
		aabb_min.x = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());
		aabb_max.x = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem2, op_maximum<float>()); 	
		
		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].y; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].y; 
		s_mem[threadIdx.x] = fminf(v1, v2);
		if (threadIdx.x >= s_num_elems)
			v1 = -M_INFINITY; 
		if (threadIdx.x + blockDim.x >= s_num_elems)
			v2 = -M_INFINITY;
		s_mem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads(); 
		aabb_min.y = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());
		aabb_max.y = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem2, op_maximum<float>()); 

		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].z; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].z;
		s_mem[threadIdx.x] = fminf(v1, v2);
		if (threadIdx.x >= s_num_elems)
			v1 = -M_INFINITY; 
		if (threadIdx.x + blockDim.x >= s_num_elems)
			v2 = -M_INFINITY;
		s_mem2[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads(); 
		aabb_min.z = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());
		aabb_max.z = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem2, op_maximum<float>()); 
	}
	else // num_elems_pts == 2
	{
		// first elem point, min 
		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].x; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].x; 
		s_mem[threadIdx.x] = fminf(v1, v2);
		__syncthreads(); 
		aabb_min.x = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());

		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].y; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].y; 
		s_mem[threadIdx.x] = fminf(v1, v2);
		__syncthreads(); 
		aabb_min.y = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());

		v1 = M_INFINITY; v2 = M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x].z; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point1[s_first_elem_idx + threadIdx.x + blockDim.x].z; 
		s_mem[threadIdx.x] = fminf(v1, v2);
		__syncthreads(); 
		aabb_min.z = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_minimum<float>>(s_mem, op_minimum<float>());

		// second elem point, max 
		v1 = -M_INFINITY; v2 = -M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x].x; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x + blockDim.x].x; 
		s_mem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads(); 
		aabb_max.x = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem, op_maximum<float>());

		v1 = -M_INFINITY; v2 = -M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x].y; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x + blockDim.x].y; 
		s_mem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads(); 
		aabb_max.y = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem, op_maximum<float>());

		v1 = -M_INFINITY; v2 = -M_INFINITY;
		if (threadIdx.x < s_num_elems)
			v1 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x].z; 
		if (threadIdx.x + blockDim.x < s_num_elems)
			v2 = node_list.d_elem_point2[s_first_elem_idx + threadIdx.x + blockDim.x].z; 
		s_mem[threadIdx.x] = fmaxf(v1, v2);
		__syncthreads(); 
		aabb_max.z = device_reduce_fast<float, (uint32)(KD_CHUNKSIZE/2), op_maximum<float>>(s_mem, op_maximum<float>());
	}

	if (threadIdx.x == 0)
	{
		chunk_list.d_aabb_min[chk] = make_float4(aabb_min);
		chunk_list.d_aabb_max[chk] = make_float4(aabb_max);
	}
}

__global__ void kernel_can_cutoff_empty_space(c_kd_node_list active_list, uint32 axis, bool bmax,  uint32 *d_out_can_cutoff)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < active_list.num_nodes)
	{
		// Check for empty space on given side.
		float4 aabb_min_inherit = active_list.d_aabb_inherit_min[idx];
		float4 aabb_max_inherit = active_list.d_aabb_inherit_max[idx];
		float4 aabb_min_tight =  active_list.d_aabb_tight_min[idx];
		float4 aabb_max_tight =  active_list.d_aabb_tight_max[idx];
		float inherit_min = ((float*)&aabb_min_inherit)[axis];
		float inherit_max = ((float*)&aabb_max_inherit)[axis];
		float tight_min = ((float*)&aabb_min_tight)[axis];
		float tight_max = ((float*)&aabb_max_tight)[axis];

		float total = inherit_max - inherit_min;
		float empty_space_ratio = k_empty_space_ratio;

		uint32 can_cutoff = 0;
		if (!bmax)
		{
			// minimum check 
			float empty = tight_min - inherit_min;
			if (empty > total*empty_space_ratio)
				can_cutoff = 1; 
		}
		else
		{
			// maximum check 
			float empty = inherit_max - tight_max;
			if (empty > total*empty_space_ratio)
				can_cutoff = 1; 
		}
		
		d_out_can_cutoff[idx] = can_cutoff;
	}
}


// ---------------------------------------------------------------------
/*
/// \brief	Marks left and right elements in next list ENA.
/// 		
/// 		It is assumed that active list's elements were duplicated in the following way: The
/// 		ENA was copied to the first \c lstActive.nextFreePos elements and to the second \c
/// 		lstActive.nextFreePos elements.  

/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstActive			The active list. 
/// \param	lstChunks			The chunk list constructed for the active list. 
/// \param [in]		d_randoms	Uniform random numbers used to avoid endless split loops if
/// 							several elements lie within a splitting plane. There should be a
/// 							random number for each thread, i.e. for each element processed. 
/// \param [out]	d_outValid	Binary 0/1 array ordered as the ENA of the next list, i.e. the
/// 							left child valid flags are in the first half and the right child
/// 							valid flags are in the right half respectively.  
*/ 
// ---------------------------------------------------------------------
template <uint32 num_elem_pts>
__global__ void kernel_mark_left_right_elems(c_kd_node_list active_list, 
											c_kd_chunk_list chunks_list, 
											float *d_randoms, 
											uint32 *d_out_valid)
{
	uint32 chunk = CUDA_GRID2DINDEX;
	uint32 idx = threadIdx.x;
	
	__shared__ uint32 s_num_elems_chunk;
	__shared__ uint32 s_idx_node; 
	__shared__ uint32 s_idx_first_elem;
	__shared__ uint32 s_split_axis;
	__shared__ float s_split_pos;

	if (threadIdx.x == 0)
	{
		s_num_elems_chunk = chunks_list.d_num_elems[chunk];
		s_idx_node = chunks_list.d_node_idx[chunk];
		s_idx_first_elem = chunks_list.d_first_elem_idx[chunk];
		s_split_axis = active_list.d_split_axis[s_idx_node];
		s_split_pos = active_list.d_split_pos[s_idx_node];
	}

	__syncthreads();

	if (idx < s_num_elems_chunk)
	{
		uint32 idx_tna = s_idx_first_elem + idx; 
		uint32 tid = blockDim.x * CUDA_GRID2DINDEX + threadIdx.x;
		bool is_left = false; 
		bool is_right = false; 

		if (num_elem_pts == 2)	// Compile Time 
		{
			// Get bounds
			float bounds_min = ((float*)&active_list.d_elem_point1[idx_tna])[s_split_axis];
			float bounds_max = ((float*)&active_list.d_elem_point2[idx_tna])[s_split_axis];
			
			// Check on which sides the triangle is. It might be on both sides!
			if (d_randoms[tid] < 0.5f)
			{
				is_left = bounds_min < s_split_pos || (bounds_min == s_split_pos && bounds_min == bounds_max);	
				is_right = s_split_pos < bounds_max; 
			}
			else 
			{
				is_left = bounds_min < s_split_pos;
				is_right = s_split_pos < bounds_max || (bounds_min == s_split_pos && bounds_min == bounds_max); 
			}
		}
		else 
		{
			float val = ((float*)&active_list.d_elem_point1[idx_tna])[s_split_axis];
			// Cannot use the same criterion (i.e. < and <= or <= and <) for all points.
			// Else we would have the special case where all points lie in the splitting plane
			// and therefore all points land on a single side. This would result in an endless
			// loop in large node stage!
			if (d_randoms[tid] < 0.5f)
			{
				is_left = val < s_split_pos;
				is_right = s_split_pos <= val;
			}
			else 
			{
				is_left = val <= s_split_pos;
				is_right = s_split_pos < val;
			}
		}
		
		// left child
		d_out_valid[s_idx_first_elem + idx] = (is_left ? 1 : 0);
		
		// right child
		d_out_valid[active_list.next_free_pos + s_idx_first_elem + idx] = (is_right ? 1 : 0);
	}
}

// ---------------------------------------------------------------------
/*
/// \brief	Performs empty space cutting. 
///
///			For all nodes of the active list, for which empty space cutting on the given side
///			of their AABBs can be performed, the empty space is cut off. This is done by
///			generating new empty and non-empty nodes and inserting them into the final node list.
///			Furthermore the active list is updated to contain the new non-empty nodes. 

/// \param	lstActive					The active node list. Contains the nodes that are to be
/// 									subdivided. When empty space is cut off for some node, its
///										AABB and node level are updated accordingly.
/// \param	lstFinal					The final node list. Will be updated with the generated
///										empty and non-empty nodes.
/// \param	axis						The axis to check. 
/// \param	bMax						Whether to check maximum or minimum sides. 
/// \param [in]		d_canCutOff			Binary 0/1 array. Contains 1 for nodes where empty space
/// 									can be cut off. Generated by
/// 									kernel_CanCutOffEmptySpace(). 
/// \param [in]		d_cutOffsets		Cut offsets. This should be the result of a scan of \a
/// 									d_canCutOff. It will indicate the offset to use for
/// 									writing the generated empty and non-empty child nodes
/// 									into the final node list. 
/// \param	numCuts						Number of cuts. Can be obtained by reduction on \a
/// 									d_canCutOff. 
/// \param [in,out]	d_ioFinalListIndex	Will contain updated final node list indices for the
///										current active list nodes. That is, the generated non-empty
///										node for the i-th active list node can be found at the index
///										\a d_ioFinalListIndex[i].
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_empty_space_cutting(c_kd_node_list active_list, c_kd_final_node_list final_list, uint32 axis, bool bmax,
										uint32 *d_can_cutoff, uint32 *d_cut_offsets, uint32 num_cuts,
										uint32* d_final_list_idx)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < active_list.num_nodes && d_can_cutoff[idx])
	{
		float3 aabb_min_inherit = make_float3(active_list.d_aabb_inherit_min[idx]);
		float3 aabb_max_inherit = make_float3(active_list.d_aabb_inherit_max[idx]);
		float3 aabb_min_tight =  make_float3(active_list.d_aabb_tight_min[idx]);
		float3 aabb_max_tight =  make_float3(active_list.d_aabb_tight_max[idx]);
		
		float aabb_min_child[3] = {((float*)&aabb_min_inherit)[0], ((float*)&aabb_min_inherit)[1], ((float*)&aabb_min_inherit)[2]};
		float aabb_max_child[3] = {((float*)&aabb_max_inherit)[0], ((float*)&aabb_max_inherit)[1], ((float*)&aabb_max_inherit)[2]};		

		uint32 node_level_parent = active_list.d_node_level[idx];

		// Compute indices for left and right node in final node list.
		uint32 cut_offset = d_can_cutoff[idx];
		uint32 idx_parent = d_final_list_idx[idx];
		uint32 idx_left = final_list.num_nodes + cut_offset; 
		uint32 idx_right = final_list.num_nodes + num_cuts + cut_offset;
		float split_pos = (bmax ? ((float*)&aabb_max_tight)[axis] : ((float*)&aabb_min_tight)[axis]);

		if (!bmax)
		{
			// Below (left) is the empty node.
			aabb_max_child[axis] = split_pos;
			device_create_empty_leaf(final_list, idx_left, aabb_min_child, aabb_max_child, node_level_parent+1);
			
			// Above (right) is the tighter node.
			aabb_min_child[axis] = aabb_max_child[axis];
			aabb_max_child[axis] = ((float*)&aabb_max_inherit)[axis];
			device_create_final_node_copy(final_list, idx_parent, idx_right, aabb_min_child, aabb_max_child, node_level_parent+1);

			// Update active list node to describe the above node. Change inherited to be tighter.
			active_list.d_aabb_inherit_min[idx] = make_float4(aabb_min_child[0], aabb_min_child[1], aabb_min_child[2], 0.0f);
			active_list.d_aabb_inherit_max[idx] = make_float4(aabb_max_child[0], aabb_max_child[1], aabb_max_child[2], 0.0f);
			active_list.d_node_level[idx] = node_level_parent+1; 
		}
		else 
		{
			// Below (left) is the tighter node.
			aabb_max_child[axis] = split_pos;
			device_create_final_node_copy(final_list, idx_parent, idx_left, aabb_min_child, aabb_max_child, node_level_parent+1);

			// Update active list node to describe the above node. Change inherited to be tighter.
			active_list.d_aabb_inherit_min[idx] = make_float4(aabb_min_child[0], aabb_min_child[1], aabb_min_child[2], 0.0f);
			active_list.d_aabb_inherit_max[idx] = make_float4(aabb_max_child[0], aabb_max_child[1], aabb_max_child[2], 0.0f);
			active_list.d_node_level[idx] = node_level_parent+1; 

			aabb_min_child[axis] = aabb_max_child[axis];
			aabb_max_child[axis] = ((float*)&aabb_max_inherit)[axis];
			device_create_empty_leaf(final_list, idx_right, aabb_min_child, aabb_max_child, node_level_parent+1);
		}
		
		// Write split information to original node in final node list.
		final_list.d_split_axis[idx_parent] = axis; 
		final_list.d_split_pos[idx_parent] = split_pos;
		final_list.d_child_left[idx_parent] = idx_left;
		final_list.d_child_right[idx_parent] = idx_right;

		// Update final list index to point to the tighter node.
		d_final_list_idx[idx] = (bmax ? idx_left : idx_right);
	}
}

// ---------------------------------------------------------------------
/*
/// \brief	Splits large nodes in active list into smaller nodes.
/// 		
/// 		The resulting nodes are put in the next list, starting from index 0. Left (below)
/// 		nodes are written at the same indices as in active list. Right (above) nodes are
/// 		offsetted by the number of active list nodes.  
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_split_large_nodes(const c_kd_node_list active_list, kd_node_list_aabb next_list)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < active_list.num_nodes)
	{
		// Empty space cutting was performed. Therefore our inherited bounds can be
		// used directly as basis.

		float3 aabb_min_inherit = make_float3(active_list.d_aabb_inherit_min[idx]);
		float3 aabb_max_inherit = make_float3(active_list.d_aabb_inherit_max[idx]);
		
		// Find longest axis of new bounds.
		uint32 longest = 0;
		if (aabb_max_inherit.y - aabb_min_inherit.y > aabb_max_inherit.x - aabb_min_inherit.x && 
			aabb_max_inherit.y - aabb_min_inherit.y > aabb_max_inherit.z- aabb_min_inherit.z)
			longest = 1; 
		else if (aabb_max_inherit.z - aabb_min_inherit.z > aabb_max_inherit.x - aabb_min_inherit.x && 
				aabb_max_inherit.z - aabb_min_inherit.z > aabb_max_inherit.y - aabb_min_inherit.y) 
				longest = 2; 

		
		// Split position 
		float split_pos = ((float*)&aabb_min_inherit)[longest] + 0.5f * ( ((float*)&aabb_max_inherit)[longest] - ((float*)&aabb_min_inherit)[longest] );
		
		// Store split information
		active_list.d_split_axis[idx] = longest;
		active_list.d_split_pos[idx] = split_pos;

		uint32 old_level = active_list.d_node_level[idx];
		
		// Add the two children for spatial median split.
		float aabb_min_child[3] = {aabb_min_inherit.x, aabb_min_inherit.y, aabb_min_inherit.z};
		float aabb_max_child[3] = {aabb_max_inherit.x, aabb_max_inherit.y, aabb_max_inherit.z};

		// Below node 
		uint32 idx_write = idx;
		aabb_max_child[longest] = split_pos;
		device_create_child(next_list, next_list.num_nodes + idx_write, aabb_min_child, aabb_max_child, old_level+1);
		aabb_max_child[longest] = ((float*)&aabb_max_inherit)[longest];
		
		active_list.d_child_left[idx] = idx_write;
		// Set first index to same as parent node.
		next_list.d_first_elem_idx[idx_write] = active_list.d_first_elem_idx[idx];
		
		// Above node 
		idx_write = active_list.num_nodes + idx;
		aabb_min_child[longest] = split_pos;
		device_create_child(next_list, next_list.num_nodes + idx_write, aabb_min_child, aabb_max_child, old_level+1);

		active_list.d_child_right[idx] = idx_write;
		// Set first index to offsetted parent node index.
		next_list.d_first_elem_idx[idx_write] = active_list.next_free_pos + active_list.d_first_elem_idx[idx];
 	}
}


__global__ void kernel_mark_small_nodes(c_kd_node_list next_list, 
										uint32 *d_final_list_idx, 
										uint32 *d_is_small, 
										uint32 *d_small_root_parent)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < next_list.num_nodes)
	{
		uint32 num_nodes_parent = next_list.num_nodes >> 1;

		int is_small = (next_list.d_num_elems_array[idx] <= k_small_node_max? 1 : 0);
		d_is_small[idx] = is_small;

		uint32 idx_active = idx; 
		if (idx_active >= num_nodes_parent)
			idx_active -= num_nodes_parent;

		uint32 smp_value = d_final_list_idx[idx_active];

		// Set MSB to 1 for right nodes 
		if (idx < num_nodes_parent)
			smp_value &= 0x7fffffff;
		else 
			smp_value |= 0x80000000;

		d_small_root_parent[idx] = smp_value;
	}
}

__global__ void kernel_mark_elems_by_node_size(c_kd_chunk_list chunks_list, 
											uint32 *d_num_elems_next, 
											uint32 *d_out_is_small_elem,
											uint32 *d_out_is_large_elem)
{
	uint32 chk = CUDA_GRID2DINDEX;

	__shared__ uint32 s_num_elems_chunk; 
	__shared__ uint32 s_idx_node; 
	__shared__ uint32 s_idx_first_elem; 

	if (threadIdx.x == 0)
	{
		s_num_elems_chunk = chunks_list.d_num_elems[chk]; 
		s_idx_node = chunks_list.d_node_idx[chk];
		s_idx_first_elem = chunks_list.d_first_elem_idx[chk];
	}
	
	__syncthreads(); 
	
	if (threadIdx.x < s_num_elems_chunk)
	{
		uint32 is_small = (d_num_elems_next[s_idx_node] <= k_small_node_max)? 1 : 0;
		d_out_is_small_elem[s_idx_first_elem + threadIdx.x] = is_small; 
		d_out_is_large_elem[s_idx_first_elem + threadIdx.x] = 1 - is_small;
	}
}

// ---------------------------------------------------------------------
/*
/// \brief	Copies nodes from one node list to another.
///
///			Only node levels and AABBs are copied.
/// 		
/// \note	Node list data structures were split up to avoid parameter space overflow. Furthermore
///			this allows to use different source/target AABB types. 
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_move_nodes(uint32 src_num, float4 *src_aabb_min, float4 *src_aabb_max, uint32 *src_level,
								uint32 dest_num, float4 *dest_aabb_min, float4 *dest_aabb_max, uint32 *dest_level, 
								uint32 *d_move, uint32 *d_offsets)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < src_num)
	{
		// Get node offset.
		uint32 offset = dest_num + d_offsets[idx];

		if (d_move[idx] != 0)
		{
			// Copy only valid information. ENA is handled separately.
			dest_level[offset] = src_level[idx]; 
			
			// Inherited bounds. Tight bounds are computed.
			dest_aabb_min[offset] = src_aabb_min[idx];
			dest_aabb_max[offset] = src_aabb_max[idx];
		}
	}
	
}

// ---------------------------------------------------------------------
/*
/// \brief	Updates final node list child information for already added active list nodes.
/// 		
/// 		As a side-effect of empty space cutting, the active list nodes are already added to
/// 		the final node list. However, their child addresses and split axis/position are still
/// 		invalid. This kernel updates this information by using the data kernels wrote to the
/// 		active list \em after adding the nodes to the final list, i.e. it synchronizes the
/// 		active list with the final node list.  
*/ 
// ---------------------------------------------------------------------

__global__ void kernel_update_final_list_children_info(c_kd_node_list active_list, c_kd_final_node_list final_list, uint32 *d_final_list_index)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < active_list.num_nodes)
	{
		// Read final node list index for active list node.
		uint32 idx_final = d_final_list_index[idx];

		// Split information.
		final_list.d_split_axis[idx_final] = active_list.d_split_axis[idx];
		final_list.d_split_pos[idx_final] = active_list.d_split_pos[idx]; 

		// Child information. Note that we have to offset the indices by the number of
		// final nodes as they are currently relative to offset zero.
		final_list.d_child_left[idx_final] = final_list.num_nodes + active_list.d_child_left[idx]; 
		final_list.d_child_right[idx_final] = final_list.num_nodes + active_list.d_child_right[idx_final]; 
	}
}

template <uint32 num_elem_pts>
__global__ void kernel_create_split_candidates(c_kd_node_list small_list, c_kd_split_list split_list)
{
	uint32 idx_node = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_node < small_list.num_nodes)
	{
		uint32 idx_first_tri = small_list.d_first_elem_idx[idx_node]; 
		uint32 num_elems = small_list.d_num_elems_array[idx_node]; 
		uint32 idx_first_split = split_list.d_first_split_idx[idx_node];

		// NOTE: Do not try to simplify the loops since the order is significant for other kernels.
		uint32 idx_split = idx_first_split; 
		for (uint32 axis = 0; axis < 3; ++axis)
		{
			for (uint32 i = 0; i < num_elems; ++i)
			{
				uint32 idx_tna = idx_first_tri + i; 
				// Store both position and axis (lowest 2 bits).
				split_list.d_split_pos_array[idx_split] = ((float*)&small_list.d_elem_point1[idx_tna])[axis]; 
				uint32 info = axis; 
				split_list.d_split_info_array[idx_split] = info; 
				idx_split++; 
			}
		}

		if (num_elem_pts == 2) // compile time 
		{
			for (uint32 axis = 0; axis < 3; ++axis)
			{
				for (uint32 i = 0; i < num_elems; ++i)
				{
					uint32 idx_tna = idx_first_tri + i; 
					split_list.d_split_pos_array[idx_split] = ((float*)&small_list.d_elem_point2[idx_tna])[axis]; 
					uint32 info = 4; 
					info |= axis; 
					split_list.d_split_info_array[idx_split] = info; 
					idx_split++; 
				}
			}
		}	
	}
}

template <bool is_max, uint32 num_elem_pts>
__global__ void kernel_init_split_masks(c_kd_node_list small_list, float *d_rands, c_kd_split_list split_list)
{
	uint32 idx_node = CUDA_GRID2DINDEX; 

	__shared__ uint32 s_idx_first_elem;
	__shared__ uint32 s_num_elems;
	__shared__ uint32 s_idx_first_split; 
	if (threadIdx.x == 0)
	{
		s_idx_first_elem = small_list.d_first_elem_idx[idx_node]; 
		s_num_elems = small_list.d_num_elems_array[idx_node];
		s_idx_first_split = split_list.d_first_split_idx[idx_node]; 
	}

	__syncthreads(); 

	uint32 idx_split = s_idx_first_split + threadIdx.x; 
	if (is_max)
		idx_split += 3*s_num_elems; 

	// First read triangle bounds into shared memory.
	__shared__ float3 s_point1[KD_SMALL_NODE_MAX];
	// Find way to eliminate s_point2 for numElementPoints == 1. Probably it's just left out
	// by the compiler.
	__shared__ float3 s_point2[KD_SMALL_NODE_MAX]; 
	if (threadIdx.x < s_num_elems)
	{
		s_point1[threadIdx.x] = make_float3(small_list.d_elem_point1[s_idx_first_elem + threadIdx.x]); 
		if (num_elem_pts == 2)
			s_point2[threadIdx.x] = make_float3(small_list.d_elem_point2[s_idx_first_elem + threadIdx.x]); 
	}

	__syncthreads(); 

	if (threadIdx.x < 3 * s_num_elems)
	{
		// Get check element on both sides.
		elem_mask_t mask_l = 0; 
		elem_mask_t mask_r = 0; 

		// Get split information.
		// This *will* lead to uncoalesced access when s_numElems is not aligned.
		float split_pos = split_list.d_split_pos_array[idx_split];
		uint32 split_axis = split_list.d_split_info_array[idx_split];
		split_axis &= 3; 

		// TNA index of the element that defined the split.
		uint32 idx_tna = s_idx_first_elem + (threadIdx.x - split_axis * s_num_elems); 

		// NOTE: We are working on the small root list here. Therefore all relevant bits
		//		 of the element mask are set and we do not have to check which bit to
		//		 set. Instead we can just iterate from 0 to triCount-1.

		if (num_elem_pts == 1)
		{
			// Unrolling won't work when moving the compile time if into the loop.
#pragma unroll
			for (uint32 i = 0; i < s_num_elems; ++i)
			{
				// Get point on our axis.
				float elem_pos = ((float*)&s_point1[i])[split_axis]; 

				uint32 is_left = 0; 
				uint32 is_right = 0; 
				if (d_rands[s_idx_first_elem+i] < 0.5f)
				{
					if (elem_pos < split_pos)
						is_left = 1; 
					if (split_pos <= elem_pos)
						is_right = 1; 
				}
				else 
				{
					if (elem_pos <= split_pos)
						is_left = 1; 
					if (split_pos < elem_pos)
						is_right = 1; 
				}

				mask_l |= (((elem_mask_t)is_left) << i); 
				mask_r |= (((elem_mask_t)is_right) << i); 
			}
		}
		else
		{
#pragma unroll
			for (uint32 i = 0; i < s_num_elems; ++i)
			{
				//  Get triangle bounds on our axis.
				float f_min = 0; 
				float f_max = 0; 
				f_min = ((float*)&s_point1[i])[split_axis]; 
				f_max = ((float*)&s_point2[i])[split_axis]; 

				uint32 is_left = 0; 
				uint32 is_right = 0; 
				if (f_min < split_pos)
					is_left = 1; 
				if (split_pos < f_max)
					is_right = 1;

				// Check whether the triangle is the split triangle and classify it according to
				// the type of the split:
				//  - Maximum splits split off the right part of the volume. Therefore the triangle
				//	  has to lie on the left side only.
				//  - Minimum splits split off the left part of the volume. Here the triangle has
				//	  to lie on the right side only.
				// NOTE: According to the profiler, this won't generate too many warp serializes.
				if (is_max)
					if (s_idx_first_elem + i == idx_tna)
						is_left = 1;
				if (!is_max)
					if (s_idx_first_elem + i == idx_tna)
						is_right = 1; 

				// Additionally check whether the element lies directly in the splitting plane
				// and is *not* the split element.
				// NOTE: This generates many warp serializes.
				if (s_idx_first_elem + i != idx_tna && f_min == f_max && f_min == split_pos)
				{
					if (d_rands[s_idx_first_elem + i] < 0.5f)
						is_left = 1;
					else 
						is_right = 1; 
				}

				mask_l |= (((elem_mask_t)is_left) << i);
				mask_r |= (((elem_mask_t)is_right) << i); 
			}
		}

		split_list.d_mask_left_array[idx_split] = mask_l; 
		split_list.d_mask_right_array[idx_split] = mask_r; 
	}
}

__global__ void kernel_update_small_root_parents(c_kd_final_node_list final_list, uint32 *d_small_root_parents, uint32 num_small_nodes)
{
	uint32 idx_small = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_small < num_small_nodes)
	{
		uint32 smp = d_small_root_parents[idx_small]; 
		uint32 idx_node = smp & 0x7fffffff; 
		uint32 is_right = smp >> 31;

		if (is_right)
			final_list.d_child_right[idx_node] = final_list.num_nodes + idx_small;
		else 
			final_list.d_child_left[idx_node] = final_list.num_nodes + idx_small; 
	}
}


// ---------------------------------------------------------------------
/*
/// \brief	Looks for best splits by checking all nodes in parallel.
/// 		
/// 		Depending on the value of \a numElementPoints either VVH cost model (for value 1)
///			or SAH cost model (for value 2) are employed. For each node, the cost for using
///			each of the splitting planes is evaluated and the minimum of the costs is computed.
///			To do this in parallel, ::dev_ReduceFast() is used for reduction.
///
///			The used thread block size for this kernel has to be at least 
///
///			\code KD_SMALLNODEMAX * 6 / 3 = KD_SMALLNODEMAX * 2 \endcode
///
///			to ensure all splits will be handled. Each thread block works on a node. Each thread
///			reduces the number of cost values internally by the factor of 3 to improve reduction
///			performance (hence the division by 3). The factor 6 used above results from the fact
///			that \a numElementPoints = 2 leads to at most 6 possible splitting planes per
///			element.
///
/// \note	Required shared memory per thread block: 4 * 128 + 60 bytes.
/// 
/// \tparam	numElementPoints	The number of points per element determines the type of the
/// 							elements. For one point, it's a simple point. For two points, we
/// 							assume a bounding box. 
///
/// \param	lstActive				The active node list. 
/// \param	lstSplit				The split list. 
/// \param [out]	d_outBestSplit	The best split positions for each node. 
/// \param [out]	d_outSplitCost	The best split costs for each node.  
*/ 
// --------------------------------------------------------------------- 

template <uint32 num_elem_pts> 
__global__ void kernel_find_best_splits(c_kd_node_list active_list, 
										c_kd_split_list split_list, 
										uint32 *d_out_best_split, 
										float *d_out_split_cost)
{
	uint32 idx_node = CUDA_GRID2DINDEX;

	__shared__ uint32 s_idx_first_split; 
	__shared__ uint32 s_num_splits; 
	__shared__ float s_cost_denom; 
	__shared__ elem_mask_t s_mask_node;
	__shared__ float s_aabb_node_min[3]; 
	__shared__ float s_aabb_node_max[3]; 
	__shared__ float s_extend_node[3]; 
	
	if (threadIdx.x == 0)
	{
		uint32 idx_small_root = active_list.d_small_root_idx[idx_node];
		s_idx_first_split = split_list.d_first_split_idx[idx_small_root]; 
		s_num_splits = split_list.d_num_splits[idx_small_root]; 
		
		// Get node's tight bounds for SAH calculation. This is in small stage, therefore no
		// inherited bounds available!
		float4 min4 = active_list.d_aabb_tight_min[idx_node];
		float4 max4 = active_list.d_aabb_tight_max[idx_node]; 
		for (uint32 i = 0; i < 3; ++i)
		{
			s_aabb_node_min[i] = ((float*)&min4)[i]; 
			s_aabb_node_max[i] = ((float*)&max4)[i]; 
			s_extend_node[i] = s_aabb_node_max[i] - s_aabb_node_min[i]; 
		}

		// Get node's element mask.
		s_mask_node = active_list.d_elem_mask[idx_node]; 
		
		// Get denominator for cost evaluation.
		if (num_elem_pts == 1)
		{
			// Get the inverse of the extended node volume.
			float ex_node_vol = 1.0f;
			float max_qr2 = 2 * k_max_query_radius;
			for (uint32 i = 0; i < 3; ++i)
				ex_node_vol *= s_extend_node[i] + max_qr2; 
			s_cost_denom = 1.0f / ex_node_vol; 
		}
		else 
		{
			float area_node = 2.0f * s_extend_node[0]*s_extend_node[1] + 
									s_extend_node[0]*s_extend_node[2] + 
									s_extend_node[1]*s_extend_node[2]; 
			s_cost_denom = 1.0f / area_node; 
		}
	}

	__syncthreads();

	// Shared memory for best split SAH.
	__shared__ float s_split_costs[128];
	
	// Do first step of the reduction by reducing three values per thread.
	uint32 idx_split[3]; 
	float my_cost[3]; 
	elem_mask_t mask_node = s_mask_node; 
	if (num_elem_pts == 1)	// Compile Time
	{
		// Moving if inside loop makes auto unrolling impossible...
#pragma unroll 
		for (uint32 i = 0; i < 3; ++i)
		{
			uint32 my_tid = threadIdx.x + i * blockDim.x;
			idx_split[i] = s_idx_first_split + my_tid; 
			my_cost[i] = M_INFINITY; 
			if (my_tid < s_num_splits)
			{
				float split_pos = split_list.d_split_pos_array[idx_split[i]]; 
				uint32 split_axis = split_list.d_split_info_array[idx_split[i]] & 3; 

				// Get AND'ed element masks to only recognize the contained elements.
				elem_mask_t mask_l = split_list.d_mask_left_array[idx_split[i]] & mask_node; 
				elem_mask_t mask_r = split_list.d_mask_right_array[idx_split[i]] & mask_node; 

				// Count triangles using parallel bit counting.
				uint32 count_l = device_count_bits(mask_l); 
				uint32 count_r = device_count_bits(mask_r); 

				// Get child areas to perform SAH cost calculation.
				uint32 other_axis1 = split_axis+1; 
				uint32 other_axis2 = split_axis+2; 
				if (other_axis1 == 3)
					other_axis1 = 0; 
				if (other_axis2 > 2)
					other_axis2 -= 3; 
				float min_split_axis = s_aabb_node_min[split_axis];
				float max_qr2 = 2 * k_max_query_radius; 
				float volume_l = (s_extend_node[other_axis1] + max_qr2) * 
								(split_pos - min_split_axis + max_qr2) *
								(s_extend_node[other_axis2] + max_qr2); 
				float max_split_axis = s_aabb_node_max[split_axis]; 
				float volume_r = (s_extend_node[other_axis1] + max_qr2) *
								(max_split_axis - split_pos + max_qr2) *
								(s_extend_node[other_axis2] + max_qr2); 

				// Compute VVH cost for this split.
				// WARNING: Picking traversal cost too low can result in endless splitting.
				my_cost[i] = (count_l*volume_l + count_r*volume_r)*s_cost_denom + KD_COST_TRAVERSE;
				
				if (count_l == 0 || count_r == 0)
					my_cost[i] = M_INFINITY; 
			}
		}
	}
	else 
	{
#pragma unroll 
		for (uint32 i = 0; i < 3; ++i)
		{
			uint32 my_tid = threadIdx.x + i * blockDim.x;
			idx_split[i] = s_idx_first_split + my_tid; 
			my_cost[i] = M_INFINITY; 
			if (my_tid < s_num_splits)
			{
				float split_pos = split_list.d_split_pos_array[idx_split[i]]; 
				uint32 split_axis = split_list.d_split_info_array[idx_split[i]] & 3; 

				// Get AND'ed element masks to only recognize the contained elements.
				elem_mask_t mask_l = split_list.d_mask_left_array[idx_split[i]] & mask_node; 
				elem_mask_t mask_r = split_list.d_mask_right_array[idx_split[i]] & mask_node; 

				// Count triangles using parallel bit counting.
				uint32 count_l = device_count_bits(mask_l); 
				uint32 count_r = device_count_bits(mask_r); 

				// Get child areas to perform SAH cost calculation.
				uint32 other_axis1 = split_axis+1; 
				uint32 other_axis2 = split_axis+2; 
				if (other_axis1 == 3)
					other_axis1 = 0; 
				if (other_axis2 > 2)
					other_axis2 -= 3; 
				float min_split_axis = s_aabb_node_min[split_axis];
				float area_l = 2.0f * (s_extend_node[other_axis1] * s_extend_node[other_axis2] +
									(split_pos - min_split_axis) +
									(s_extend_node[other_axis1] + s_extend_node[other_axis2])); 
				float max_split_axis = s_aabb_node_max[split_axis]; 
				float area_r = 2.0f * (s_extend_node[other_axis1] * s_extend_node[other_axis2] +
									(max_split_axis - split_pos) +
									(s_extend_node[other_axis1] + s_extend_node[other_axis2])); 

				// Compute VVH cost for this split.
				// WARNING: Picking traversal cost too low can result in endless splitting.
				my_cost[i] = (count_l*area_l + count_r*area_r)*s_cost_denom + KD_COST_TRAVERSE;
				
				if (count_l == 0 || count_r == 0)
					my_cost[i] = M_INFINITY; 
			}
		}
	}

	s_split_costs[threadIdx.x] = fminf(fminf(my_cost[0], my_cost[1]), my_cost[2]); 
	__syncthreads(); 

	// Now perform reduction on costs to find minimum.
	float min_cost = device_reduce_fast<float, 128, op_minimum<float>>(s_split_costs);

	// Get minimum index. Initialize index to 0xffffffff to identify cases where we could not
	// determine the correct split index for the minimum.
	__shared__ volatile uint32 s_idx_min; 
	if (threadIdx.x == 0)
		s_idx_min = 0xffffffff; 
	__syncthreads(); 

	if (threadIdx.x < s_num_splits && my_cost[0] == min_cost)
		s_idx_min = idx_split[0]; 
	if (threadIdx.x + blockDim.x < s_num_splits && my_cost[1] == min_cost)
		s_idx_min = idx_split[1];
	if (threadIdx.x + 2*blockDim.x < s_num_splits && my_cost[2] == min_cost)
		s_idx_min = idx_split[2]; 
	__syncthreads(); 

	if (threadIdx.x == 0)
	{
		d_out_best_split[idx_node] = s_idx_min; 
		d_out_split_cost[idx_node] = min_cost; 
	}
}

__global__ void kernel_split_small_nodes(kd_small_node_list active_list, 
										c_kd_split_list split_list, 
										kd_small_node_list next_list, 
										uint32 *d_in_best_split,
										float *d_in_split_cost, 
										uint32 *d_out_is_split)
{
	uint32 idx_node = blockIdx.x * blockDim.x + threadIdx.x; 
	
	// Over all nodes in (small) active list.
	if (idx_node < active_list.num_nodes)
	{
		elem_mask_t mask_node = active_list.d_elem_mask[idx_node];
		uint32 idx_small_root = active_list.d_idx_small_root[idx_node];
		uint32 old_level = active_list.d_node_level[idx_node];
		
		// Get node's tight bounds.
		float4 temp = active_list.d_aabb_tight_min[idx_node]; 
		float aabb_node_min[3] = {temp.x, temp.y, temp.z};
		temp = active_list.d_aabb_tight_max[idx_node]; 
		float aabb_node_max[3] = {temp.x, temp.y, temp.z}; 

		// Just read out element count for no split cost, no bit counting here
		float cost0 = active_list.d_num_elems[idx_node]; 
		uint32 idx_split_min = d_in_best_split[idx_node];
		float cost_min = d_in_split_cost[idx_node];

		// // Check whether leaf node or not.
		uint32 left, right;
		uint32 is_split;
		uint32 split_axis = 0; 
		float split_pos = 0.0f;
		
		if (cost_min >= cost0 || idx_split_min == 0xffffffff)
		{
			left = 0; 
			right = 0; 
			is_split = 0; 
		}
		else
		{
			is_split = 1; 
			split_pos = split_list.d_split_pos_array[idx_split_min]; 
			split_axis = split_list.d_split_info_array[idx_split_min] & 3; 

			elem_mask_t mask_l = split_list.d_mask_left_array[idx_split_min] & mask_node; 
			elem_mask_t mask_r = split_list.d_mask_right_array[idx_split_min] & mask_node; 
			uint32 count_l = device_count_bits(mask_l);
			uint32 count_r = device_count_bits(mask_r); 
			
			// Left child.
			left = idx_node; 
			float temp_max = aabb_node_max[split_axis]; 
			next_list.d_aabb_tight_min[left] = make_float4(aabb_node_min[0], aabb_node_min[1], aabb_node_min[2], 0.0f); 
			next_list.d_aabb_tight_max[left] = make_float4(aabb_node_max[0], aabb_node_max[1], aabb_node_max[2], 0.0f); 
			next_list.d_idx_small_root[left] = idx_small_root; 
			next_list.d_num_elems[left] = count_l; 
			next_list.d_node_level[left] = old_level + 1; 
			next_list.d_elem_mask[left] = mask_l; 

			// Right child.
			right = active_list.num_nodes + idx_node;
			aabb_node_min[split_axis] = split_pos; 
			aabb_node_max[split_axis] = temp_max; 
			next_list.d_aabb_tight_min[right] = make_float4(aabb_node_min[0], aabb_node_min[1], aabb_node_min[2], 0.0f); 
			next_list.d_aabb_tight_max[right] = make_float4(aabb_node_max[0], aabb_node_max[1], aabb_node_max[2], 0.0f); 
			next_list.d_idx_small_root[right] = idx_small_root; 
			next_list.d_num_elems[right] = count_r; 
			next_list.d_node_level[right] = old_level + 1; 
			next_list.d_elem_mask[right] = mask_r; 
		}
	
		// Also store split information since we need it later, even for small nodes! 
		active_list.d_split_pos[idx_node] = split_pos;
		active_list.d_split_axis[idx_node] = split_axis; 
		d_out_is_split[idx_node] = is_split;	
	}
} 
	
__global__ void kernel_gen_ena_from_masks(kd_ena_node_list active_list, kd_ena_node_list small_roots_list)
{
	uint32 idx_node = blockIdx.x * blockDim.x + threadIdx.x;

	// Over all nodes in (small) active list.
	if (idx_node < active_list.num_nodes)
	{
		elem_mask_t mask_node = active_list.d_elem_marks[idx_node]; 
		uint32 idx_first_elem = active_list.d_idx_first_elem[idx_node]; 
		uint32 idx_small_root = active_list.d_idx_small_root[idx_node]; 
		uint32 idx_first_tri_sr = small_roots_list.d_idx_first_elem[idx_small_root]; 

		// NOTE: lstSmallRoots is correct here since we need to consider all elements
		//		 in the small root list.
		uint32 offset = idx_first_elem; 
		for (uint32 i = 0; i < small_roots_list.d_num_elems[idx_small_root]; ++i)
		{
			if (mask_node & 0x1)
			{
				uint32 idx_small_elem = small_roots_list.d_elem_node_assoc[idx_first_tri_sr+i];
				active_list.d_elem_node_assoc[offset] = idx_small_elem; 
				offset++; 
			}
			mask_node = mask_node >> 1; 
		}
	}
}

// ---------------------------------------------------------------------
/*
/// \brief	Generates node sizes for a given node level.
/// 		
/// 		Considers all nodes on a given node level and writes the node side to the size array.
///			The node size is composed the following way:
///
///			\li Inner node: left size + elem idx (1) + parent info (2) + right size
///			\li Leaf: node index (1) + element count (1) + element indices (n)
///
///			Here left size is the size of the left subtree, i.e. \c d_sizes[\c left], when \c left
///			is the index of the left child (right size respectively). 
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_traversal_up_path(c_kd_final_node_list final_list, uint32 cur_level, uint32 *d_sizes)
{
	uint32 idx_node = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_node < final_list.num_nodes && final_list.d_node_level[idx_node] == cur_level)
	{
		uint32 left = final_list.d_child_left[idx_node]; 
		uint32 right = final_list.d_child_right[idx_node]; 
		uint32 num_elem = final_list.d_num_elems[idx_node];
		
		uint32 size; 
		if (left == right)
		{
			// Leaf node (node index (1) + element count (1) + element indices (n)).
			size = 1 + 1 + num_elem; 
		}
		else
		{
			// Internal node: left size + elem idx (1) + parent info (2) + right size.
			size = d_sizes[left] + 1 + 2 + d_sizes[right]; 
		}

		d_sizes[idx_node] = size;
	}
}

__global__ void kernel_traversal_down_path(c_kd_final_node_list final_list, 
										uint32 cur_level, 
										uint32 *d_sizes, 
										uint32 *d_addresses, 
										c_kdtree_data kd_data)
{
	uint32 idx_node = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx_node < final_list.num_nodes && final_list.d_node_level[idx_node] == cur_level)
	{
		uint32 left = final_list.d_child_left[idx_node]; 
		uint32 right = final_list.d_child_right[idx_node]; 
		uint32 my_addr = d_addresses[idx_node]; 
		uint32 idx_first_tri = final_list.d_first_elem_idx[idx_node];
		uint32 num_elems = final_list.d_num_elems[idx_node]; 

		// Add in leaf information.
		uint32 idx_node_leaf = idx_node; 
		idx_node_leaf |= ((left == right) ? 0x80000000 : 0); 
		
		float split_pos = final_list.d_split_pos[idx_node]; 
		uint32 split_axis = final_list.d_split_axis[idx_node]; 

		uint32 addr_l, addr_r; 
		if (left != right)
		{
			addr_l = my_addr + 2 + 1;
			addr_r = my_addr + 2 + 1 + d_sizes[left]; 
			d_addresses[left] = addr_l; 
			d_addresses[right] = addr_r; 

			// Write parent info.
			uint2 parent_info; 
			parent_info.x = addr_r; 
			parent_info.x &= 0x0ffffff;	// only 28 bits 
			
			// Write split axis (2 bits) to most significant two bits. Leave custom bits 28, 29 alone.
			parent_info.y |= (split_axis << 30); 
			parent_info.y = *(uint32*)&split_pos; 
			kd_data.d_preorder_tree[my_addr+1] = parent_info.x;
			kd_data.d_preorder_tree[my_addr+2] = parent_info.y; 
		}
		else 
		{
			addr_l = 0; 
			addr_r = 0; 

			kd_data.d_preorder_tree[my_addr+1] = num_elems; 

			// Write element indices to preorder tree.
			for (uint32 i = 0; i < num_elems; ++i)
				kd_data.d_preorder_tree[my_addr+i] = final_list.d_elem_node_assoc[idx_first_tri+i]; 
		}

		// Compute and write node extent (center, radius).
		float3 aabb_min = make_float3(final_list.d_aabb_min[idx_node]); 
		float3 aabb_max = make_float3(final_list.d_aabb_max[idx_node]); 

		float3 diagonal = aabb_max - aabb_min;
		float radius = 0.5f * length(diagonal); 
		float3 node_center = aabb_min + 0.5f * diagonal; 
		kd_data.d_node_extent[idx_node] = make_float4(node_center.x, node_center.y, node_center.z, radius); 
	}
}

//////////////////////////////////////////////////////////////////////////

extern "C"
void init_kd_kernels()
{
	// Find out how many thread blocks (grid dimension) we can use on the current device.
	int cur_device;
	cuda_safe_call_no_sync(cudaGetDevice(&cur_device));
	cuda_safe_call_no_sync(cudaGetDeviceProperties(&device_props, cur_device));
}

extern "C" 
void kernel_set_kd_params(uint32 small_node_max)
{
	assert(small_node_max <= KD_SMALL_NODE_MAX);
	
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("k_small_node_max", &small_node_max, sizeof(uint32)));
}

extern "C++"
template <uint32 num_elem_pts> 
void kernel_wrapper_gen_chunk_aabb(const c_kd_node_list& active_list, c_kd_chunk_list& chunks_list)
{
	// Note that we use half the chunk size here. This is a reduction optimization.
	dim3 block_size = dim3(KD_CHUNKSIZE/2, 1, 1);
	// Avoid the maximum grid size by using two dimensions.
	dim3 grid_size = CUDA_MAKEGRID2D(chunks_list.num_chunks, device_props.maxGridSize[0]);
	
	kernel_gen_chunk_aabb<num_elem_pts><<<grid_size, block_size>>>(active_list, chunks_list);
	CUDA_CHECKERROR;
}

extern "C"
void kernel_wrappper_empty_space_cutting(c_kd_node_list& active_list, 
										c_kd_final_node_list& final_list, 
										float empty_space_ratio, 
										uint32 *d_io_final_list_idx)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(active_list.num_nodes, block_size.x), 1, 1);
	
	c_cuda_memory<uint32> d_can_cutoff(active_list.num_nodes);
	c_cuda_memory<uint32> d_cut_offsets(active_list.num_nodes);

	// Set empty space ratio
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("k_empty_space_ratio", &empty_space_ratio, sizeof(float)));
	
	for (uint32 is_max = 0; is_max < 2; ++is_max)
	{
		for (uint32 axis = 0; axis < 3; ++axis)
		{
			bool bmax = (is_max == 1);
			kernel_can_cutoff_empty_space<<<grid_size, block_size>>>(active_list, axis, bmax, d_can_cutoff.buf_ptr());
			CUDA_CHECKERROR; 

			cuda_scan(d_can_cutoff.buf_ptr(), active_list.num_nodes, false, d_cut_offsets.buf_ptr());
			
			// Get number of cuts by reduction.
			uint32 num_cuts;
			cuda_reduce_add(num_cuts, (uint32*)d_can_cutoff.buf_ptr(), active_list.num_nodes, (uint32)0);
			
			others_ofs << "Axis " << axis << " " << "is max? " << is_max << std::endl; 
			print_device_array(others_ofs, d_can_cutoff.buf_ptr(), active_list.num_nodes); 
			
			if (num_cuts > 0)
			{
				// Verify we have enough space.
				if (final_list.max_nodes < final_list.num_nodes + 2*num_cuts)
					final_list.resize_node_data(final_list.num_nodes + 2*num_cuts);

				// Perform cut and generate new final list nodes and update active list nodes.
				kernel_empty_space_cutting<<<grid_size, block_size>>>(active_list, final_list, 
																	axis, 
																	bmax, 
																	d_can_cutoff.buf_ptr(), 
																	d_cut_offsets.buf_ptr(), 
																	num_cuts, 
																	d_io_final_list_idx);
				CUDA_CHECKERROR;
				
				// Update final list node count. It increases by 2*numCuts since we both had to create
				// the empty cut-off node and the tighter node.
				final_list.num_nodes += 2*num_cuts;
			}
		}
	} 
}

extern "C"
void kernel_wrapper_split_large_nodes(const c_kd_node_list& active_list, c_kd_node_list& next_list)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(active_list.num_nodes, block_size.x), 1, 1);
	
	// Convert next list to internal representation.
	kd_node_list_aabb next_list_in(next_list);
	
	kernel_split_large_nodes<<<grid_size, block_size>>>(active_list, next_list_in);
	CUDA_CHECKERROR;
}

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_mark_left_right_elems(const c_kd_node_list& active_list, const c_kd_chunk_list& chunks_list, uint32 *d_valid)
{
	dim3 block_size = dim3(KD_CHUNKSIZE, 1, 1);
	dim3 grid_size = CUDA_MAKEGRID2D(chunks_list.num_chunks, device_props.maxGridSize[0]);

	// Build random number array.
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	uint32 num_rands = rng.get_aligned_cnt(chunks_list.num_chunks*KD_CHUNKSIZE);
	c_cuda_memory<float> d_randoms(num_rands);
	rng.seed(rand());
	cuda_safe_call_no_sync(rng.gen_rand(d_randoms.buf_ptr(), num_rands));
	
	kernel_mark_left_right_elems<num_elem_pts><<<grid_size, block_size>>>(active_list, chunks_list, d_randoms.buf_ptr(), d_valid); 
}


extern "C"
void kernel_wrapper_mark_small_nodes(const c_kd_node_list& next_list, uint32 *d_final_list_idx, uint32 *d_is_small, uint32 *d_small_root_parent)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(next_list.num_nodes, block_size.x), 1, 1);
	
	kernel_mark_small_nodes<<<grid_size, block_size>>>(next_list, d_final_list_idx, d_is_small, d_small_root_parent);
	CUDA_CHECKERROR; 
}

extern "C"
void kernel_wrapper_mark_elems_by_node_size(const c_kd_chunk_list& chunks_list, 
											uint32 *d_num_elems_next, 
											uint32 *d_out_is_small_elem, 
											uint32 *d_out_is_large_elem)
{
	dim3 block_size = dim3(KD_CHUNKSIZE, 1, 1);
	dim3 grid_size = CUDA_MAKEGRID2D(chunks_list.num_chunks, device_props.maxGridSize[0]);
	
	kernel_mark_elems_by_node_size<<<grid_size, block_size>>>(chunks_list, d_num_elems_next, d_out_is_small_elem, d_out_is_large_elem);
	CUDA_CHECKERROR; 
}

extern "C"
void kernel_wrapper_move_nodes(const c_kd_node_list& src_list, c_kd_node_list& dest_list, uint32 *d_move, uint32 *d_offsets, bool b_dest_is_small)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(src_list.num_nodes, block_size.x), 1, 1);
	
	if (b_dest_is_small)
	{
		// For small nodes we have to move to the tight bounds since for them we make no
		// difference between inherited and tight bounds!
		kernel_move_nodes<<<grid_size, block_size>>>(src_list.num_nodes, 
													src_list.d_aabb_inherit_min, 
													src_list.d_aabb_inherit_max, 
													src_list.d_node_level, 
													dest_list.num_nodes, 
													dest_list.d_aabb_tight_min, 
													dest_list.d_aabb_tight_max,
													dest_list.d_node_level,
													d_move,
													d_offsets); 
		
	}
	else 
	{
		// For the remaining large nodes tight bounds are calculated, therefore move to
		// inherited bounds.
		kernel_move_nodes<<<grid_size, block_size>>>(src_list.num_nodes, 
													src_list.d_aabb_inherit_min, 
													src_list.d_aabb_inherit_max, 
													src_list.d_node_level, 
													dest_list.num_nodes, 
													dest_list.d_aabb_inherit_min, 
													dest_list.d_aabb_inherit_max,
													dest_list.d_node_level,
													d_move,
													d_offsets); 
	}
	CUDA_CHECKERROR;
}


extern "C"
void kernel_wrapper_update_final_list_child_info(const c_kd_node_list& active_list, c_kd_final_node_list& final_list, uint32 *d_final_list_index)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(active_list.num_nodes, block_size.x), 1, 1);
	
	kernel_update_final_list_children_info<<<grid_size, block_size>>>(active_list, final_list, d_final_list_index);
	CUDA_CHECKERROR;
}

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_create_split_candidates(const c_kd_node_list& small_list, c_kd_split_list& split_list)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(CUDA_DIVUP(small_list.num_nodes, block_size.x), 1, 1); 

	kernel_create_split_candidates<num_elem_pts><<<grid_size, block_size>>>(small_list, split_list); 
	CUDA_CHECKERROR; 
}

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_init_split_masks(const c_kd_node_list& small_list, uint32 small_node_max, c_kd_split_list& split_list)
{
	dim3 block_size = dim3(3*small_node_max, 1, 1);
	dim3 grid_size = CUDA_MAKEGRID2D(small_list.num_nodes, device_props.maxGridSize[0]);
	
	// Build random number array.
	c_cuda_rng& rng = c_cuda_rng::get_instance(); 
	uint32 num_rands = rng.get_aligned_cnt(small_list.next_free_pos); 
	c_cuda_memory<float> d_rands(num_rands); 
	rng.seed(rand());
	cuda_safe_call_no_sync(rng.gen_rand(d_rands.buf_ptr(), num_rands)); 

	// Minimums / Single point
	kernel_init_split_masks<0, num_elem_pts><<<grid_size, block_size>>>(small_list, d_rands.buf_ptr(), split_list);

	if (num_elem_pts == 2)
	{
		kernel_init_split_masks<1, num_elem_pts><<<grid_size, block_size>>>(small_list, d_rands.buf_ptr(), split_list);
	}
}

extern "C"
void kernel_wrapper_update_small_root_parents(const c_kd_final_node_list& final_list, uint32 *d_small_root_parents, uint32 num_small_nodes)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 gird_size = dim3(CUDA_DIVUP(num_small_nodes, block_size.x), 1, 1);

	kernel_update_small_root_parents<<<gird_size, block_size>>>(final_list, d_small_root_parents, num_small_nodes); 
	CUDA_CHECKERROR; 
}

extern "C++"
template <uint32 num_elem_pts> 
void kernel_wrapper_find_best_split(const c_kd_node_list& active_list, 
									const c_kd_split_list& split_list, 
									float max_query_radius, 
									uint32 *d_out_best_split, 
									float *d_out_split_cost)
{
	// We use KD_SMALLNODEMAX*6/3 = 384/3 = 128 here for accelerated reduction. 
	dim3 block_size = dim3(128, 1, 1); 
	dim3 grid_size = CUDA_MAKEGRID2D(active_list.num_nodes, device_props.maxGridSize[0]);
	
	// Set maximum query radius.
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("k_max_query_radius", &max_query_radius, sizeof(float)));

	kernel_find_best_splits<num_elem_pts><<<grid_size, block_size>>>(active_list, split_list, d_out_best_split, d_out_split_cost);  
}

extern "C"
void kernel_wrapper_split_small_nodes(const c_kd_node_list& active_list, 
									const c_kd_split_list& split_list,
									const c_kd_node_list& next_list, 
									uint32 *d_in_best_split, 
									float *d_in_split_cost, 
									uint32 *d_out_is_split)
{
	dim3 block_size = dim3(256, 1, 1);
	dim3 gird_size = dim3(CUDA_DIVUP(active_list.num_nodes, block_size.x), 1, 1);
	
	kd_small_node_list next_list_internal; 
	next_list_internal.num_nodes = next_list.num_nodes; 
	next_list_internal.d_num_elems = next_list.d_num_elems_array; 
	next_list_internal.d_node_level = next_list.d_node_level; 
	next_list_internal.d_aabb_tight_min = next_list.d_aabb_tight_min; 
	next_list_internal.d_aabb_tight_max = next_list.d_aabb_tight_max; 
	next_list_internal.d_split_axis = next_list.d_split_axis; 
	next_list_internal.d_split_pos = next_list.d_split_pos; 
	next_list_internal.d_idx_small_root = next_list.d_small_root_idx; 
	next_list_internal.d_elem_mask = next_list.d_elem_mask; 

	kd_small_node_list active_list_internal;
	active_list_internal.num_nodes = active_list.num_nodes; 
	active_list_internal.d_num_elems = active_list.d_num_elems_array; 
	active_list_internal.d_node_level = active_list.d_node_level; 
	active_list_internal.d_aabb_tight_min = active_list.d_aabb_tight_min; 
	active_list_internal.d_aabb_tight_max = active_list.d_aabb_tight_max; 
	active_list_internal.d_split_axis = active_list.d_split_axis; 
	active_list_internal.d_split_pos = active_list.d_split_pos; 
	active_list_internal.d_idx_small_root = active_list.d_small_root_idx; 
	active_list_internal.d_elem_mask = active_list.d_elem_mask; 
	
	
	kernel_split_small_nodes<<<gird_size, block_size>>>(active_list_internal, 
														split_list, 
														next_list_internal, 
														d_in_best_split, 
														d_in_split_cost, 
														d_out_is_split); 
}

extern "C"
void kernel_wrapper_gen_ena_from_masks(c_kd_node_list& active_list, const c_kd_node_list& small_roots)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(CUDA_DIVUP(active_list.num_nodes, block_size.x), 1, 1); 
	
	// Convert next list to internal representation.
	kd_ena_node_list small_roots_internal; 
	small_roots_internal.d_num_elems = small_roots.d_num_elems_array; 
	small_roots_internal.d_idx_first_elem = small_roots.d_first_elem_idx; 
	small_roots_internal.d_idx_small_root = small_roots.d_small_root_idx; 
	small_roots_internal.d_elem_marks = small_roots.d_elem_mask; 
	small_roots_internal.d_elem_node_assoc = small_roots.d_num_elems_array; 

	kd_ena_node_list active_list_internal; 
	active_list_internal.d_num_elems = active_list.d_num_elems_array; 
	active_list_internal.d_idx_first_elem = active_list.d_first_elem_idx; 
	active_list_internal.d_idx_small_root = active_list.d_small_root_idx; 
	active_list_internal.d_elem_marks = active_list.d_elem_mask; 
	active_list_internal.d_elem_node_assoc = active_list.d_num_elems_array; 

	kernel_gen_ena_from_masks<<<grid_size, block_size>>>(active_list_internal, small_roots_internal); 
}

extern "C"
void kernel_wrapper_traversal_up_path(const c_kd_final_node_list& final_list, uint32 cur_level, uint32 *d_sizes)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(CUDA_DIVUP(final_list.num_nodes, block_size.x), 1, 1); 
	
	kernel_traversal_up_path<<<grid_size, block_size>>>(final_list, cur_level, d_sizes); 
	
}

extern "C"
void kernel_wrapper_traversal_down_path(const c_kd_final_node_list& final_list, 
										uint32 cur_level, 
										uint32 *d_sizes, 
										uint32 *d_addresses, 
										c_kdtree_data& kd_data)
{
	dim3 block_size = dim3(256, 1, 1); 
	dim3 grid_size = dim3(CUDA_DIVUP(final_list.num_nodes, block_size.x), 1, 1); 
	
	kernel_traversal_down_path<<<grid_size, block_size>>>(final_list, cur_level, d_sizes, d_addresses, kd_data);
}

extern "C++" template void kernel_wrapper_gen_chunk_aabb<1>(const c_kd_node_list& active_list, c_kd_chunk_list& chunks_list);
extern "C++" template void kernel_wrapper_gen_chunk_aabb<2>(const c_kd_node_list& active_list, c_kd_chunk_list& chunks_list);

extern "C++" template void kernel_wrapper_init_split_masks<1>(const c_kd_node_list& small_list, uint32 small_node_max, c_kd_split_list& split_list); 
extern "C++" template void kernel_wrapper_init_split_masks<2>(const c_kd_node_list& small_list, uint32 small_node_max, c_kd_split_list& split_list); 

extern "C++" template void kernel_wrapper_create_split_candidates<1>(const c_kd_node_list& small_list, c_kd_split_list& split_list);
extern "C++" template void kernel_wrapper_create_split_candidates<2>(const c_kd_node_list& small_list, c_kd_split_list& split_list);

extern "C++" template void kernel_wrapper_mark_left_right_elems<1>(const c_kd_node_list& active_list, const c_kd_chunk_list& chunks_list, uint32 *d_valid);
extern "C++" template void kernel_wrapper_mark_left_right_elems<2>(const c_kd_node_list& active_list, const c_kd_chunk_list& chunks_list, uint32 *d_valid);

extern "C++" template void kernel_wrapper_find_best_split<1>(const c_kd_node_list& active_list, 
															const c_kd_split_list& split_list, 
															float max_query_radius, 
															uint32 *d_out_best_split, 
															float *d_out_split_cost);

extern "C++" template void kernel_wrapper_find_best_split<2>(const c_kd_node_list& active_list, 
															const c_kd_split_list& split_list, 
															float max_query_radius, 
															uint32 *d_out_best_split, 
															float *d_out_split_cost);