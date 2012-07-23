#include "kdtree_gpu.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"
#include "cuda_primitives.h"
#include "math_utils.h"

// For kd-tree debug 
#include "kdtree_debug.h"

extern "C"
void kernel_set_kd_params(uint32 samll_node_max);

extern "C"
void init_kd_kernels();

extern "C"
void kernel_wrapper_get_chunk_counts(uint32 *d_num_elems_node, uint32 num_nodes, uint32 *d_out_chunk_counts);

extern "C"
void kernel_wrapper_kd_gen_chunks(uint32 *d_num_elems_array, 
								uint32 *d_idx_first_elem_array, 
								uint32 num_nodes, 
								uint32 *d_offsets, 
								c_kd_chunk_list& chunk_list);

extern "C"
void kernel_wrapper_mark_small_nodes(const c_kd_node_list& next_list, uint32 *d_final_list_idx, uint32 *d_is_small, uint32 *d_small_root_parent);

extern "C"
void kernel_wrapper_mark_elems_by_node_size(const c_kd_chunk_list& chunks_list, 
											uint32 *d_num_elems_next, 
											uint32 *d_out_is_small_elem, 
											uint32 *d_out_is_large_elem);

extern "C"
void kernel_wrapper_move_nodes(const c_kd_node_list& src_list,
							c_kd_node_list& dest_list, 
							uint32 *d_move, uint32 *d_offsets, 
							bool b_dest_is_small); 


extern "C"
void kernel_wrappper_empty_space_cutting(c_kd_node_list& active_list, c_kd_final_node_list& final_list, float empty_space_ratio, uint32 *d_io_final_list_idx);

extern "C"
void kernel_wrapper_split_large_nodes(const c_kd_node_list& active_list, c_kd_node_list& next_list); 

extern "C"
void kernel_wrapper_count_elems_chunk(const c_kd_chunk_list& chunks_list, uint32 *d_valid_flags, uint32 *d_out_num_elems);

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_gen_chunk_aabb(const c_kd_node_list& active_list, c_kd_chunk_list& chunks_list);

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_mark_left_right_elems(const c_kd_node_list& active_list, const c_kd_chunk_list& chunks_list, uint32 *d_valid); 

extern "C"
void kernel_wrapper_update_final_list_child_info(const c_kd_node_list& active_list, 
												c_kd_final_node_list& final_list,
												uint32 *d_final_list_index);

extern "C++"
template <uint32 num_elem_pts> 
void kernel_wrapper_create_split_candidates(const c_kd_node_list& small_list, c_kd_split_list& split_list); 

extern "C++"
template <uint32 num_elem_pts>
void kernel_wrapper_init_split_masks(const c_kd_node_list& small_list, uint32 small_node_max, c_kd_split_list& split_list); 

extern "C"
void kernel_wrapper_update_small_root_parents(const c_kd_final_node_list& final_list, uint32 *d_small_root_parents, uint32 num_small_nodes);

extern "C++"
template <uint32 num_elems_pts>
void kernel_wrapper_find_best_split(const c_kd_node_list& active_list, 
									const c_kd_split_list&  split_list,
									float max_query_radius,
									uint32 *d_out_best_split, 
									float *d_out_split_cost); 

extern "C"
void kernel_wrapper_split_small_nodes(const c_kd_node_list& active_list, 
									const c_kd_split_list& split_list,
									const c_kd_node_list& next_list, 
									uint32 *d_in_best_split, 
									float *d_in_split_cost, 
									uint *d_out_is_split); 

extern "C"
void kernel_wrapper_gen_ena_from_masks(c_kd_node_list& active_list, const c_kd_node_list& small_roots); 

extern "C"
void kernel_wrapper_traversal_up_path(const c_kd_final_node_list& final_list, uint32 cur_level, uint32 *d_sizes);

extern "C"
void kernel_wrapper_traversal_down_path(const c_kd_final_node_list& final_list, 
										uint32 cur_level,
										uint32 *d_sizes,
										uint32 *d_addresses,
										c_kdtree_data& kd_data); 

std::ofstream chunk_ofs("chunks_list_debug.txt"); 
std::ofstream active_ofs("active_list_debug.txt"); 
std::ofstream others_ofs("intermidiate_debug.txt"); 

//////////////////////////////////////////////////////////////////////////

c_kdtree_gpu::c_kdtree_gpu(size_t num_input_elems, uint32 num_elems_points, float3 root_aabb_min, float3 root_aabb_max, float empty_ratio, uint32 num_small_node_max)
	: d_temp_val(1)
	, m_current_chunk_list_src(NULL)
	, m_num_input_elements(num_input_elems)
	, m_num_element_points(num_elems_points)
	, m_root_aabb_min(root_aabb_min)
	, m_root_aabb_max(root_aabb_max)
	, m_final_list(NULL)
	, m_active_node_list(NULL) 
	, m_small_node_list(NULL)
	, m_next_node_list(NULL)
	, m_chunk_list(NULL)
	, m_split_list(NULL)
	, m_kd_data(NULL)
	, d_small_root_parents(NULL)
	, m_empty_scene_ratio(empty_ratio)
	, m_small_nodes_max(num_small_node_max) 
	, m_max_query_radius(1.0f)
{ 
	init_kd_kernels();
}

void c_kdtree_gpu::pre_build()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 

	m_final_list = new c_kd_final_node_list();
	m_final_list->initialize(2*m_num_input_elements, 16*m_num_input_elements);
	
	m_active_node_list = new c_kd_node_list();
	m_active_node_list->initialize(m_num_input_elements, 4*m_num_input_elements, m_num_element_points);

	m_small_node_list = new c_kd_node_list();
	m_small_node_list->initialize(m_num_input_elements, 4*m_num_input_elements, m_num_element_points);

	m_next_node_list = new c_kd_node_list();
	m_next_node_list->initialize(m_num_input_elements, 4*m_num_input_elements, m_num_element_points);

	m_chunk_list = new c_kd_chunk_list();
	m_chunk_list->initialize(m_num_input_elements);

	// Do not initialize here since size is unknown. 
	m_split_list = NULL;
	m_kd_data = NULL;

	
	// Initialize small root parent vector. We need at most x entries, where x is the
	// maximum number of nodes in the small (root) list.
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_small_root_parents, m_num_input_elements*sizeof(uint32)));

	kernel_set_kd_params(m_small_nodes_max);
}

void c_kdtree_gpu::post_build()
{
	if (!d_small_root_parents)
		return;

	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();

	cuda_safe_call_no_sync(mem_pool.release(d_small_root_parents));
	d_small_root_parents = NULL;
	
	m_final_list->free_memory();
	SAFE_DELETE(m_final_list);

	m_active_node_list->free_memory();
	SAFE_DELETE(m_active_node_list);

	m_small_node_list->free_memory();
	SAFE_DELETE(m_small_node_list);

	m_next_node_list->free_memory();
	SAFE_DELETE(m_next_node_list);

	m_chunk_list->free_memory();
	SAFE_DELETE(m_chunk_list);
	
	if (m_split_list)
		m_split_list->free_memory();
	SAFE_DELETE(m_split_list);
	
}

bool c_kdtree_gpu::build_tree()
{
	pre_build(); 

	// Reset lists
	
	m_final_list->clear();
	m_active_node_list->clear();
	m_small_node_list->clear();
	m_next_node_list->clear();
	m_chunk_list->clear();

	// Create and add root node
	add_root_node(m_active_node_list);
	
	// Large node stage
	large_node_stage();

	/*
	// Small node stage
	small_node_stage();

	// Generates final node list m_pKDData.
	preorder_traversal(); 
	*/ 

	chunk_ofs.close(); 
	active_ofs.close();
	
	post_build();

	return true; 
}

void c_kdtree_gpu::large_node_stage()
{
	// Iterate until the active list is empty, that is until there are
	// no more large nodes to work on.
	while (!m_active_node_list->is_empty())
	{
		// Append the active list to the final node list. This is done even if the child information
		// aren't available, yet. It is fixed later.
		m_final_list->append_list(m_active_node_list, false, true);

		// Keeps track of where the current active list's nodes are in the final node list.
		c_cuda_memory<uint32> d_final_list_idx(m_active_node_list->num_nodes);
		cuda_init_identity(d_final_list_idx.buf_ptr(), m_active_node_list->num_nodes);
		cuda_constant_add<uint32>(d_final_list_idx.buf_ptr(), 
								m_active_node_list->num_nodes, 
								m_final_list->num_nodes - m_active_node_list->num_nodes);
	
		// Clear the next list which stores the nodes for the next step.
		m_next_node_list->clear();

		// Process the active nodes. This generated both small nodes and new
		// next nodes.
		process_large_nodes(d_final_list_idx.buf_ptr());

		// Swap active and next list for next pass.
		c_kd_node_list *temp = m_active_node_list; 
		m_active_node_list = m_next_node_list; 
		m_next_node_list = temp;
	} 
}

void c_kdtree_gpu::small_node_stage()
{
	pre_process_small_nodes();

	m_active_node_list->clear(); 
	m_active_node_list->append_list(m_small_node_list, true); 
	while (m_active_node_list->is_empty())
	{
		// NOTE: The paper tells to append the active list to the final node list here.
		//		 I do not follow since this way the changes (children, ...) won't
		//		 get into the final node list. Instead I moved this to the end of the
		//		 small processing stage.

		// Clear the next list which stores the nodes for the next step.
		m_next_node_list->clear(); 
		process_small_nodes();

		// No swapping required here. This is done in ProcessSmallNodes to avoid
		// using temporary memory
	}
	
}

void c_kdtree_gpu::process_large_nodes(uint32 *d_final_list_idx_active)
{
	assert(!m_active_node_list->is_empty());
	
	// Group elements into chunks.
	create_chunk_list(m_active_node_list);

	// Compute per node bounding boxes.
	compute_nodes_aabbs(); 

	// Split large nodes.
	split_large_nodes(d_final_list_idx_active);

	// Sort and clip elements to child nodes.
	// sort_clip_to_child_nodes();

	/* 
	// Now we have unclipped element bounds, so perform split clipping. Per default, this
	// does nothing. Clipping has to be realized in subclasses.
	perform_split_clipping(m_active_node_list, m_next_node_list); 

	// Update lists for next run 
	update_small_list(d_final_list_idx_active);

	*/ 
}

void c_kdtree_gpu::pre_process_small_nodes()
{
	c_cuda_memory<uint32> d_aligned_split_counts(m_small_node_list->num_nodes);

	m_split_list = new c_kd_split_list();
	m_split_list->initialize(m_small_node_list);

	// Compute candidate number. This is exactly numElementPoints the number of
	// elements times three axes for each node.
	cuda_safe_call_no_sync(cudaMemcpy(m_split_list->d_num_splits, 
									m_small_node_list->d_num_elems_array, 
									m_small_node_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice));
	cuda_constant_mul<uint32>(m_split_list->d_num_splits, m_small_node_list->num_nodes, m_small_node_list->num_elem_points*3);
	
	// Align split counts before scanning to get aligned split offsets.
	cuda_align_counts(d_aligned_split_counts.buf_ptr(), m_split_list->d_num_splits, m_small_node_list->num_nodes); 

	// Compute offsets from counts using scan.
	cuda_scan(d_aligned_split_counts.buf_ptr(), m_small_node_list->num_nodes, false, m_split_list->d_first_split_idx);

	// Get number of entries required for split list.
	// NOTE: Reduction is currently required because alignment prevents from simply
	//       calculating the total number.
	uint32 num_split_total; 
	cuda_reduce_add(num_split_total, (uint32*)d_aligned_split_counts.buf_ptr(), m_small_node_list->num_nodes, (uint32)0); 
	
	// Allocate split memory. Use 128 byte alignment for 64 bit element masks.
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&m_split_list->d_split_pos_array, num_split_total*sizeof(float), "kd-tree misc"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&m_split_list->d_split_info_array, num_split_total*sizeof(uint32), "kd-tree misc"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&m_split_list->d_mask_left_array, num_split_total*sizeof(elem_mask_t), "kd-tree misc", 128));
	cuda_safe_call_no_sync(mem_pool.request((void**)&m_split_list->d_mask_right_array, num_split_total*sizeof(elem_mask_t), "kd-tree misc", 128));
	
	// Create split candidates (inits split list). Parallelized over nodes.
	if (m_small_node_list->num_elem_points == 1)
		kernel_wrapper_create_split_candidates<1>(*m_small_node_list, *m_split_list); 
	else 
		kernel_wrapper_create_split_candidates<2>(*m_small_node_list, *m_split_list); 
	
	if (m_small_node_list->num_elem_points == 1)
		kernel_wrapper_init_split_masks<1>(*m_small_node_list, m_small_nodes_max, *m_split_list);
	else
		kernel_wrapper_init_split_masks<2>(*m_small_node_list, m_small_nodes_max, *m_split_list);

	// Initialize small list extra data (d_elemMask, d_idxSmallRoot). We just set the masks
	// completely (even if there are less elements in the node) to allow the use of
	// cudaMemset.
	cuda_init_identity(m_small_node_list->d_small_root_idx, m_small_node_list->num_nodes); 
	cuda_safe_call_no_sync(cudaMemset(m_small_node_list->d_elem_mask, 0xff, m_small_node_list->num_nodes*sizeof(elem_mask_t)));

	// We now need to update small root parents in m_pListFinal. This is neccessary to
	// retain the connection between large tree and small root trees. It is assumed
	// here that the small root nodes are added right after the current m_pListFinal
	// last node.
	kernel_wrapper_update_small_root_parents(*m_final_list, d_small_root_parents, m_small_node_list->num_nodes); 

}

void c_kdtree_gpu::process_small_nodes()
{
	assert(m_active_node_list->num_nodes > 0); 
	c_cuda_memory<uint32> d_is_split(m_active_node_list->num_nodes); 
	c_cuda_memory<uint32> d_best_splits(m_active_node_list->num_nodes); 
	c_cuda_memory<float> d_min_sahs(m_active_node_list->num_nodes);

	if (m_active_node_list->num_elem_points == 1)
		kernel_wrapper_find_best_split<1>(*m_active_node_list, 
										*m_split_list, 
										m_max_query_radius, 
										d_best_splits.buf_ptr(), 
										d_min_sahs.buf_ptr()); 
	else 
		kernel_wrapper_find_best_split<1>(*m_active_node_list, 
										*m_split_list, 
										m_max_query_radius, 
										d_best_splits.buf_ptr(), 
										d_min_sahs.buf_ptr());

	// Resize node data if required.
	if (m_next_node_list->num_nodes < 2 * m_active_node_list->num_nodes)
		m_next_node_list->resize_node_data(2*m_active_node_list->num_nodes); 
	
	// Generate children into next list. The result contains the left result in the first
	// m_pListActive->numNodes elements and the right results in the following elements.
	// It contains holes where no children were generated. Therefore we need to compact
	// the result and require a isSplit array in d_isSplit.
	kernel_wrapper_split_small_nodes(*m_active_node_list, 
									*m_split_list, 
									*m_next_node_list, 
									d_best_splits.buf_ptr(), 
									d_min_sahs.buf_ptr(), 
									d_is_split.buf_ptr());
	
	uint32 *d_child_offsets = d_best_splits.buf_ptr();
	// The child data is invalid cause we compact the next list later. Therefore both left
	// and right child indices have to be updated. This can be done by scanning the inverse
	// of the isSplit array. Example:
	//
	// 0 1 2 3 4 5 (Identity)
	// 0 0 1 0 0 1 (isSplit)
	// 0 0 2 0 0 5 (isSplit * Identity)
	// 1 1 0 1 1 0 (not isSplit)
	// 0 1 2 2 3 4 (scan not isSplit)
	// 0 0 2 0 0 4 (isSplit * (scan not isSplit))
	// 0 0 0 0 0 1 (Left := Identity - scan not isSplit)
	cuda_inverse_binary(d_is_split.buf_ptr(), m_active_node_list->num_nodes); 

	cuda_scan(d_is_split.buf_ptr(), m_active_node_list->num_nodes, false, d_child_offsets); 
	
	cuda_inverse_binary(d_is_split.buf_ptr(), m_active_node_list->num_nodes); 
	cuda_array_op<cuda_op_mul, uint32>(d_child_offsets, d_is_split.buf_ptr(), m_active_node_list->num_nodes); 

	cuda_init_identity(m_active_node_list->d_child_left, m_active_node_list->num_nodes); 
	cuda_array_op<cuda_op_mul, uint32>(d_child_offsets, d_is_split.buf_ptr(), m_active_node_list->num_nodes); 

	cuda_init_identity(m_active_node_list->d_child_left, m_active_node_list->num_nodes); 
	cuda_array_op<cuda_op_mul, uint32>(m_active_node_list->d_child_left, d_is_split.buf_ptr(), m_active_node_list->num_nodes); 
	cuda_array_op<cuda_op_sub, uint32>(m_active_node_list->d_child_left, d_child_offsets, m_active_node_list->num_nodes); 

	uint32 num_splits;
	cuda_reduce_add(num_splits, (uint32*)d_is_split.buf_ptr(), m_active_node_list->num_nodes, (uint32)0); 
	
	// Left child indices are OK now. Right indices we get the following way:
	//
	// 0 0 0 0 0 1 (Left)
	// 0 0 1 0 0 1 (isSplit)
	// 2 2 2 2 2 2 (numSplits)
	// 0 0 2 0 0 2 (isSplit * numSplits)
	// 0 0 2 0 0 3 (Right = Left + (isSplit * numSplits))
	cuda_safe_call_no_sync(cudaMemcpy(m_active_node_list->d_child_right, 
									m_active_node_list->d_child_left, 
									m_active_node_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice)); 
	cuda_init_constant(d_child_offsets, m_active_node_list->num_nodes, num_splits); 
	if (num_splits > 0)
	{
		cuda_array_op<cuda_op_mul, uint32>(d_child_offsets, d_is_split.buf_ptr(), m_active_node_list->num_nodes); 
		cuda_array_op<cuda_op_add, uint32>(m_active_node_list->d_child_right, d_child_offsets, m_active_node_list->num_nodes);
	}
	
	// Child data is up to date. Append to final node list.
	m_final_list->append_list(m_active_node_list, true, false);

	uint32 num_nodes_old = m_active_node_list->num_nodes; 
	m_active_node_list->clear(); 

	// Compact the result into the active list. This avoids using temporary buffers
	// and is possible since the active list is no longer needed. The following elements
	// have to be compacted:
	// d_aabbMinTight, d_aabbMaxTight, d_idxSmallRoot, d_numElems, d_elemMask
	if (num_splits > 0)
	{
		if (m_active_node_list->max_nodes < 2*num_splits)
			m_active_node_list->resize_node_data(2*num_splits); 

		c_cuda_memory<uint32> d_src_addr(num_nodes_old); 
		cuda_gen_compact_addresses(d_is_split.buf_ptr(), num_nodes_old, d_src_addr.buf_ptr()); 

		for (uint32 j = 0; j < 2; ++j)
		{
			uint32 offset = j * num_nodes_old; 

			cuda_set_from_address(m_active_node_list->d_aabb_tight_min+m_active_node_list->num_nodes, 
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_aabb_tight_min+offset, 
								num_splits); 
			cuda_set_from_address(m_active_node_list->d_aabb_tight_max+m_active_node_list->num_nodes, 
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_aabb_tight_max+offset, 
								num_splits); 

			cuda_set_from_address(m_active_node_list->d_small_root_idx+m_active_node_list->num_nodes, 
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_small_root_idx+offset,
								num_splits); 
			cuda_set_from_address(m_active_node_list->d_num_elems_array+m_active_node_list->num_nodes, 
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_num_elems_array+offset, 
								num_splits); 
			cuda_set_from_address(m_active_node_list->d_node_level+m_active_node_list->num_nodes, 
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_node_level+offset, 
								num_splits); 
			cuda_set_from_address(m_active_node_list->d_elem_mask+m_active_node_list->num_nodes,
								d_src_addr.buf_ptr(), 
								m_next_node_list->d_elem_mask+offset, 
								num_splits); 
			
			m_active_node_list->num_nodes += num_splits; 
		}
	}

	if (m_active_node_list->num_nodes > 0)
	{
		// Generate TNA for the new nodes. Currently we only have the masks.
		// At first compute the offsets (idxFirstTri) into the TNA by scanning the
		// element numbers.
		cuda_scan(m_active_node_list->d_num_elems_array, m_active_node_list->num_nodes, false, m_active_node_list->d_first_elem_idx); 
		
		// Get element total.
		uint32 num_elems;
		cuda_reduce_add(num_elems, (uint32*)m_active_node_list->d_num_elems_array, m_active_node_list->num_nodes, (uint32)0);

		// Ensure the active's ENA is large enough.
		if (m_active_node_list->max_elems < 2*CUDA_ALIGN(num_elems))
			m_active_node_list->resize_elem_data(2*CUDA_ALIGN(num_elems)); 

		m_active_node_list->next_free_pos = CUDA_ALIGN(num_elems); 
	}

	kernel_wrapper_gen_ena_from_masks(*m_active_node_list, *m_small_node_list); 
}

void c_kdtree_gpu::preorder_traversal()
{
	c_cuda_memory<uint32> d_node_sizes(m_final_list->num_nodes);

	// Now we have the final tree node list. Notify listeners.
	/*
	for(uint i=0; i<m_vecListeners.size(); i++)
		m_vecListeners[i]->OnFinalNodeList(m_pListFinal);
	*/ 

	// Initialize final node list now.
	m_kd_data = new c_kdtree_data();
	m_kd_data->initialize(m_final_list, m_root_aabb_min, m_root_aabb_max); 

	// Compute maximim node level.
	uint32 max_level; 
	cuda_reduce_max(max_level, m_final_list->d_node_level, m_final_list->num_nodes, (uint32)0);
	
	// At first perform a bottom-top traversal to determine the size of the
	// preorder tree structure.
	for (int lvl = max_level; lvl >= 0; --lvl)
	{
		// Write sizes into d_nodeSizes.
		kernel_wrapper_traversal_up_path(*m_final_list, lvl, d_node_sizes.buf_ptr()); 
	}

	// Now we have the total tree size in root's size. 
	m_kd_data->size_tree = d_node_sizes.read(0); 

	// Allocate preorder tree. Use special request for alignment.
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	cuda_safe_call_no_sync(mem_pool.request_tex((void**)&m_kd_data->d_preorder_tree, m_kd_data->size_tree*sizeof(uint32), "kd-tree_result"));
	
	// Top-down traversal to generate tree from sizes.
	for (uint32 lvl = 0; lvl <= max_level; ++lvl)
	{
		// Generate preorder tree.
		kernel_wrapper_traversal_down_path(*m_final_list, lvl, d_node_sizes.buf_ptr(), m_kd_data->d_node_addresses_array, *m_kd_data); 
	} 
}

void c_kdtree_gpu::compute_nodes_aabbs()
{
	// First compute the bounding boxes of all chunks in parallel.
	if (m_active_node_list->num_elem_points == 1)
		kernel_wrapper_gen_chunk_aabb<1>(*m_active_node_list, *m_chunk_list);
	else 
		kernel_wrapper_gen_chunk_aabb<2>(*m_active_node_list, *m_chunk_list);
	
	
	print_chunks_list(chunk_ofs, m_chunk_list);
	
	// Now compute the tight bounding boxes of all nodes in parallel using
	// segmented reduction. 
	cuda_segmented_reduce_min(m_chunk_list->d_aabb_min, m_chunk_list->d_node_idx, m_chunk_list->num_chunks, make_float4(M_INFINITY), 
		m_active_node_list->d_aabb_tight_min, m_active_node_list->num_nodes);
	
	cuda_segmented_reduce_max(m_chunk_list->d_aabb_max, m_chunk_list->d_node_idx, m_chunk_list->num_chunks, make_float4(-M_INFINITY),
		m_active_node_list->d_aabb_tight_max, m_active_node_list->num_nodes);
}

void c_kdtree_gpu::split_large_nodes(uint32 *d_final_list_index_active)
{
	assert(m_final_list->num_nodes >= m_active_node_list->num_nodes);
	
	// Cut of empty space. This updates the final list to include the cut-off empty space nodes
	// as well as the updated nodes, that are in the active list, too. It keeps track where the
	// active list nodes reside in the final list by updating the parent index array appropriately.
	kernel_wrappper_empty_space_cutting(*m_active_node_list, 
										*m_final_list, 
										m_empty_scene_ratio, 
										d_final_list_index_active);
	
	
	
	// Now we can perform real spatial median splitting to create exactly two child nodes for
	// each active list node (into the next list).
	
	// Check if there is enough space in the next list.
	if(m_next_node_list->max_nodes < 2*m_active_node_list->num_nodes)
		m_next_node_list->resize_node_data(2*m_active_node_list->num_nodes);

	// Perform splitting. Also update the final node child relationship.
	kernel_wrapper_split_large_nodes(*m_active_node_list, *m_next_node_list);

	// Set new number of nodes.
	m_next_node_list->num_nodes = 2*m_active_node_list->num_nodes; 

	active_ofs << "Next node list: " << std::endl; 
	print_node_list(active_ofs, m_next_node_list); 
}

void c_kdtree_gpu::sort_clip_to_child_nodes() 
{
	assert(m_chunk_list->num_chunks > 0);
	
	uint32 next_free_l; 
	c_cuda_memory<uint32> d_counts_unaligned(2*m_active_node_list->num_nodes);
	c_cuda_memory<uint32> d_chunk_counts(2*CUDA_ALIGN(m_chunk_list->num_chunks));
	// Size: 2*m_pListActive->nextFreePos, first half for left marks, second for right marks.
	c_cuda_memory<uint32> d_elem_marks(2*m_active_node_list->next_free_pos);
	
	// Ensure the next's ENA is large enough.
	if (m_next_node_list->max_elems < 2*m_active_node_list->next_free_pos)
		m_next_node_list->resize_elem_data(2*m_active_node_list->next_free_pos);

	// We virtually duplicate the TNA of the active list and write it virtually twice into the
	// temporary TNA of the next list which we do not store explicitly.

	// Zero the marks. This is required since not all marks represent valid elements.
	cuda_safe_call_no_sync(cudaMemset(d_elem_marks.buf_ptr(), 0, 2*m_active_node_list->next_free_pos*sizeof(uint32)));
	
	// Mark the valid elements in the virtual TNA. This is required since not all elements are
	// both in left and right child. The marked flags are of the same size as the virtual TNA and
	// hold 1 for valid tris, else 0. 
	if (m_active_node_list->num_elem_points == 1)
		kernel_wrapper_mark_left_right_elems<1>(*m_active_node_list, *m_chunk_list, d_elem_marks.buf_ptr());
	else 
		kernel_wrapper_mark_left_right_elems<2>(*m_active_node_list, *m_chunk_list, d_elem_marks.buf_ptr());
	
	// Determine per chunk element count for nodes using per block reduction.
	// ... left nodes
	kernel_wrapper_count_elems_chunk(*m_chunk_list, d_elem_marks.buf_ptr(), d_chunk_counts.buf_ptr());
	// ... right nodes
	kernel_wrapper_count_elems_chunk(*m_chunk_list, d_elem_marks.buf_ptr()+m_active_node_list->next_free_pos, d_chunk_counts.buf_ptr()+CUDA_ALIGN(m_chunk_list->num_chunks));
	
	// Perform segmented reduction on per chunk results to get per child nodes results. The owner
	// list is the chunk's idxNode list.
	
	// left nodes 
	cuda_segmented_reduce_add(d_chunk_counts.buf_ptr(), 
							m_chunk_list->d_node_idx, 
							m_chunk_list->num_chunks, 
							(uint32)0, 
							d_counts_unaligned.buf_ptr(), 
							m_active_node_list->num_nodes);

	others_ofs << "d_counts_unaligned: "<< std::endl;
	//  print_device_array(others_ofs, d_counts_unaligned.buf_ptr(), )
	
	// right nodes 
	cuda_segmented_reduce_add(d_chunk_counts.buf_ptr()+CUDA_ALIGN(m_chunk_list->num_chunks),
							m_chunk_list->d_node_idx, 
							m_chunk_list->num_chunks, 
							(uint32)0, 
							d_counts_unaligned.buf_ptr()+m_active_node_list->num_nodes,
							m_active_node_list->num_nodes); 

	next_free_l = compact_elem_data(m_next_node_list, 0, 0, 
									m_active_node_list, 0, 2 * m_active_node_list->num_nodes, 
									d_elem_marks.buf_ptr(), d_counts_unaligned.buf_ptr(), 2);
	

	m_next_node_list->next_free_pos = next_free_l;
}

void c_kdtree_gpu::create_chunk_list(c_kd_node_list *node_list)
{
	if (m_current_chunk_list_src == node_list)
		return;
	
	c_cuda_memory<uint32> d_counts(node_list->num_nodes);
	c_cuda_memory<uint32> d_offsets(node_list->num_nodes);
	
	// Clear old list first.
	m_chunk_list->clear();
	
	// Check if the chunk list is large enough. For now we do NO resizing since we allocated a
	// big enough chunk list (probably too big in most cases).
	uint32 max_chunks = node_list->next_free_pos / KD_CHUNKSIZE + node_list->num_nodes;
	assert(max_chunks <= m_chunk_list->max_chunks);

	if (max_chunks > m_chunk_list->max_chunks)
		yart_log_message("Chunk list too small (max: %d; need: %d).\n", m_chunk_list->max_chunks, max_chunks);

	// Get the count of chunks for each node. Store them in d_counts
	kernel_wrapper_get_chunk_counts(node_list->d_num_elems_array, 
								node_list->num_nodes, 
								d_counts.buf_ptr());
	
	// Scan the counts to d_offsets. Use exclusive scan cause then we have
	// the start index for the i-th node in the i-th element of d_offsets.
	cuda_scan(d_counts.buf_ptr(), node_list->num_nodes, false, d_offsets.buf_ptr()); 

	// Generate chunk list.
	kernel_wrapper_kd_gen_chunks(node_list->d_num_elems_array, 
								node_list->d_first_elem_idx, 
								node_list->num_nodes, 
								d_offsets.buf_ptr(), 
								*m_chunk_list);
	
	// Set total number of chunks.
	cuda_reduce_add<uint32>(m_chunk_list->num_chunks, d_counts.buf_ptr(), node_list->num_nodes, (uint32)0); 
	
}

uint32 c_kdtree_gpu::compact_elem_data(c_kd_node_list *dest_node_list, 
									uint32 dest_offset, 
									uint32 node_offset, 
									c_kd_node_list *src_node_list, 
									uint32 src_offset, 
									uint32 num_src_nodes, 
									uint32 *d_valid_marks, 
									uint32 *d_counts_unaligned, 
									uint32 num_segments /* = 1 */)
{
	assert(num_segments > 0);
	
	c_cuda_memory<uint32> d_offsets_unaligned(num_src_nodes); 
	c_cuda_memory<uint32> d_counts_aligned(num_src_nodes); 
	c_cuda_memory<uint32> d_offsets_aligned(num_src_nodes);

	// Get unaligned offsets.
	cuda_scan(d_counts_aligned.buf_ptr(), num_src_nodes, false, d_offsets_aligned.buf_ptr());

	// Get aligned counts to temp buffer to avoid uncoalesced access (here and later).
	cuda_align_counts(d_counts_aligned.buf_ptr(), d_counts_unaligned, num_src_nodes);

	// Scan to produce the offsets 
	cuda_scan(d_counts_aligned.buf_ptr(), num_src_nodes, false, d_offsets_aligned.buf_ptr());
	
	// Now copy in resulting *unaligned* counts and aligned offsets.
	cuda_safe_call_no_sync(cudaMemcpy(dest_node_list->d_num_elems_array+node_offset, d_counts_unaligned, num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(dest_node_list->d_first_elem_idx+node_offset, d_offsets_aligned.buf_ptr(), num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
	
	// Offset d_idxFirstElem by destOffset.
	if (dest_offset > 0)
		cuda_constant_add<uint32>(dest_node_list->d_first_elem_idx+node_offset, num_src_nodes, dest_offset);

	// Get next free position by reduction. Using two device-to-host memcpys were slower than this!
	uint32 aligned_sum, next_free_pos; 
	cuda_reduce_add(aligned_sum, d_counts_aligned.buf_ptr(), num_src_nodes, (uint32)0);

	others_ofs << "d_counts.aligned: " << std::endl;
	print_device_array(others_ofs, d_counts_aligned.buf_ptr(), num_src_nodes); 
	
	next_free_pos = aligned_sum + dest_offset;

	// Move elements only if there are any!
	if (aligned_sum == 0)
		yart_log_message("kd - WARNING: Splitting nodes and zero elements on one side!\n");
	else
	{
		// Algorithm to compute addresses (where to put which compacted element).
		// N        -  N  -  N        -  (node starts)
		// 0  0  0  0 -1  0 -1  0  0  0  (inverse difference of counts at aligned offsets, offsetted by one node!)
		// 0  0  0  0 -1 -1 -2 -2 -2 -2  (inclusive scan to distribute values)
		// 0  1  2  3  3  4  4  5  6  7  (AddIdentity)
		// x  x  x  -  x  -  x  x  x  -
		// -> Addresses where to put which compact elements.
		c_cuda_memory<uint32> d_buf_temp(aligned_sum);
		c_cuda_memory<uint32> d_addresses(aligned_sum);
		c_cuda_memory<uint32> d_src_addr(num_segments*src_node_list->next_free_pos);

		// Nothing to compute if there is only one source node!
		if (num_src_nodes == 1)
			cuda_init_identity(d_addresses.buf_ptr(), aligned_sum); 
		else
		{
			// 1. Get inverse count differences (that is unaligned - aligned = -(aligned - unaligned)).
			c_cuda_memory<uint32> d_inv_count_diffs(num_src_nodes);
			cuda_safe_call_no_sync(cudaMemcpy(d_inv_count_diffs.buf_ptr(), d_counts_unaligned, num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
			cuda_array_op<cuda_op_sub, uint32>(d_inv_count_diffs.buf_ptr(), d_counts_aligned.buf_ptr(), num_src_nodes);

			// 2. Copy inverse count differences to node starts in d_bufTemp, but offset them by one node
			//    so that the first node gets zero as difference.
			cuda_safe_call_no_sync(cudaMemset(d_addresses.buf_ptr(), 0, aligned_sum*sizeof(uint32)));
			//    Only need to set numSourceNode-1 elements, starting with d_offsetsAligned + 1 as
			//    first address.
			cuda_set_at_address((uint32*)d_addresses.buf_ptr(), d_offsets_aligned.buf_ptr() + 1, (uint32*)d_inv_count_diffs.buf_ptr(), num_src_nodes - 1);
			
			// 3. Scan the differences to distribute them to the other node elements.
			//    Do this inplace in d_addresses.
			cuda_scan(d_addresses.buf_ptr(), aligned_sum, true, d_addresses.buf_ptr());

			// 4. Add identity
			cuda_add_identity(d_addresses.buf_ptr(), aligned_sum);
		}

		// To avoid multiple calls of compact we just compact an identity array once
		// to generate the source addresses. 
		for (uint32 seg = 0; seg < num_segments; ++seg)
			cuda_init_identity(d_src_addr.buf_ptr() + seg * src_node_list->next_free_pos, src_node_list->next_free_pos);

		cuda_safe_call_no_sync(cudaMemset(d_buf_temp.buf_ptr(), 0, aligned_sum*sizeof(uint32)));
		cuda_compact(d_src_addr.buf_ptr(), d_valid_marks, num_segments*src_node_list->next_free_pos, d_buf_temp.buf_ptr(), d_temp_val.buf_ptr());
		
		// Ensure the destination list element data is large enough.
		if (dest_node_list->max_elems < next_free_pos)
			dest_node_list->resize_elem_data(next_free_pos);

		// Now we can generate the source addresses by setting the compacted data
		// at the positions defined by d_addresses:
		// N     - N   N     -  (node starts)
		// 0 1 2 3 3 4 5 6 7 8  (identity - seg-scan result)
		// A B C D E F G H 0 0  (compact)
		// A B C D D E F G H 0	(SetFromAddress(d_srcAddr, d_addresses, compact)
		//       -           -
		// NOTE: We assume here that the compacted array has at least as many elements
		//       as the address array. Therefore the 0-read is possible. It doesn't
		//       destroy the result because the 0 values don't matter.

		cuda_set_from_address((uint32*)d_src_addr.buf_ptr(), 
							d_addresses.buf_ptr(), 
							(uint32*)d_buf_temp.buf_ptr(), 
							aligned_sum);
		
		cuda_set_from_address(dest_node_list->d_node_elems_list + dest_offset, 
							d_src_addr.buf_ptr(), 
							src_node_list->d_node_elems_list, 
							aligned_sum);

		cuda_set_from_address(dest_node_list->d_elem_point1 + dest_offset, 
							d_src_addr.buf_ptr(), 
							src_node_list->d_elem_point1, 
							aligned_sum);
		
		if (dest_node_list->num_elem_points == 2)
			cuda_set_from_address(dest_node_list->d_elem_point2 + dest_offset, 
								d_src_addr.buf_ptr(), 
								src_node_list->d_elem_point2, 
								aligned_sum);
	}

	return next_free_pos;
}

void c_kdtree_gpu::update_small_list(uint32 *d_final_list_index_active)
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	
	c_cuda_memory<uint32> d_counts_unaligned(m_next_node_list->num_nodes);
	c_cuda_memory<uint32> d_node_marks(m_next_node_list->num_nodes);
	c_cuda_memory<uint32> d_small_parents(m_next_node_list->num_nodes);
	c_cuda_memory<uint32> d_node_list_offsets(m_next_node_list->num_nodes);
	
	uint32 num_small, num_large, next_free_small, next_free_large;
	
	// Group next list elements into chunks.
	create_chunk_list(m_next_node_list);
	
	// Mark small nodes. Result to d_nodeMarks. Small node parent array to d_smallParents.
	kernel_wrapper_mark_small_nodes(*m_next_node_list, d_final_list_index_active, d_node_marks.buf_ptr(), d_small_root_parents);
	
	// Compact small root parents array to get d_smallRootParents for new small roots.
	cuda_compact(d_small_parents.buf_ptr(), 
				d_node_marks.buf_ptr(), 
				m_next_node_list->num_nodes, 
				d_small_root_parents+m_small_node_list->num_nodes, 
				d_temp_val.buf_ptr());
	
	// Store the number of small nodes, but do not update the list's value for now.
	// This ensures we still have the old value.
	cuda_safe_call_no_sync(cudaMemcpy(&num_small, d_temp_val.buf_ptr(), sizeof(uint32), cudaMemcpyDeviceToHost));
	
	// Get element markers to d_elemMarks. Zero first to avoid marked empty space.
	c_cuda_memory<uint32> d_elem_marks(2*m_next_node_list->next_free_pos);
	cuda_safe_call_no_sync(cudaMemset(d_elem_marks.buf_ptr(), 0, 2*m_next_node_list->next_free_pos*sizeof(uint32)));
	uint32 *d_is_small_elem = d_elem_marks.buf_ptr(); 
	uint32 *d_is_large_elem = d_elem_marks.buf_ptr() + m_next_node_list->next_free_pos;
	kernel_wrapper_mark_elems_by_node_size(*m_chunk_list, m_next_node_list->d_num_elems_array, d_is_small_elem, d_is_large_elem);
	

	if (num_small == 0)
		next_free_small = m_small_node_list->next_free_pos;
	else 
	{
		// Compact element count array to get d_numElems for small list.
		cuda_compact(m_next_node_list->d_num_elems_array, 
					d_node_marks.buf_ptr(), 
					m_next_node_list->num_nodes, 
					d_counts_unaligned.buf_ptr(), 
					d_temp_val.buf_ptr()); 
		
		// Scan nodes marks to get node list offsets.
		cuda_scan(d_node_marks.buf_ptr(), m_next_node_list->num_nodes, false, d_node_list_offsets.buf_ptr());
		
		// Resize small list if required.
		if (m_small_node_list->num_nodes + num_small > m_small_node_list->max_nodes)
			m_small_node_list->resize_node_data(m_small_node_list->num_nodes + num_small);

		// Now remove small nodes and add them to the small list.
		kernel_wrapper_move_nodes(*m_next_node_list, *m_small_node_list, d_node_marks.buf_ptr(), d_node_list_offsets.buf_ptr(), true);

		// Need to update left & right child pointers in current active list here. This is
		// neccessary since we remove the small nodes and the current pointers point to
		// an array enriched with those small nodes.
		// Scan isSmall array d_nodeMarks to get an array we can subtract from the left/right indices
		// to get the final positions of the large nodes. Example:
		//
		// 0 1 2 3 4 5 6  (d_childLeft)
		// 0 1 1 0 1 0 0  (d_nodeMarks)
		// 0 0 1 2 2 3 3  (Scan d_nodeMarks -> d_nodeListOffsets)
		// 0 1 1 1 2 2 3  (d_childLeft - d_nodeListOffsets)
		cuda_array_op<cuda_op_sub, uint32>(m_active_node_list->d_child_left, 
										d_node_list_offsets.buf_ptr(), 
										m_active_node_list->num_nodes);
		cuda_array_op<cuda_op_sub, uint32>(m_active_node_list->d_child_right, 
										d_node_list_offsets.buf_ptr()+m_active_node_list->num_nodes, 
										m_active_node_list->num_nodes);

		// Update ENA of small list. This can be done by compacting the next list ENA
		// using the marks for demarcation.
		next_free_small = compact_elem_data(m_small_node_list, 
											m_small_node_list->next_free_pos, 
											m_small_node_list->num_nodes, 
											m_next_node_list,
											0, 
											num_small, 
											d_is_small_elem, 
											d_counts_unaligned.buf_ptr());
	}
	
	// Do the same with the remaining large nodes. But do not calculate the markers
	// as they can be computed by inversion.
	//  - Large node markers. 
	cuda_inverse_binary(d_node_marks.buf_ptr(), m_next_node_list->num_nodes);

	// Now we have updated child and split information. We need to update the corresponding
	// final node list entries to reflect these changes.
	kernel_wrapper_update_final_list_child_info(*m_active_node_list, *m_final_list, d_final_list_index_active);

	// Compact next list. This is slightly difficult since we cannot do this inplace.
	// Instead we abuse the active list as temporary target. This is no problem since
	// this is the last stage of the large node processing. The active list is cleared
	// after this (under the name next list, as they are swapped).
	m_active_node_list->clear();

	// Compact element count array to get d_numElems for large list.
	cuda_compact(m_next_node_list->d_num_elems_array, 
				d_node_marks.buf_ptr(), 
				m_next_node_list->num_nodes, 
				d_counts_unaligned.buf_ptr(), 
				d_temp_val.buf_ptr());
	cuda_safe_call_no_sync(cudaMemcpy(&num_large, d_temp_val.buf_ptr(), sizeof(uint32), cudaMemcpyDeviceToHost));
	
	if (num_large == 0)
		next_free_large = 0; 
	else 
	{
		// Scan nodes marks to get node list offsets.
		cuda_scan(d_node_marks.buf_ptr(),
				m_next_node_list->num_nodes, 
				false, d_node_list_offsets.buf_ptr());
		
		// Resize active list if required.
		if (num_large > m_active_node_list->max_nodes)
			m_active_node_list->resize_node_data(num_large);

		kernel_wrapper_move_nodes(*m_next_node_list, *m_active_node_list, d_node_marks.buf_ptr(), d_node_list_offsets.buf_ptr(), false);

		// Compact ENA of next list.
		next_free_large = compact_elem_data(m_active_node_list,	
											0, 
											0,
											m_next_node_list, 
											0, 
											num_large, 
											d_is_large_elem, 
											d_counts_unaligned.buf_ptr());
		
		// Now the new next list is the active list as we used it to temporarily build the
		// next list to avoid overwriting. We make the active list real by swapping it with
		// the next list. This avoids copying stuff back.
		
		c_kd_node_list *ptemp = m_active_node_list; 
		m_active_node_list = m_next_node_list; 
		m_next_node_list = ptemp; 
		m_active_node_list->clear(); 

		// update counts 
		m_small_node_list->num_nodes += num_small;
		m_small_node_list->next_free_pos = next_free_small; 
		m_next_node_list->num_nodes = num_large; 
		m_next_node_list->next_free_pos = next_free_large; 

	}

} 

