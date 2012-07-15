#include "kdtree_gpu.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"
#include "cuda_primitives.h"
#include "math_utils.h"

extern "C"
void kernel_set_kd_params(uint32 samll_node_max);

extern "C"
void kernel_init_kd_gpu();

extern "C"
void kernel_wrapper_get_chunk_counts(uint32 *d_num_elems_node, uint32 num_nodes, uint32 *d_out_chunk_counts);

extern "C"
void kernel_wrapper_kd_gen_chunks(uint32 *d_num_elems_array, 
								uint32 *d_idx_first_elem_array, 
								uint32 num_nodes, 
								uint32 *d_offsets, 
								c_kd_chunk_list& chunk_list);

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

//////////////////////////////////////////////////////////////////////////

c_kdtree_gpu::c_kdtree_gpu(size_t num_input_elems, uint32 num_elems_points, float3 root_aabb_min, float3 root_aabb_max)
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
	, m_empty_scene_ratio(0.25f)
	, m_small_nodes_max(64) 
	, m_max_query_radius(1.0f)
{ 
	kernel_init_kd_gpu();
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

	// Small node stage
	small_node_stage();

	// Generates final node list m_pKDData.
	preorder_traversal();

	post_build();

	return true; 
}

void c_kdtree_gpu::large_node_stage()
{
	/*
	// Iterate until the active list is empty, that is until there are
	// no more large nodes to work on.
	while (!m_active_node_list->is_empty())
	{
		// Append the active list to the final node list. This is done even if the child information
		// aren't available, yet. It is fixed later.
		m_final_list->append_list(m_active_node_list, false, true);

		// Keeps track of where the current active list's nodes are in the final node list.
		c_cuda_memory<uint32> d_final_list_idx(m_active_node_list->num_nodes);
		cuda_init_identity(d_final_list_idx.get_writable_buf_ptr(), m_active_node_list->num_nodes);
		cuda_constant_op<cuda_op_add, uint32>(d_final_list_idx.get_buf_ptr(), m_active_node_list->num_nodes, m_final_list->num_nodes - m_active_node_list->num_nodes);
	
		// Clear the next list which stores the nodes for the next step.
		m_next_node_list->clear();

		// Process the active nodes. This generated both small nodes and new
		// next nodes.
		process_large_nodes(d_final_list_idx.get_writable_buf_ptr());

		// Swap active and next list for next pass.
		c_kd_node_list *temp = m_active_node_list; 
		m_active_node_list = m_next_node_list; 
		m_next_node_list = temp;
	}
	*/
}

void c_kdtree_gpu::process_large_nodes(uint32 *d_final_list_idx_active)
{
	assert(m_active_node_list->is_empty());
	
	// Group elements into chunks.
	create_chunk_list(m_active_node_list);
	
	// Compute per node bounding boxes.
	compute_nodes_aabbs(); 

	// Split large nodes.
	split_large_nodes(d_final_list_idx_active);

	// Sort and clip elements to child nodes.
	sort_clip_to_nodes();

	// Now we have unclipped element bounds, so perform split clipping. Per default, this
	// does nothing. Clipping has to be realized in subclasses.
	
}

void c_kdtree_gpu::compute_nodes_aabbs()
{
	// First compute the bounding boxes of all chunks in parallel.
	if (m_active_node_list->num_elem_points == 1)
		kernel_wrapper_gen_chunk_aabb<1>(*m_active_node_list, *m_chunk_list);
	else 
		kernel_wrapper_gen_chunk_aabb<2>(*m_active_node_list, *m_chunk_list);
	
	
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
	kernel_wrappper_empty_space_cutting(*m_active_node_list, *m_final_list, m_empty_scene_ratio, d_final_list_index_active);
	
	// Now we can perform real spatial median splitting to create exactly two child nodes for
	// each active list node (into the next list).
	
	// Check if there is enough space in the next list.
	if(m_next_node_list->max_nodes < 2*m_active_node_list->num_nodes)
		m_next_node_list->resize_node_data(2*m_active_node_list->num_nodes);

	// Perform splitting. Also update the final node child relationship.
	kernel_wrapper_split_large_nodes(*m_active_node_list, *m_next_node_list);

	// Set new number of nodes.
	m_next_node_list->num_nodes = 2*m_active_node_list->num_nodes;
	
	
}

void c_kdtree_gpu::sort_clip_to_nodes()
{
	assert(m_chunk_list->num_chunks > 0);
	
	uint32 next_free_l; 
	c_cuda_memory<uint32> d_counts_unaligned(2*m_active_node_list->num_nodes);
	c_cuda_memory<uint32> d_chunk_counts(2*CUDA_ALIGN(m_chunk_list->num_chunks));
	// Size: 2*m_pListActive->nextFreePos, first half for left marks, second for right marks.
	c_cuda_memory<uint32> d_elem_marks(2*m_active_node_list->next_free_pos);
	
	// Ensure the next's ENA is large enough.
	if (m_next_node_list->max_nodes < 2*m_active_node_list->next_free_pos)
		m_next_node_list->resize_elem_data(2*m_active_node_list->next_free_pos);

	// We virtually duplicate the TNA of the active list and write it virtually twice into the
	// temporary TNA of the next list which we do not store explicitly.

	// Zero the marks. This is required since not all marks represent valid elements.
	cuda_safe_call_no_sync(cudaMemset(d_elem_marks.get_writable_buf_ptr(), 0, 2*m_active_node_list->next_free_pos*sizeof(uint32)));
	
	// Mark the valid elements in the virtual TNA. This is required since not all elements are
	// both in left and right child. The marked flags are of the same size as the virtual TNA and
	// hold 1 for valid tris, else 0. 
	if (m_active_node_list->num_elem_points == 1)
		kernel_wrapper_mark_left_right_elems<1>(*m_active_node_list, *m_chunk_list, d_elem_marks.get_writable_buf_ptr());
	else 
		kernel_wrapper_mark_left_right_elems<2>(*m_active_node_list, *m_chunk_list, d_elem_marks.get_writable_buf_ptr());
	
	// Determine per chunk element count for nodes using per block reduction.
	// ... left nodes
	kernel_wrapper_count_elems_chunk(*m_chunk_list, d_elem_marks.get_buf_ptr(), d_chunk_counts.get_writable_buf_ptr());
	// ... right nodes
	kernel_wrapper_count_elems_chunk(*m_chunk_list, d_elem_marks.get_buf_ptr()+m_active_node_list->next_free_pos, d_chunk_counts.get_writable_buf_ptr()+CUDA_ALIGN(m_chunk_list->num_chunks));
	
	// Perform segmented reduction on per chunk results to get per child nodes results. The owner
	// list is the chunk's idxNode list.
	
	// left nodes 
	cuda_segmented_reduce_add(d_chunk_counts.get_buf_ptr(), 
		m_chunk_list->d_node_idx, 
		m_chunk_list->num_chunks, 
		(uint32)0, 
		d_counts_unaligned.get_writable_buf_ptr(), 
		m_active_node_list->num_nodes);
	
	// right nodes 
	cuda_segmented_reduce_add(d_chunk_counts.get_buf_ptr()+CUDA_ALIGN(m_chunk_list->num_chunks),
		m_chunk_list->d_node_idx, 
		m_chunk_list->num_chunks, 
		(uint32)0, 
		d_counts_unaligned.get_writable_buf_ptr()+m_active_node_list->num_nodes,
		m_active_node_list->num_nodes); 

	next_free_l = compact_elem_data(m_next_node_list, 0, 0, 
									m_active_node_list, 0, 2 * m_active_node_list->num_nodes, 
									d_elem_marks.get_buf_ptr(), d_counts_unaligned.get_buf_ptr(), 2);
	

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
	kernel_wrapper_get_chunk_counts(node_list->d_num_elems_array, node_list->num_nodes, d_counts.get_writable_buf_ptr());
	
	// Scan the counts to d_offsets. Use exclusive scan cause then we have
	// the start index for the i-th node in the i-th element of d_offsets.
	c_cuda_primitives& cuda_prims = c_cuda_primitives::get_instance();
	cuda_prims.scan(d_counts.get_buf_ptr(), node_list->num_nodes, false, d_offsets.get_writable_buf_ptr()); 

	// Generate chunk list.
	kernel_wrapper_kd_gen_chunks(node_list->d_num_elems_array, node_list->d_first_elem_idx, node_list->num_nodes, d_offsets.get_buf_ptr(), *m_chunk_list);
	
	// Set number of chunks.
	cuda_reduce_add<uint32>(m_chunk_list->num_chunks, d_counts.get_buf_ptr(), node_list->num_nodes, (uint32)0); 
	
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
	cuda_scan(d_counts_aligned.get_buf_ptr(), num_src_nodes, false, d_offsets_aligned.get_writable_buf_ptr());

	// Get aligned counts to temp buffer to avoid uncoalesced access (here and later).
	cuda_align_counts(d_counts_aligned.get_writable_buf_ptr(), d_counts_unaligned, num_src_nodes);

	cuda_scan(d_counts_aligned.get_buf_ptr(), num_src_nodes, false, d_offsets_aligned.get_writable_buf_ptr());
	
	// Now copy in resulting *unaligned* counts and aligned offsets.
	cuda_safe_call_no_sync(cudaMemcpy(dest_node_list->d_num_elems_array+node_offset, d_counts_unaligned, num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(dest_node_list->d_first_elem_idx+node_offset, d_offsets_aligned.get_buf_ptr(), num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
	
	// Offset d_idxFirstElem by destOffset.
	if (dest_offset > 0)
		cuda_constant_add<uint32>(dest_node_list->d_first_elem_idx+node_offset, num_src_nodes, dest_offset);

	// Get next free position by reduction. Using two device-to-host memcpys were slower than this!
	uint32 aligned_sum, next_free_pos; 
	cuda_reduce_add(aligned_sum, d_counts_aligned.get_buf_ptr(), num_src_nodes, (uint32)0);
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
			cuda_init_identity(d_addresses.get_writable_buf_ptr(), aligned_sum); 
		else
		{
			// 1. Get inverse count differences (that is unaligned - aligned = -(aligned - unaligned)).
			c_cuda_memory<uint32> d_inv_count_diffs(num_src_nodes);
			cuda_safe_call_no_sync(cudaMemcpy(d_inv_count_diffs.get_writable_buf_ptr(), d_counts_unaligned, num_src_nodes*sizeof(uint32), cudaMemcpyDeviceToDevice));
			cuda_array_op<cuda_op_sub, uint32>(d_inv_count_diffs.get_writable_buf_ptr(), d_counts_aligned.get_buf_ptr(), num_src_nodes);

			// 2. Copy inverse count differences to node starts in d_bufTemp, but offset them by one node
			//    so that the first node gets zero as difference.
			cuda_safe_call_no_sync(cudaMemset(d_addresses.get_writable_buf_ptr(), 0, aligned_sum*sizeof(uint32)));
			//    Only need to set numSourceNode-1 elements, starting with d_offsetsAligned + 1 as
			//    first address.
			cuda_set_at_address((uint32*)d_addresses.get_writable_buf_ptr(), d_offsets_aligned.get_buf_ptr() + 1, (uint32*)d_inv_count_diffs.get_buf_ptr(), num_src_nodes - 1);
			
			// 3. Scan the differences to distribute them to the other node elements.
			//    Do this inplace in d_addresses.
			cuda_scan(d_addresses.get_buf_ptr(), aligned_sum, true, d_addresses.get_writable_buf_ptr());

			// 4. Add identity
			cuda_add_identity(d_addresses.get_writable_buf_ptr(), aligned_sum);
		}

		// To avoid multiple calls of compact we just compact an identity array once
		// to generate the source addresses. 
		for (uint32 seg = 0; seg < num_segments; ++seg)
			cuda_init_identity(d_src_addr.get_writable_buf_ptr() + seg * src_node_list->next_free_pos, src_node_list->next_free_pos);

		cuda_safe_call_no_sync(cudaMemset(d_buf_temp.get_writable_buf_ptr(), 0, aligned_sum*sizeof(uint32)));
		cuda_compact(d_src_addr.get_buf_ptr(), d_valid_marks, num_segments*src_node_list->next_free_pos, d_buf_temp.get_writable_buf_ptr(), d_temp_val.get_writable_buf_ptr());
		

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

		cuda_set_from_address((uint32*)d_src_addr.get_writable_buf_ptr(), d_addresses.get_buf_ptr(), (uint32*)d_buf_temp.get_buf_ptr(), aligned_sum);
		
		cuda_set_from_address(dest_node_list->d_node_elems_list + dest_offset, d_src_addr.get_buf_ptr(), src_node_list->d_node_elems_list, aligned_sum);

		cuda_set_from_address(dest_node_list->d_elem_point1 + dest_offset, d_src_addr.get_buf_ptr(), src_node_list->d_elem_point1, aligned_sum);
		
		if (dest_node_list->num_elem_points == 2)
			cuda_set_from_address(dest_node_list->d_elem_point2 + dest_offset, d_src_addr.get_buf_ptr(), src_node_list->d_elem_point2, aligned_sum);
	}

	return next_free_pos;
}