#include "kdtree_gpu.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"
#include "cuda_primitives.h"

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
	
	
}