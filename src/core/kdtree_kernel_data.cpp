#include "kdtree_kernel_data.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

void c_kd_node_list::initialize(uint32 _max_nodes, uint32 _max_elems, uint32 _num_elem_points /* = 2 */)
{
	assert(_max_nodes > 0 && _max_elems > 0 && _num_elem_points <= 2 && _num_elem_points > 0);

	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();

	// Starting with zero nodes.
	num_nodes = 0; 
	max_nodes = CUDA_ALIGN(_max_nodes); 
	next_free_pos = 0; 

	// Ensure aligned access (16 * 4 byte = 64 byte alignment) to get coalesced access in kernels.
	max_elems = CUDA_ALIGN(_max_elems);
	num_elem_points = std::max((uint32)1, std::min((uint32)2, _num_elem_points));

	// Allocate memory

	// Node sized 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_first_elem_idx, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_num_elems_array, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_node_level, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_split_axis, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_split_pos, max_nodes*sizeof(float), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_child_left, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_child_right, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_aabb_tight_min, max_nodes*sizeof(float4), "kd-tree node", 256));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_aabb_tight_max, max_nodes*sizeof(float4), "kd-tree node", 256));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_aabb_inherit_min, max_nodes*sizeof(float4), "kd-tree node", 256));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_aabb_inherit_max, max_nodes*sizeof(float4), "kd-tree node", 256));

	// Small node
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_small_root_idx, max_nodes*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_elem_mask, max_nodes*sizeof(elem_mask_t), "kd-tree node", 128));

	// Element sized 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_node_elems_list, max_elems*sizeof(uint), "kd-tree node"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_elem_point1, max_elems*sizeof(float4), "kd-tree node"));
	if(num_elem_points > 1)
		cuda_safe_call_no_sync(mem_pool.request((void**)&d_elem_point2, max_elems*sizeof(float4), "kd-tree node"));
}

void c_kd_node_list::append_list(c_kd_node_list *nodes_list, bool append_data)
{

}

void c_kd_node_list::resize_node_data(uint32 required)
{

}

void c_kd_node_list::resize_elem_data(uint32 required)
{

}

void c_kd_node_list::clear()
{

}

bool c_kd_node_list::is_empty() const 
{
	return num_nodes == 0; 
}

void c_kd_node_list::free_memory()
{

}
 