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
	assert(nodes_list);
	
	// Resize node data if required.
	if (num_nodes + nodes_list->num_nodes)
	{
		resize_node_data(num_nodes + nodes_list->num_nodes);
	}

	// Copy data 
	cuda_safe_call_no_sync(cudaMemcpy(d_first_elem_idx + num_nodes, 
									nodes_list->d_first_elem_idx, 
									nodes_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_num_elems_array + num_nodes, 
									nodes_list->d_num_elems_array,
									nodes_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_node_level + num_nodes, 
									nodes_list->d_node_level,
									nodes_list->num_nodes*sizeof(uint32),
									cudaMemcpyDeviceToDevice));
	
	// Copy AABB
	cuda_safe_call_no_sync(cudaMemcpy(d_aabb_tight_min + num_nodes, 
									nodes_list->d_aabb_tight_min,
									nodes_list->num_nodes*sizeof(float4),
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_aabb_tight_max + num_nodes,
									nodes_list->d_aabb_tight_max, 
									nodes_list->num_nodes*sizeof(float4),
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_aabb_inherit_min + num_nodes, 
									nodes_list->d_aabb_inherit_min, 
									nodes_list->num_nodes*sizeof(float4),
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_aabb_inherit_max + num_nodes, 
									nodes_list->d_aabb_inherit_max,
									nodes_list->num_nodes+sizeof(float4),
									cudaMemcpyDeviceToDevice));

	// Copy split information.
	cuda_safe_call_no_sync(cudaMemcpy(d_split_axis + num_nodes, 
									nodes_list->d_split_axis,
									nodes_list->num_nodes*sizeof(uint32),
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_split_pos + num_nodes, 
									nodes_list->d_split_pos, 
									nodes_list->num_nodes*sizeof(float),
									cudaMemcpyDeviceToDevice));

	// Copy child relationship data. Need to update this after that cause the child relationship
	// indices are relative to the next node list's indices.
	cuda_safe_call_no_sync(cudaMemcpy(d_child_left + num_nodes, 
									nodes_list->d_child_left,
									nodes_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_child_right + num_nodes, 
									nodes_list->d_child_right, 
									nodes_list->num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToDevice));
	cuda_constant_add<uint32>(d_child_right + num_nodes, nodes_list->num_nodes, num_nodes + nodes_list->num_nodes);
	
	// Copy element masks and small root indices 
	cuda_safe_call_no_sync(cudaMemcpy(d_elem_mask + num_nodes, 
									nodes_list->d_elem_mask, 
									nodes_list->num_nodes*sizeof(elem_mask_t), 
									cudaMemcpyDeviceToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(d_small_root_idx + num_nodes, 
									nodes_list->d_small_root_idx, 
									nodes_list->num_nodes*sizeof(uint32),
									cudaMemcpyDeviceToDevice));
	
	if (append_data)
	{
		// Check if there's still enough space in the ENA.
		if (next_free_pos + nodes_list->next_free_pos > max_elems)
		{
			resize_elem_data(next_free_pos + nodes_list->next_free_pos); 
		}
		
		// Copy in new other ENA.
		cuda_safe_call_no_sync(cudaMemcpy(d_node_elems_list + next_free_pos, 
										nodes_list->d_node_elems_list, 
										nodes_list->next_free_pos*sizeof(uint32), 
										cudaMemcpyDeviceToDevice));

		// Copy in element points. 
		cuda_safe_call_no_sync(cudaMemcpy(d_elem_point1 + next_free_pos, 
										nodes_list->d_elem_point1, 
										nodes_list->next_free_pos*sizeof(float4), 
										cudaMemcpyDeviceToDevice));
		
		cuda_safe_call_no_sync(cudaMemcpy(d_elem_point2 + next_free_pos, 
										nodes_list->d_elem_point2, 
										nodes_list->next_free_pos*sizeof(float4), 
										cudaMemcpyDeviceToDevice));
		
		// Shift first element indices in d_idxFirstElem for new nodes.
		if (next_free_pos != 0)
			cuda_constant_add<uint32>(d_first_elem_idx + num_nodes, nodes_list->num_nodes, next_free_pos); 
	}

	// Now update counts. 
	if (append_data)
		next_free_pos += nodes_list->next_free_pos; 
	
	num_nodes += nodes_list->num_nodes; 
}

void c_kd_node_list::resize_node_data(uint32 required)
{
	assert(required > max_nodes);

	// Add some space to avoid multiple resizes.
	uint32 new_max = std::max(2*max_nodes, required);

	new_max = cuda_resize_mem(&d_first_elem_idx, max_nodes, new_max);
	cuda_resize_mem(&d_num_elems_array, max_nodes, new_max);
	cuda_resize_mem(&d_node_level, max_nodes, new_max);
	
	cuda_resize_mem(&d_aabb_tight_min, max_elems, new_max);  
	cuda_resize_mem(&d_aabb_tight_max, max_elems, new_max);
	cuda_resize_mem(&d_aabb_inherit_min, max_elems, new_max);
	cuda_resize_mem(&d_aabb_inherit_max, max_elems, new_max);
	
	cuda_resize_mem(&d_split_axis, max_nodes, new_max);
	cuda_resize_mem(&d_split_pos, max_nodes, new_max);
	cuda_resize_mem(&d_child_left, max_nodes, new_max);
	cuda_resize_mem(&d_child_right, max_nodes, new_max);
	
	cuda_resize_mem(&d_small_root_idx, max_nodes, new_max);
	cuda_resize_mem(&d_elem_mask, max_nodes, new_max);

	max_nodes = new_max; 
}

void c_kd_node_list::resize_elem_data(uint32 required)
{
	assert(required > max_elems);
	
	// Add some space to avoid multiple resizes.
	uint32 new_max = std::max(2 * max_elems, required);
	
	// Element count might change due to alignment.
	new_max = cuda_resize_mem(&d_node_elems_list, max_elems, new_max); 
	cuda_resize_mem(&d_elem_point1, max_elems, new_max);
	if (num_elem_points > 1)
		cuda_resize_mem(&d_elem_point2, max_elems, new_max);

	max_elems = new_max;
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
 