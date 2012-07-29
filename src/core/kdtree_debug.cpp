#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <Windows.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "kdtree_debug.h"
#include "kdtree_gpu.h"

void print_node_list(std::ostream& os, c_kd_node_list *node_list)
{
	uint32 num_nodes = node_list->num_nodes;  
	uint32 max_elems = node_list->max_elems; 
	
	uint32 *h_first_elem_idx = new uint32[num_nodes]; 
	uint32 *h_node_level = new uint32[num_nodes];
	uint32 *h_num_elems = new uint32[num_nodes]; 
	uint32 *h_elems = new uint32[node_list->max_elems]; 
	uint32 *h_left = new uint32[node_list->num_nodes]; 
	uint32 *h_right = new uint32[node_list->num_nodes];
	
	float4 *h_aabb_min_inherit = new float4[num_nodes]; 
	float4 *h_aabb_max_inherit = new float4[num_nodes]; 
	float4 *h_aabb_min_tight = new float4[num_nodes]; 
	float4 *h_aabb_max_tight = new float4[num_nodes]; 

	elem_mask_t *h_elem_marks = new elem_mask_t[num_nodes];

	cuda_safe_call_no_sync(cudaMemcpy(h_first_elem_idx, 
									node_list->d_first_elem_idx,
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 
	cuda_safe_call_no_sync(cudaMemcpy(h_num_elems, 
									node_list->d_num_elems_array,
									num_nodes*sizeof(uint32),
									cudaMemcpyDeviceToHost)); 
	cuda_safe_call_no_sync(cudaMemcpy(h_node_level, 
									node_list->d_node_level, 
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 

	// Copy back AABB inherit
	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_min_inherit, 
									node_list->d_aabb_inherit_min,
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_max_inherit, 
									node_list->d_aabb_inherit_max, 
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	// Copy back AABB tight
	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_min_tight, 
									node_list->d_aabb_tight_min,
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_max_tight, 
									node_list->d_aabb_tight_max, 
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	// Copy left and right child 
	cuda_safe_call_no_sync(cudaMemcpy(h_left,
									node_list->d_child_left,
									num_nodes*sizeof(uint32),
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_right,
									node_list->d_child_right, 
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 


	// Copy back elements 
	cuda_safe_call_no_sync(cudaMemcpy(h_elems, 
									node_list->d_node_elems_list, 
									max_elems*sizeof(uint32),
									cudaMemcpyDeviceToHost)); 

	// Copy elem marks 
	cuda_safe_call_no_sync(cudaMemcpy(h_elem_marks,
									node_list->d_elem_mask,
									num_nodes*sizeof(elem_mask_t), 
									cudaMemcpyDeviceToHost)); 
	
	for (uint32 i = 0; i < node_list->num_nodes; ++i)
	{
		os << "Node #" << i << std::endl; 
		os << "\t" << "First elem idx: " << h_first_elem_idx[i] << std::endl;
		os << "\t" << "Num elems: " << h_num_elems[i] << std::endl; 
		os << "\t" << "Node level: " << h_node_level[i] << std::endl; 

		os << "\t" << "AABB Min inherit: " << h_aabb_min_inherit[i].x << " "
										<< h_aabb_min_inherit[i].y << " "
										<< h_aabb_min_inherit[i].z << " "
										<< h_aabb_min_inherit[i].w << std::endl; 

		os << "\t" << "AABB Max inherit: " << h_aabb_max_inherit[i].x << " "
										<< h_aabb_max_inherit[i].y << " "
										<< h_aabb_max_inherit[i].z << " "
										<< h_aabb_max_inherit[i].w << std::endl; 

		os << "\t" << "AABB Min tight: " << h_aabb_min_tight[i].x << " "
										<< h_aabb_min_tight[i].y << " "
										<< h_aabb_min_tight[i].z << " "
										<< h_aabb_min_tight[i].w << std::endl; 

		os << "\t" << "AABB Max tight: " << h_aabb_max_tight[i].x << " "
										<< h_aabb_max_tight[i].y << " "
										<< h_aabb_max_tight[i].z << " "
										<< h_aabb_max_tight[i].w << std::endl; 

		os << "\t" << "Left child: " << h_left[i] << std::endl; 
		os << "\t" << "Right child: " << h_right[i] << std::endl; 

		os << "\t" << "Elem masks: " << h_elem_marks[i] << std::endl; 
		
		// Print elements list 
		for (uint32 e = 0; e < h_num_elems[i]; ++e)
		{
			os << h_elems[h_first_elem_idx[i]+e] << " | " ; 
			if ( e != 0 && e % 30 == 0)
				os << std::endl;
		}

		os << std::endl; 
	}

	os << std::endl; 

	SAFE_DELETE_ARRAY(h_first_elem_idx); 
	SAFE_DELETE_ARRAY(h_node_level); 
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_elems); 
	SAFE_DELETE_ARRAY(h_elem_marks);
	SAFE_DELETE_ARRAY(h_left); 
	SAFE_DELETE_ARRAY(h_right); 

	SAFE_DELETE_ARRAY(h_aabb_min_inherit); 
	SAFE_DELETE_ARRAY(h_aabb_max_inherit); 
	SAFE_DELETE_ARRAY(h_aabb_min_tight); 
	SAFE_DELETE_ARRAY(h_aabb_max_tight); 

}

void print_final_node_list(std::ostream& os, c_kd_final_node_list *final_node_list)
{
	uint32 num_nodes = final_node_list->num_nodes;  
	uint32 max_elems = final_node_list->max_elems; 

	uint32 *h_first_elem_idx = new uint32[num_nodes]; 
	uint32 *h_node_level = new uint32[num_nodes];
	uint32 *h_num_elems = new uint32[num_nodes]; 
	uint32 *h_elems = new uint32[final_node_list->max_elems]; 
	uint32 *h_left = new uint32[final_node_list->num_nodes]; 
	uint32 *h_right = new uint32[final_node_list->num_nodes];

	float4 *h_aabb_min = new float4[num_nodes]; 
	float4 *h_aabb_max = new float4[num_nodes]; 

	cuda_safe_call_no_sync(cudaMemcpy(h_first_elem_idx, 
		final_node_list->d_first_elem_idx,
		num_nodes*sizeof(uint32), 
		cudaMemcpyDeviceToHost)); 
	cuda_safe_call_no_sync(cudaMemcpy(h_num_elems, 
		final_node_list->d_num_elems,
		num_nodes*sizeof(uint32),
		cudaMemcpyDeviceToHost)); 
	cuda_safe_call_no_sync(cudaMemcpy(h_node_level, 
		final_node_list->d_node_level, 
		num_nodes*sizeof(uint32), 
		cudaMemcpyDeviceToHost));  
	
	// Copy back AABB 
	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_min, 
		final_node_list->d_aabb_min,
		num_nodes*sizeof(float4), 
		cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_max, 
		final_node_list->d_aabb_max, 
		num_nodes*sizeof(float4), 
		cudaMemcpyDeviceToHost)); 

	// Copy back elements 
	cuda_safe_call_no_sync(cudaMemcpy(h_elems, 
		final_node_list->d_elem_node_assoc, 
		max_elems*sizeof(uint32),
		cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_left, 
		final_node_list->d_child_left, 
		num_nodes*sizeof(uint32), 
		cudaMemcpyDeviceToHost)); 
	
	cuda_safe_call_no_sync(cudaMemcpy(h_right, 
		final_node_list->d_child_right, 
		num_nodes*sizeof(uint32), 
		cudaMemcpyDeviceToHost));
	
	for (uint32 i = 0; i < final_node_list->num_nodes; ++i)
	{
		os << "Node #" << i << std::endl; 
		os << "\t" << "First elem idx: " << h_first_elem_idx[i] << std::endl;
		os << "\t" << "Num elems: " << h_num_elems[i] << std::endl; 
		os << "\t" << "Node level: " << h_node_level[i] << std::endl; 

		os << "\t" << "AABB Min: " 
			<< h_aabb_min[i].x << " "
			<< h_aabb_min[i].y << " "
			<< h_aabb_min[i].z << " "
			<< h_aabb_min[i].w << std::endl; 

		os << "\t" << "AABB Max: " 
			<< h_aabb_max[i].x << " "
			<< h_aabb_max[i].y << " "
			<< h_aabb_max[i].z << " "
			<< h_aabb_max[i].w << std::endl; 

		// Print elements list 
		for (uint32 e = 0; e < h_num_elems[i]; ++e)
		{
			os << h_elems[h_first_elem_idx[i]+e] << " | " ; 
			if ( e != 0 && e % 30 == 0)
				os << std::endl;
		}

		os << std::endl; 
	}

	os << std::endl; 
	
	SAFE_DELETE_ARRAY(h_first_elem_idx); 
	SAFE_DELETE_ARRAY(h_node_level); 
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_elems); 
	SAFE_DELETE_ARRAY(h_left); 
	SAFE_DELETE_ARRAY(h_right); 

	SAFE_DELETE_ARRAY(h_aabb_min); 
	SAFE_DELETE_ARRAY(h_aabb_max); 
}

void print_chunks_list(std::ostream& os, c_kd_chunk_list *chunks_list)
{
	uint32 num_chunks = chunks_list->num_chunks; 

	// Create host data 
	uint32 *h_node_idx = new uint32[num_chunks]; 
	uint32 *h_num_elems = new uint32[num_chunks]; 
	uint32 *h_first_elem = new uint32[num_chunks]; 
	float4 *h_aabb_min = new float4[num_chunks]; 
	float4 *h_aabb_max = new float4[num_chunks]; 

	// Copy back data
	cuda_safe_call_no_sync(cudaMemcpy(h_node_idx, 
									chunks_list->d_node_idx, 
									num_chunks*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_num_elems, 
									chunks_list->d_num_elems, 
									num_chunks*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_first_elem, 
									chunks_list->d_first_elem_idx,
									num_chunks*sizeof(uint32),
									cudaMemcpyDeviceToHost));
	
	// Copy back AABB 
	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_min, 
									chunks_list->d_aabb_min,
									num_chunks*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 
	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_max, 
									chunks_list->d_aabb_max, 
									num_chunks*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 
	
	for (uint32 i = 0; i < num_chunks; ++i)
	{ 
		os << "Chunk #" << i << std::endl;
		os << "\t" << "Node #" << h_node_idx[i] << std::endl;
		os << "\t" << "Elems: " << h_num_elems[i] << std::endl; 
		os << "\t" << "First elem idx: " << h_first_elem[i] << std::endl;
		os << "\t" << "AABB Min: " << h_aabb_min[i].x << " "
									<< h_aabb_min[i].y << " "
									<< h_aabb_min[i].z << " "
									<< h_aabb_min[i].w << std::endl; 

		os << "\t" << "AABB Max: " << h_aabb_max[i].x << " "
									<< h_aabb_max[i].y << " "
									<< h_aabb_max[i].z << " "
									<< h_aabb_max[i].w << std::endl; 
	} 

	os << std::endl;
	
	SAFE_DELETE_ARRAY(h_node_idx);
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_first_elem); 

	SAFE_DELETE_ARRAY(h_aabb_min); 
	SAFE_DELETE_ARRAY(h_aabb_max);
}

void print_splits_list(std::ostream& os, c_kd_split_list *splits_list, c_kd_node_list *small_list)
{
	c_cuda_memory<uint32> d_aligned_split_counts(small_list->num_nodes);
	cuda_align_counts(d_aligned_split_counts.buf_ptr(), splits_list->d_num_splits, small_list->num_nodes);

	uint32 num_split_total; 
	cuda_reduce_add(num_split_total, 
					(uint32*)d_aligned_split_counts.buf_ptr(), 
					small_list->num_nodes, 
					(uint32)0); 

	uint32 num_small_nodes = small_list->num_nodes;

	uint32 *h_num_splits = new uint32[num_small_nodes]; 
	uint32 *h_first_split_idx = new uint32[num_small_nodes]; 
	uint32 *h_split_count_aligned = new uint32[num_small_nodes];
	
	cuda_safe_call_no_sync(cudaMemcpy(h_num_splits, 
									splits_list->d_num_splits, 
									num_small_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost));
	cuda_safe_call_no_sync(cudaMemcpy(h_first_split_idx, 
									splits_list->d_first_split_idx, 
									num_small_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost));
	cuda_safe_call_no_sync(cudaMemcpy(h_split_count_aligned, 
									d_aligned_split_counts.buf_ptr(), 
									num_small_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost));
	
	float *h_split_pos = new float[num_split_total];
	cuda_safe_call_no_sync(cudaMemcpy(h_split_pos, 
									splits_list->d_split_pos_array,
									num_split_total*sizeof(float), 
									cudaMemcpyDeviceToHost));

	elem_mask_t *h_split_left_masks = new elem_mask_t[num_split_total]; 
	cuda_safe_call_no_sync(cudaMemcpy(h_split_left_masks, 
									splits_list->d_mask_left_array,
									num_split_total*sizeof(elem_mask_t), 
									cudaMemcpyDeviceToHost)); 

	elem_mask_t *h_split_right_masks = new elem_mask_t[num_split_total];
	cuda_safe_call_no_sync(cudaMemcpy(h_split_right_masks, 
									splits_list->d_mask_right_array,
									num_split_total*sizeof(elem_mask_t), 
									cudaMemcpyDeviceToHost));

	std::ios::fmtflags old_flags = os.flags(); 
	os.setf(std::ios::left, std::ios::adjustfield); 
	int prec = 4; 
	int width = 4;

	for (uint32 i = 0; i < num_small_nodes; ++i)
	{	
		os << "small node #" << i << std::endl;
		for (uint32 j = 0; j < h_num_splits[i]; ++j)
		{
			os << std::setprecision(prec) 
				<< std::setw(width) 
				<< std::setfill(' ') 
				<< h_split_pos[h_first_split_idx[i]+j] << " | "; 

			os << std::endl; 

			os << std::setprecision(prec) 
				<< std::setw(width) 
				<< std::setfill(' ') 
				<< h_split_left_masks[h_first_split_idx[i]+j] << " | "; 

			os << std::endl;

			os << std::setprecision(prec) 
				<< std::setw(width) 
				<< std::setfill(' ') 
				<< h_split_right_masks[h_first_split_idx[i]+j] << " | ";
			
			os << std::endl;
		}
		os << std::endl;
	}

	os << std::endl;
	os.setf(old_flags); 

	SAFE_DELETE_ARRAY(h_split_pos);
	SAFE_DELETE_ARRAY(h_split_left_masks); 
	SAFE_DELETE_ARRAY(h_split_right_masks);
	SAFE_DELETE_ARRAY(h_num_splits); 
	SAFE_DELETE_ARRAY(h_first_split_idx); 
	SAFE_DELETE_ARRAY(h_split_count_aligned);
}

template <typename T> 
void print_device_array(std::ostream& os, T *d_array, uint32 count)
{
	T *h_array = new T[count];

	cuda_safe_call_no_sync(cudaMemcpy(h_array, d_array, count*sizeof(T), cudaMemcpyDeviceToHost));

	std::ios::fmtflags old_flags = os.flags(); 
	os.setf(std::ios::left, std::ios::adjustfield); 
	int prec = 4; 
	int width = 4;
	for (uint32 i = 0; i < count; ++i)
	{
		os << std::setprecision(prec) << std::setw(width) << std::setfill(' ') << h_array[i] << " | "; 
		if (i != 0 && i % 20 == 0)
			os << std::endl;
	}

	os << std::endl;

	os.setf(old_flags); 

	SAFE_DELETE_ARRAY(h_array); 
}

void print_kdtree_data(std::ostream& os, c_kdtree_data *kd_data)
{
	uint32 num_nodes = kd_data->num_nodes;
	uint32 tree_size = kd_data->size_tree; 
	
	uint32 *h_node_addresses = new uint32[num_nodes]; 
	uint32 *h_num_elems = new uint32[num_nodes]; 
	uint32 *h_child_left = new uint32[num_nodes];
	uint32 *h_child_right = new uint32[num_nodes]; 
	float4 *h_node_extent = new float4[num_nodes];

	cuda_safe_call_no_sync(cudaMemcpy(h_node_addresses, 
									kd_data->d_node_addresses, 
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_num_elems, 
									kd_data->d_num_elems, 
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 
	
	cuda_safe_call_no_sync(cudaMemcpy(h_child_left, 
									kd_data->d_child_left,
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 
	
	cuda_safe_call_no_sync(cudaMemcpy(h_child_right,
									kd_data->d_child_right, 
									num_nodes*sizeof(uint32), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_node_extent, 
									kd_data->d_node_extent, 
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 
	os << "Root aabb min: " 
		<< kd_data->aabb_root_min.x << " "
		<< kd_data->aabb_root_min.y << " "
		<< kd_data->aabb_root_min.z << " "
		<< std::endl; 

	os << "Root aabb max: " 
		<< kd_data->aabb_root_max.x << " "
		<< kd_data->aabb_root_max.y << " "
		<< kd_data->aabb_root_max.z << " "
		<< std::endl; 
	 
	os << "KD-Tree Data: " << std::endl;
	for (uint32 i = 0; i < num_nodes; ++i)
	{ 
		os << "Node #" << i << std::endl; 
		os << "\t" << "Address: " << h_node_addresses[i] << std::endl;
		os << "\t" << "Elem counts: " << h_num_elems[i] << std::endl;
		os << "\t" << "Child left: " << h_child_left[i] << std::endl; 
		os << "\t" << "Child right: " << h_child_right[i] << std::endl;
		os << "\t" << "Node extent: " 
			<< h_node_extent[i].x << ", " 
			<< h_node_extent[i].y << ", "
			<< h_node_extent[i].z << ", "
			<< h_node_extent[i].w << std::endl; 
	}

	SAFE_DELETE_ARRAY(h_node_addresses);
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_child_left);
	SAFE_DELETE_ARRAY(h_child_right); 
	SAFE_DELETE_ARRAY(h_node_extent); 
}

void print_kdtree_preorder_data(std::ostream& os, c_kdtree_data *kd_data, c_kd_final_node_list *final_list)
{
	uint32 *preorder_tree = new uint32[kd_data->size_tree]; 
	
	cuda_safe_call_no_sync(cudaMemcpy(preorder_tree, 
									kd_data->d_preorder_tree, 
									kd_data->size_tree*sizeof(uint32), 
									cudaMemcpyDeviceToHost));

	// Traversal stack.
	uint32 todo_addr[KD_MAX_HEIGHT]; 
	uint32 todo_level[KD_MAX_HEIGHT]; 
	uint32 todo_pos = 0; 
	int addr_node = 0; 
	int cur_level = -1; 
	int max_level_trav = -1; 

	uint32 num_leafs = 0; 
	uint32 num_leaf_elems = 0;

	while (addr_node != -1)
	{
		uint32 idx_node = preorder_tree[addr_node];
		uint32 is_leaf = idx_node & 0x80000000; 
		idx_node &= 0x7fffffff; 
		cur_level++; 
		max_level_trav = std::max(max_level_trav, cur_level); 
		
		if (!is_leaf && idx_node >= kd_data->num_nodes)
		{
			os << "Illegal node idx " << idx_node << "at address " << addr_node << std::endl; 
			assert(false); 
		}
		
		if (!is_leaf)
		{
			assert(todo_pos < KD_MAX_HEIGHT); 
			
			// Internal node 
			uint32 left = addr_node + 1 + 2; 
			uint32 right = preorder_tree[addr_node+1] & 0xfffffff; 
			todo_addr[todo_pos] = right; 
			todo_level[todo_pos] = cur_level; 
			todo_pos++; 

			addr_node = left;
		}
		else
		{
			uint32 num = preorder_tree[addr_node+1]; 

			num_leafs++; 
			num_leaf_elems += num; 

			if (todo_pos > 0)
			{
				todo_pos--; 
				addr_node = todo_addr[todo_pos]; 
				cur_level = todo_level[todo_pos]; 
			}
			else 
				break; 
		}
	}
	
	uint32 max_level; 
	cuda_reduce_max(max_level, final_list->d_node_level, final_list->num_nodes, (uint32)0); 
	
	os << "KD-Tree nodes: " << kd_data->num_nodes << std::endl; 
	os << "KD-Tree size: " << kd_data->size_tree << std::endl; 
	os << "KD-Tree height: " << max_level+1 << std::endl;
	os << "KD-Tree leafs: " << num_leafs << std::endl; 
	os << "Avg elems: " << (float)num_leaf_elems/(float)num_leafs, num_leaf_elems; 
	os << std::endl; 

	SAFE_DELETE_ARRAY(preorder_tree); 
}

template void print_device_array<uint32>(std::ostream& os, uint32 *d_array, uint32 count);
template void print_device_array<float>(std::ostream& os, float *d_array, uint32 count);