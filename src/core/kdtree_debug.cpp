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
	
	float4 *h_aabb_min_inherit = new float4[num_nodes]; 
	float4 *h_aabb_max_inherit = new float4[num_nodes]; 
	float4 *h_aabb_min_tight = new float4[num_nodes]; 
	float4 *h_aabb_max_tight = new float4[num_nodes]; 

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
									node_list->d_aabb_inherit_min,
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	cuda_safe_call_no_sync(cudaMemcpy(h_aabb_max_tight, 
									node_list->d_aabb_inherit_max, 
									num_nodes*sizeof(float4), 
									cudaMemcpyDeviceToHost)); 

	// Copy back elements 
	cuda_safe_call_no_sync(cudaMemcpy(h_elems, 
									node_list->d_node_elems_list, 
									max_elems*sizeof(uint32),
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

		// Print elements list 
		for (uint32 e = 0; e < h_num_elems[i]; ++e)
		{
			os << h_elems[e] << " | " ; 
			if ( e != 0 && e % 30 == 0)
				os << std::endl;
		}
	}

	SAFE_DELETE_ARRAY(h_first_elem_idx); 
	SAFE_DELETE_ARRAY(h_node_level); 
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_elems); 

	SAFE_DELETE_ARRAY(h_aabb_min_inherit); 
	SAFE_DELETE_ARRAY(h_aabb_max_inherit); 
	SAFE_DELETE_ARRAY(h_aabb_min_tight); 
	SAFE_DELETE_ARRAY(h_aabb_max_tight); 

}

void print_final_node_list(std::ostream& os, c_kd_final_node_list *final_node_list)
{
	
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
	
	SAFE_DELETE_ARRAY(h_node_idx);
	SAFE_DELETE_ARRAY(h_num_elems); 
	SAFE_DELETE_ARRAY(h_first_elem); 

	SAFE_DELETE_ARRAY(h_aabb_min); 
	SAFE_DELETE_ARRAY(h_aabb_max);
}


void print_device_array(std::ostream& os, uint32 *d_array, uint32 count)
{
	uint32 *h_array = new uint32[count];
	
	cuda_safe_call_no_sync(cudaMemcpy(h_array, d_array, count*sizeof(uint32), cudaMemcpyDeviceToHost));
	
	for (uint32 i = 0; i < count; ++i)
	{
		os << h_array[i] << " | "; 
		if (i != 0 && i % 20 == 0)
			os << std::endl;
	}
	
	os << std::endl;
	

	SAFE_DELETE_ARRAY(h_array); 
}