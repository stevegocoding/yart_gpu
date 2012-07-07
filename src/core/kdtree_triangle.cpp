#include "kdtree_triangle.h"
#include "kernel_data.h"
#include "cuda_defs.h"
#include "cuda_utils.h"

extern "C"
void kernel_kd_generate_tri_aabbs(const c_kd_node_list& root_list, const c_triangle_data &tri_data);

void c_kdtree_triangle::add_root_node(c_kd_node_list *node_list)
{
	cuda_safe_call_no_sync(cudaMemset(node_list->d_first_elem_idx, 0, sizeof(uint32)));
	cuda_safe_call_no_sync(cudaMemset(node_list->d_node_level, 0, sizeof(uint32)));
	uint32 temp = m_tri_data->num_tris;
	cuda_safe_call_no_sync(cudaMemcpy(node_list->d_num_elems_array, &temp, sizeof(uint32), cudaMemcpyHostToDevice));

	// Set inherited bounds to scene bounds.
	float4 aabb_min = make_float4(m_root_aabb_min);
	float4 aabb_max = make_float4(m_root_aabb_max);
	cuda_safe_call_no_sync(cudaMemcpy(node_list->d_aabb_inherit_min, &aabb_min, sizeof(float4), cudaMemcpyHostToDevice));
	cuda_safe_call_no_sync(cudaMemcpy(node_list->d_aabb_inherit_max, &aabb_max, sizeof(float4), cudaMemcpyHostToDevice));
	
	// All elements are contained in the first node, therefore the list is just the identity relation.
	cuda_init_identity(node_list->d_node_elems_list, m_tri_data->num_tris);

	node_list->num_nodes = 1; 

	// Align first free tri index.
	node_list->next_free_pos = CUDA_ALIGN(m_tri_data->num_tris);
	
	// Compute AABBs for all triangles in root node in parallel. This initializes
	// the d_elemPoint1/2 members of pList.
	
	kernel_kd_generate_tri_aabbs(*node_list, *m_tri_data); 
}: