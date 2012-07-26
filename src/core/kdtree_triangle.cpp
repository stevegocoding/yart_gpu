#include "kdtree_triangle.h"
#include "kernel_data.h"
#include "cuda_defs.h"
#include "cuda_utils.h"

extern "C"
void kernel_wrapper_gen_tri_aabbs(const c_kd_node_list& root_list, const c_triangle_data &tri_data);

extern "C"
void kernel_wrapper_do_split_clipping(const c_kd_node_list& active_list, 
									const c_kd_node_list& next_list,
									const c_kd_chunk_list& chunks_list,
									const c_triangle_data& tri_data);


c_kdtree_triangle::c_kdtree_triangle(const c_triangle_data& tri_data)
	: c_kdtree_gpu(tri_data.num_tris, 2, tri_data.aabb_min, tri_data.aabb_max, 0.25f, 9)
	, m_tri_data(&tri_data)
{
}

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
	
	kernel_wrapper_gen_tri_aabbs(*node_list, *m_tri_data); 
} 

void c_kdtree_triangle::perform_split_clipping(c_kd_node_list *parent_list, c_kd_node_list *child_list)
{
	create_chunk_list(child_list);
	
	kernel_wrapper_do_split_clipping(*parent_list, *child_list, *m_chunk_list, *m_tri_data);
}