#include "kernel_data.h"
#include "kdtree_kernel_data.h"
#include "cuda_utils.h"

// ---------------------------------------------------------------------
/*
/// \brief	Slim version of TriangleData to avoid parameter space overflows.
///
///			This structure stores the vertices only. Used for trianlge kd-tree construction kernels.
*/ 
// ---------------------------------------------------------------------
struct tri_vertex_data
{
#ifdef __cplusplus
public:
	/// Initializes helper struct from given triangle data.
	void initialize(const c_triangle_data& td)
	{
		for(uint i=0; i<3; i++)
			d_verts[i] = td.d_verts[i];
		num_tris = td.num_tris;
	}
#endif // __cplusplus

	/// See TriangleData::d_verts.
	float4 *d_verts[3];
	/// See TriangleData::numTris.
	uint32 num_tris; 
	
};

struct kd_tri_node_list
{

#ifdef __cplusplus
public:
	/// Initializes helper struct from given node list.
	void initialize(const c_kd_node_list& src)
	{
		num_nodes = src.num_nodes;
		d_first_elem_idx = src.d_first_elem_idx;
		d_num_elems = src.d_num_elems_array;
		d_min_tight_aabb = src.d_aabb_tight_min;
		d_min_inherit_aabb = src.d_aabb_inherit_min;
		d_max_tight_aabb = src.d_aabb_tight_max;
		d_max_inherit_aabb = src.d_aabb_inherit_max;
		d_node_elems_list = src.d_node_elems_list;
		d_aabb_tri_min = src.d_elem_point1;
		d_aabb_tri_max = src.d_elem_point2;
	}
#endif // __cplusplus

	/// See KDNodeList::numNodes.
	uint32 num_nodes;

	/// See KDNodeList::d_idxFirstElem.
	uint32* d_first_elem_idx;
	/// See KDNodeList::d_numElems.
	uint32* d_num_elems;
	/// See KDNodeList::d_aabbMinTight.
	float4* d_min_tight_aabb;
	/// See KDNodeList::d_aabbMaxTight.
	float4* d_max_tight_aabb;
	/// See KDNodeList::d_aabbMinInherit.
	float4* d_min_inherit_aabb;
	/// See KDNodeList::d_aabbMaxInherit.
	float4* d_max_inherit_aabb;

	/// See KDNodeList::d_elemNodeAssoc.
	uint32 *d_node_elems_list;
	/// See KDNodeList::d_elemPoint1.
	float4* d_aabb_tri_min;
	/// See KDNodeList::d_elemPoint2.
	float4* d_aabb_tri_max;
	
};


//////////////////////////////////////////////////////////////////////////

// ---------------------------------------------------------------------
/*
// \brief	Computes the axis aligned bounding boxes for all triangles in the given node list. 
///
///			It is assumed that the ENA is initialized with identity values. Therefore no ENA access
///			is required to find the triangles corresponding to node's elements.
*/ 
// ---------------------------------------------------------------------

__global__ void kernel_gen_tris_aabb(kd_tri_node_list root_list, tri_vertex_data td)
{
	uint32 idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < td.num_tris)
	{
		float4 v0 = td.d_verts[0][idx];
		float4 v1 = td.d_verts[1][idx];
		float4 v2 = td.d_verts[2][idx];

		root_list.d_aabb_tri_min[idx] = make_float4(fminf(v0.x, fminf(v1.x, v2.x)), fminf(v0.y, fminf(v1.y, v2.y)), fminf(v0.z, fminf(v1.z, v2.z)), 0.0f);
		root_list.d_aabb_tri_max[idx] = make_float4(fmaxf(v0.x, fmaxf(v1.x, v2.x)), fmaxf(v0.y, fmaxf(v1.y, v2.y)), fmaxf(v0.z, fmaxf(v1.z, v2.z)), 0.0f);
	}
}
 

extern "C"
void kernel_kd_generate_tri_aabbs(const c_kd_node_list& root_list, const c_triangle_data &tri_data)
{ 
	dim3 block_size = dim3(256, 1, 1);
	dim3 grid_size = dim3(CUDA_DIVUP(tri_data.num_tris, block_size.x), 1, 1);

	tri_vertex_data verts_data; 
	verts_data.initialize(tri_data);

	kd_tri_node_list node_tri_list;
	node_tri_list.initialize(root_list);
	
	// Generate AABBs using kernel 
	kernel_gen_tris_aabb<<<grid_size, block_size>>>(node_tri_list, verts_data);

}