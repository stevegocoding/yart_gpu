#include "kernel_data.h"
#include "kdtree_kernel_data.h"
#include "cuda_utils.h"
#include "math_utils.h"

extern cudaDeviceProp device_props; 

// ---------------------------------------------------------------------
/*
/// \brief	Simple point list for split clipping.
/// 		
/// 		Keeps track of the clipped triangle points during kd-tree construction's splitt
/// 		clipping. As all segments of the triangle are clipped against all sides of the node's
/// 		AABB, the number of segments, i.e. the number of points might increase. 
*/ 
// ---------------------------------------------------------------------
struct point_list
{
	uint32 count; 

	// ---------------------------------------------------------------------
	/*
	///			Currently I allow up to 8 clipped points. This was enough for my test scenes. Using
	///			less points I got overflows. Note that there have to be at least 6 points, as there can
	///			be clipped triangles with that many points. 
	*/ 
	// ---------------------------------------------------------------------
	float3 pts[8];
};

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
		d_aabb_min_tight = src.d_aabb_tight_min;
		d_aabb_min_inherit = src.d_aabb_inherit_min;
		d_aabb_max_tight = src.d_aabb_tight_max;
		d_aabb_max_inherit = src.d_aabb_inherit_max;
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
	float4* d_aabb_min_tight;
	/// See KDNodeList::d_aabbMaxTight.
	float4* d_aabb_max_tight;
	/// See KDNodeList::d_aabbMinInherit.
	float4* d_aabb_min_inherit;
	/// See KDNodeList::d_aabbMaxInherit.
	float4* d_aabb_max_inherit;

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
/// \brief	Clips a segment at a given bounding box side.
/// 		
/// 		Checks on which side the segment end points are. If not both points are in the inside,
/// 		a clipped point is generated and added to the clipped point list. Furthermore, when
/// 		the end point is inside, it is also added right after the clipped point. 
*/ 
// ---------------------------------------------------------------------
__device__ void device_clip_seg(float3 pt_start, float3 pt_end, uint32 axis, float axis_value, float sense, point_list& io_clipped)
{
	float *start = (float*)&pt_start;
	float *end = (float*)&pt_end;

	bool is_inside_start = (start[axis] - axis_value) * sense >= 0; 
	bool is_inside_end = (end[axis] - axis_value) * sense >= 0;
	
	if (is_inside_start != is_inside_end)
	{
		float t = (axis_value - start[axis]) / (end[axis] - start[axis]);
		float3 pt_hit = pt_start + t * (pt_end - pt_start);

		// Ensure the hit is exactly on bounds. 
		((float*)&pt_hit)[axis] = axis_value;

		io_clipped.pts[io_clipped.count] = pt_hit;
		io_clipped.count++; 
	}

	// Do not forget the end point if it lies inside.
	if (is_inside_end)
	{
		io_clipped.pts[io_clipped.count] = pt_end;
		io_clipped.count++;
	}
}

// ---------------------------------------------------------------------
/*
/// \brief	Clips all segments at a given bounding box side.
///
///			Clipping of a given segment is performed by dev_ClipSegment(). All segments are
///			handled in an iterative way.
*/ 
// ---------------------------------------------------------------------

__device__ void device_clip_segments(const point_list& pts_list, uint32 axis, float axis_value, float sense, point_list& out_clipped)
{
	out_clipped.count = 0; 

	for (uint32 i = 0; i < pts_list.count; ++i)
	{
		const float3& pt1 = pts_list.pts[i];
		uint32 ip1 = i + i; 
		if (ip1 == pts_list.count)
			ip1 = 0; 
		const float3& pt2 = pts_list.pts[ip1];

		device_clip_seg(pt1, pt2, axis, axis_value, sense, out_clipped);
	}
}

__device__ void device_clip_tri_to_bounds(float3 verts[3], float3 aabb_min, float3 aabb_max, float *aabb_tri_min, float *aabb_tri_max, point_list& out_list)
{
	point_list ohter_list;
	
	// initial list 
	out_list.count = 3; 
#pragma unroll
	
	for (uint32 i = 0; i < 3; ++i)
		out_list.pts[i] = verts[i]; 

	// Clip against axes.
	// Minimum side: Sense == 1.f
	// Maximum side: Sense == -1.f

	// No need to clip in case the current triangle bounds have no extent in current axis.
	// The neccessary clipping is performed by the caller then.
	if (aabb_tri_min[0] != aabb_tri_max[0])
	{
		device_clip_segments(out_list, 0, aabb_min.x, 1.0f, ohter_list);
		device_clip_segments(ohter_list, 0, aabb_max.x, -1.0f, out_list);
	}

	if (aabb_tri_min[1] != aabb_tri_max[1])
	{
		device_clip_segments(out_list, 1, aabb_min.y, 1.0f, ohter_list);
		device_clip_segments(ohter_list, 1, aabb_max.y, -1.0f, out_list);
	}

	if (aabb_tri_min[2] != aabb_tri_max[2])
	{
		device_clip_segments(out_list, 2, aabb_min.z, 1.0f, ohter_list);
		device_clip_segments(ohter_list, 2, aabb_max.z, -1.0f, out_list);
	}
	
}

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

// ---------------------------------------------------------------------
/*
/// \brief	Perform split clipping for given node list.
///
///			Split clipping is performed to reduce the triangle AABBs according to the node AABBs.
///			It was proposed by Havran, "Heuristic Ray Shooting Algorithms", 2000. Each triangle is
///			clipped by dev_ClipTriToBounds(). The kernel works on all triangles in parallel by
///			using a chunk list constructed for the considered node list. 
*/ 
// ---------------------------------------------------------------------
__global__ void kernel_do_split_clipping(kd_tri_node_list next_tri_list, c_kd_chunk_list chunks_list, 
										tri_vertex_data td, uint32 *d_split_axis, float *d_split_pos)
{
	uint32 chk = CUDA_GRID2DINDEX;
	uint32 idx = threadIdx.x;

	__shared__ uint s_num_tris_chunk;
	__shared__ uint s_idx_first_tri;
	__shared__ float3 s_aabb_node_min;
	__shared__ float3 s_aabb_node_max;
	__shared__ uint s_split_axis;
	__shared__ float s_split_pos;

	if (threadIdx.x == 0)
	{
		s_num_tris_chunk = chunks_list.d_num_elems[chk];
		uint32 idx_node = chunks_list.d_node_idx[chk];
		s_idx_first_tri = chunks_list.d_first_elem_idx[chk];

		// Split Information
		uint32 idx_node_parent = idx_node;
		uint32 num_nodes_parent = next_tri_list.num_nodes >> 1;
		if (idx_node >= num_nodes_parent)
			idx_node_parent -= num_nodes_parent;
		s_split_axis = d_split_axis[idx_node_parent];
		s_split_pos = d_split_pos[idx_node_parent];

		// Get node's inherited bounds. Tight bounds are not yet available at this point.
		s_aabb_node_min = make_float3(next_tri_list.d_aabb_min_inherit[idx_node]);
		s_aabb_node_max = make_float3(next_tri_list.d_aabb_max_inherit[idx_node]);
	}

	__syncthreads(); 

	uint32 split_axis = s_split_axis;
	uint32 split_pos = s_split_pos;
	if (idx < s_num_tris_chunk)
	{
		uint32 idx_tna = s_idx_first_tri + idx; 

		// Triangle AABB
		float4 temp = next_tri_list.d_aabb_tri_min[idx_tna];
		float aabb_tri_min[3] = {temp.x, temp.y, temp.z};
		temp = next_tri_list.d_aabb_tri_max[idx_tna];
		float aabb_tri_max[3] = {temp.x, temp.y, temp.z};

		// Pre-read for coalesced access...
		uint32 idx_tri = next_tri_list.d_node_elems_list[idx_tna];
		
		// We use < here since triangles on the split plane do not lead to clipping.
		bool is_left = aabb_tri_min[split_axis] < split_pos;
		bool is_right = split_pos < aabb_tri_max[split_axis];
		
		// 
		if (is_left && is_right)
		{
			// Read triangle vertices.
			// WARNING: We CANNOT reduce clipping to the non-split axes. The clipping on the
			//          split axis is CRITICAL, e.g. for triangles lying fully in a parent node.
			float3 verts[3];
			for (uint32 i = 0; i < 3; ++i)
				verts[i] = make_float3(td.d_verts[i][idx_tri]);
			
			// Clip the triangle to our node bounds.
			point_list clipped_list; 
			device_clip_tri_to_bounds(verts, s_aabb_node_min, s_aabb_node_max, aabb_tri_min, aabb_tri_max, clipped_list);

			for (uint32 a = 0; a < 3; ++a)
			{
				aabb_tri_min[a] = M_INFINITY;
				aabb_tri_max[a] = -M_INFINITY;

				for (uint32 i = 0; i < clipped_list.count; ++i)
				{
					aabb_tri_min[a] = fminf(((float*)&clipped_list.pts[i])[a], aabb_tri_min[a]);
					aabb_tri_max[a] = fmaxf(((float*)&clipped_list.pts[i])[a], aabb_tri_max[a]);
				}
			}
		}
		
		// Ensure the bounds are within the node's bounds on split axis.
		aabb_tri_min[split_axis] = fmaxf(((float*)&s_aabb_node_min)[split_axis], aabb_tri_min[split_axis]);
		aabb_tri_max[split_axis] = fminf(((float*)&s_aabb_node_max)[split_axis], aabb_tri_max[split_axis]);
	
		next_tri_list.d_aabb_tri_min[idx_tna] = make_float4(aabb_tri_min[0], aabb_tri_min[1], aabb_tri_min[2], 0.f);
		next_tri_list.d_aabb_tri_max[idx_tna] = make_float4(aabb_tri_max[0], aabb_tri_max[1], aabb_tri_max[2], 0.f);
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

extern "C"
void kernel_wrapper_do_split_clipping(const c_kd_node_list& active_list, 
									const c_kd_node_list& next_list, 
									const c_kd_chunk_list& chunks_list, 
									const c_triangle_data& tri_data)
{
	dim3 block_size = dim3(KD_CHUNKSIZE, 1, 1); 
	dim3 grid_size = CUDA_MAKEGRID2D(chunks_list.num_chunks, device_props.maxGridSize[0]);

	tri_vertex_data vtd;
	vtd.initialize(tri_data);

	kd_tri_node_list next_tri_list;
	next_tri_list.initialize(next_list);

	kernel_do_split_clipping<<<grid_size, block_size>>>(next_tri_list, chunks_list, vtd, active_list.d_split_axis, active_list.d_split_pos);
}