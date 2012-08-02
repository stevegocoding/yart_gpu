#ifndef __kernel_data_h__
#define __kernel_data_h__

#pragma once 

#include <assert.h>
#include "cuda_defs.h"


/// Maximum number of materials allowed. The number is restricted to ensure constant structure size for
/// GPU's constant memory.
#define MAX_MATERIALS			64
/// Number of texture types. It is restricted by material flag array. See MaterialProperties.
#define NUM_TEX_TYPES			2

// ---------------------------------------------------------------------
/*
	Triangle data structure for kernel use.
	Holds triangle data from a BasicScene in form of a structure of arrays for
	coalescing. \c float4 is used instead of \c float3 to ensure alignment and improve
	performance.

	/// \note	It seems that the texture fetcher can only fetch from a texture once per thread
	/// 		without using too many registers (observed on a GTS 250, CUDA 2.3). Also there seems
	/// 		to be a limit of fetches that can be done without using too many registers and having
	/// 		the performance dropping. Also note that we use linear device memory instead of cuda
	/// 		arrays since cuda arrays can only hold up to 8k elements when one dimensional, while
	/// 		linear device memory can have up to 2^27 elements (observed on a GTS 250).
*/ 
// ---------------------------------------------------------------------

class c_scene;
struct c_triangle_data
{
#ifdef __cplusplus
	
	/*
	c_triangle_data()
	{
		for (uint32 i = 0; i < 3; ++i)
		{
			d_verts[i] = NULL; 
			//d_normals[i] = NULL; 
		}
		//d_material_idx = NULL;
		//d_texcoords[0] = d_texcoords[1] = NULL;
	}
	*/ 
	
#endif 

	uint32 num_tris; 
	float3 aabb_min; 
	float3 aabb_max; 
	float4 *d_verts[3];				// In the layout of (x, x, x ...), (y, y, y ...), (z, z, z ...)
	float4 *d_normals[3];			// In the layout of (nx, nx, nx ...), (ny, ny, ny ...), (nz, nz, nz ...)
	//float2 *d_texcoords[3];			// In the layout of (u, u, u ...), (v, v, v ...)
	uint32 *d_material_idx; 
}; 

// ---------------------------------------------------------------------
/*
	Structure to hold the shading points determined during a ray tracing pass.

	The ray \em tracing kernel fills this structure with intersection information that
	describes the found hits for rays traced. Not all rays have to hit something. Therefore
	the #d_idxTris contains -1 for rays that hit nothing.

	All data is stored in global GPU memory since we generate the rays on the GPU. A
	structure of arrays (SoA) is used instead of an array of structures (AoS) to allow
	coalesced memory access. Each thread handles a single ray. The global thread index
	tid is the index of the shading point to write. Hence the shading point arrays
	have to be at least as large as the RayChunk that is traced.
*/ 
// ---------------------------------------------------------------------
struct c_shading_points 
{
#ifdef __cplusplus 
	
	void initialise(uint32 _max_points);
	void destroy();

	// ---------------------------------------------------------------------
	/*
	/// \brief	Compacts this shading point structure using the given source address array.
	///
	///			This operation assumes that the source addresses were generated before, e.g. using
	///			::mncudaGenCompactAddresses(). The latter also returns the required new number of
	///			shading points. Basically, this was done to allow compacting multiple structures
	///			using the same source addresses.
	*/ 
	// ---------------------------------------------------------------------
	void compact_src_addr(uint32 *d_src_addr, uint32 new_count);
	
#endif

	// Number of shading points stored.
	uint32 num_pts;
	// Maximum number of shading points that can be stored.
	uint32 max_pts; 
	// Index of the corresponding pixel (device array).
	uint32 *d_pixels; 
	// Index of the intersected triangle or -1, if no intersection (device array).
	int *d_tri_indices;
	// Point of intersection coordinates (device array).
	float4 *d_isect_pts;
	// Geometric normal at intersection point (device array).
	float4 *d_geo_normals; 
	// Shading normal at intersection point (device array).
	float4 *d_shading_normals; 
	// Barycentric hit coordinates (device array).
	float2 *d_isect_barycoords; 
};


//////////////////////////////////////////////////////////////////////////

enum e_light_type 
{
	light_type_point = 0,
	light_type_directional = 1, 
	light_type_area_disc = 2,
	light_type_area_quad = 3
};

struct c_light_data
{
	/// Type of light source.
	e_light_type type;
	/// Light source position. Invalid for directional light sources.
	float3 position;
	/// Light direction. Invalid for point light sources.
	float3 direction;
	/// Emitted radiance of light source. This is the intensity for point lights.
	float3 emit_l;
	/// Vector that spans first side of the rectangle for rectangle area lights.
	float3 area_v1;
	/// Vector that spans second side of the rectangle for rectangle area lights.
	float3 area_v2;
	/// Area light radius for disc area lights.
	float area_radius;
};


struct c_material_desc
{
	float3 diff_color[MAX_MATERIALS]; 
	float3 spec_color[MAX_MATERIALS]; 
	float spec_exp[MAX_MATERIALS]; 
};

struct c_material_data 
{
public: 
	void initialize(c_scene *scene); 
	void destroy();
	
	uint32 num_materials; 
	c_material_desc mats_desc; 
};


#endif // __kernel_data_h__
