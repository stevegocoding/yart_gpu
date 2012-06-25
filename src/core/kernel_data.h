#ifndef __kernel_data_h__
#define __kernel_data_h__

#pragma once 

#include <assert.h>
#include "cuda_defs.h"

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
struct c_shading_points_array 
{
#ifdef __cplusplus 
	
	void alloc_mem(uint32 _max_points);
	void destroy();
	
#endif

	// Number of shading points stored.
	uint32 num_pts;
	// Maximum number of shading points that can be stored.
	uint32 max_pts; 
	// Index of the corresponding pixel (device array).
	uint32 *d_pixels_buf; 
	// Index of the intersected triangle or -1, if no intersection (device array).
	int *d_tri_idices;
	// Point of intersection coordinates (device array).
	float4 *d_isect_pts;
	// Geometric normal at intersection point (device array).
	float4 *d_geo_normals; 
	// Shading normal at intersection point (device array).
	float4 *d_shading_normals; 
	// Barycentric hit coordinates (device array).
	float2 *d_isect_bary_coords; 
};


#endif // __kernel_data_h__
