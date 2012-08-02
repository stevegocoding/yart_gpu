#ifndef __intersect_gpu_h__
#define __intersect_gpu_h__

// ---------------------------------------------------------------------
/*
/// \param	a					First triangle vertex. 
/// \param	b					Second triangle vertex. 
/// \param	c					Third triangle vertex. 
/// \param	o					Ray origin. 
/// \param	d					Ray direction. 
/// \param [out]	out_lambda	Intersection parameter lambda. 
/// \param [out]	out_bary1	Barycentric hit coordinate 1. 
/// \param [out]	out_bary2	Barycentric hit coordinate 2. 
*/ 
// ---------------------------------------------------------------------

inline __device__ bool device_ray_tri_intersect(const float3 a, const float3 b, const float3 c, 
											const float3 o, const float3 d, 
											float& out_lambda, float& out_bary1, float& out_bary2)
{
	float3 e1 = b - a; 
	float3 e2 = c - a; 

	float3 pv = cross(d, e2); 
	float det = dot(e1, pv);

	if (det == 0.0f)
	{
		return false;
	}

	float inv_det = 1.0f / det;

	float3 tv = o - a; 
	out_bary1 = dot(tv, pv) * inv_det;
	
	float3 qv = cross(tv, e1);
	out_bary2 = dot(d, qv) * inv_det;
	out_lambda = dot(e2, qv) * inv_det;

	bool is_hit = (out_bary1 >= 0.0f && out_bary2 >= 0.0f && (out_bary1 + out_bary2) <= 1.0f);
	return is_hit;
}

inline __device__ bool device_ray_box_intersect(const float3 aabb_min, 
												const float3 aabb_max, 
												const float3 ray_origin,
												const float3 inv_ray_dir, 
												const float t_min, 
												const float t_max, 
												float& t_min_isect,
												float& t_max_isect)
{
	float t0 = t_min;
	float t1 = t_max; 

	float *origin = (float*)&ray_origin;
	float *pt_min = (float*)&aabb_min; 
	float *pt_max = (float*)&aabb_max; 
	
	bool isect = true; 
	
#pragma unroll 
	for (uint32 i = 0; i < 3; ++i)
	{
		// Update interval for ith bounding box slab.
		float val1 = (pt_min[i] - origin[i]) * ((float*)&inv_ray_dir)[i];
		float val2 = (pt_max[i] - origin[i]) * ((float*)&inv_ray_dir)[i];
		
		// Update parametric interval from slab intersection.
		float t_near = val1; 
		float t_far = val2; 

		if (val1 > val2)
			t_near = val2; 
		if (val1 > val2)
			t_far = val1; 
		t0 = ((t_near > t0) ? t_near : t0);
		t1 = ((t_far < t1) ? t_far : t1); 
		
		// DO NOT break or return here to avoid divergent branches.
		if (t0 > t1)
			isect = false; 
	}

	t_min_isect = t0; 
	t_max_isect = t1; 

	return isect;
}



#endif // __intersect_gpu_h__
