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

inline __device__ bool d_ray_tri_intersect(const float3 a, const float3 b, const float3 c, 
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



#endif // __intersect_gpu_h__
