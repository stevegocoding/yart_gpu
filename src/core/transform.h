#ifndef __TRANSFORM_H__
#define __TRANSFORM_H__

#pragma once

#include <math.h>
#include <ostream>
#include <assert.h>

#include "prerequisites.h"
#include "math_utils.h"
#include "vector3f.h"
#include "matrix44f.h"
#include "ray.h" 

class c_transform
{
public:
	c_transform() {}

	c_transform(const matrix44f& mat, const matrix44f& inv_mat)
		: m(mat)
		, inv_m(inv_mat)
	{}

	explicit c_transform(matrix44f& mat)
		: m(mat)
		, inv_m(inverse(mat))
	{}

	explicit c_transform(const float mat[4][4])
	{
		matrix44f temp = matrix44f(
			mat[0][0], mat[0][1], mat[0][2], mat[0][3],
			mat[1][0], mat[1][1], mat[1][2], mat[1][3],
			mat[2][0], mat[2][1], mat[2][2], mat[2][3],
			mat[3][0], mat[3][1], mat[3][2], mat[3][3]); 
		
		m = temp; 
		inv_m = temp.inverse();
	}
	
	// ---------------------------------------------------------------------
	/*
		Operators
	*/ 
	// ---------------------------------------------------------------------
	c_transform operator * (const c_transform& t2) const; 
	point3f operator() (const point3f& pt) const; 
	void operator() (const point3f& pt, point3f *out_pt) const; 
	vector3f operator() (const vector3f& v) const; 
	void operator() (const vector3f& v, vector3f *out_vec) const; 
	normal3f operator() (const normal3f& n) const; 
	void operator() (const normal3f& n, normal3f *out_n) const; 
	c_ray operator() (const c_ray& r) const; 
	void operator() (const c_ray& r, c_ray *out_r) const; 
	
	const matrix44f& get_matrix() const { return m; }
	const matrix44f& get_inv_matrix() const { return inv_m; }

	void print(std::ostream& os)
	{
		os << "Transform:" << std::endl;
	}
	
private:
	matrix44f m; 
	matrix44f inv_m; 
};


inline point3f c_transform::operator() (const point3f& pt) const
{
	float x = pt.x, y = pt.y, z = pt.z;

	// The homogeneous representation for points is [x, y, z, 1]^T.
	float xp = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
	float yp = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
	float zp = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
	float wp = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];

	assert(wp != 0);
	// Avoid division if possible.
	if(wp == 1.f)
		return point3f(xp, yp, zp);
	else
		return point3f(xp/wp, yp/wp, zp/wp);
	
}

inline void c_transform::operator() (const point3f& pt, point3f *out_pt) const 
{
	// Read out to allow inplace transformation.
	float x = pt.x, y = pt.y, z = pt.z;

	// The homogeneous representation for points is [x, y, z, 1]^T.
	out_pt->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z + m.m[0][3];
	out_pt->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z + m.m[1][3];
	out_pt->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z + m.m[2][3];
	float w    = m.m[3][0]*x + m.m[3][1]*y + m.m[3][2]*z + m.m[3][3];

	assert(w != 0);
	if(w != 1.f)
		*out_pt /= w;
} 

inline vector3f c_transform::operator() (const vector3f& v) const
{
	float x = v.x, y = v.y, z = v.z;

	// The homogeneous representation for vectors is [x, y, z, 0]^T. Therefore
	// there is no need to compute the w coordinate. This simplifies the
	// transform.
	return vector3f(m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z,
		m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z,
		m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z);
}

inline void c_transform::operator() (const vector3f& v, vector3f *out_vec) const
{
	// Read out to allow inplace transformation.
	float x = v.x, y = v.y, z = v.z;

	// The homogeneous representation for vectors is [x, y, z, 0]^T. Therefore
	// there is no need to compute the w coordinate. This simplifies the
	// transform.
	out_vec->x = m.m[0][0]*x + m.m[0][1]*y + m.m[0][2]*z;
	out_vec->y = m.m[1][0]*x + m.m[1][1]*y + m.m[1][2]*z;
	out_vec->z = m.m[2][0]*x + m.m[2][1]*y + m.m[2][2]*z;
}


inline normal3f c_transform::operator() (const normal3f& n) const 
{
	float x = n.x, y = n.y, z = n.z;

	// Note the swapped indices (for transpose).
	return normal3f(inv_m.m[0][0]*x + inv_m.m[1][0]*y + inv_m.m[2][0]*z,
		inv_m.m[0][1]*x + inv_m.m[1][1]*y + inv_m.m[2][1]*z,
		inv_m.m[0][2]*x + inv_m.m[1][2]*y + inv_m.m[2][2]*z); 
	
}

void c_transform::operator() (const normal3f& n, normal3f *out_n) const
{
	// Read out to allow inplace transformation.
	float x = n.x, y = n.y, z = n.z;

	// Note the swapped indices (for transpose).
	out_n->x = inv_m.m[0][0]*x + inv_m.m[1][0]*y + inv_m.m[2][0]*z;
	out_n->y = inv_m.m[0][1]*x + inv_m.m[1][1]*y + inv_m.m[2][1]*z;
	out_n->z = inv_m.m[0][2]*x + inv_m.m[1][2]*y + inv_m.m[2][2]*z;
}


inline c_ray c_transform::operator() (const c_ray& r) const 
{
	c_ray ret;
	(*this)(r.o, &ret.o);
	(*this)(r.d, &ret.d);
	return ret;
	
}

c_transform make_translate(const vector3f& trans);
c_transform make_scale(float sx, float sy, float sz);
c_transform inverse_transform(const c_transform& t); 
c_transform make_perspective_proj(float fov, float near, float far); 
c_transform make_look_at_lh(const point3f& pos, const point3f& look, const vector3f& up); 

/*
c_transform make_translate(const vector3f& trans);
c_transform make_scale(float sx, float sy, float sz);
c_transform make_rotate_x(float deg); 
c_transform make_rotate_y(float deg);
c_transform make_rotate_z(float deg); 
c_transform inverse_transform(const c_transform& t); 
c_transform make_perspective_proj(float fov, float near, float far); 
c_transform make_look_at_lh(const point3f& pos, const point3f& look, const vector3f& up); 
void build_coord_system(const vector3f& v1, PARAM_OUT vector3f* v2, PARAM_OUT vector3f *v3);
*/

#endif
