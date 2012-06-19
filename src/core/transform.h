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
