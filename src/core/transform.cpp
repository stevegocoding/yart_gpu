#include "transform.h"

c_transform c_transform::operator*(const c_transform& t2) const
{
	matrix44f mNew = m * t2.m;
	// (AB)^-1 = B^-1 A^-1
	matrix44f mNewInv = t2.inv_m * inv_m;

	return c_transform(mNew, mNewInv);
}


c_transform make_translate(const vector3f& trans)
{
	// Modifies points only.
	matrix44f m(1, 0, 0, trans.x,
				0, 1, 0, trans.y,
				0, 0, 1, trans.z,
				0, 0, 0,       1);

	matrix44f inv_m(1, 0, 0, -trans.x,
					0, 1, 0, -trans.y,
					0, 0, 1, -trans.z,
					0, 0, 0,        1);


	return c_transform(m, inv_m);
}

c_transform make_scale(float sx, float sy, float sz)
{
	// Modifies both points and vectors.
	matrix44f m(sx, 0, 0, 0,
				0, sy, 0, 0,
				0, 0, sz, 0,
				0, 0, 0, 1);

	matrix44f inv_m(1.f/sx,     0,      0, 0,
					0,		1.f/sy,     0, 0,
					0,			0, 1.f/sz, 0,
					0,			0,     0, 1);
	
	return c_transform(m, inv_m); 
}

c_transform inverse_transform(const c_transform& t)
{
	return c_transform(t.get_inv_matrix(), t.get_matrix());
}

c_transform make_perspective_proj(float fov, float near, float far)
{ 
	matrix44f m 
		= matrix44f(
			1, 0, 0, 0,
			0, 1, 0, 0, 
			0, 0, far / (far - near), -far * near / (far - near),
			0, 0, 1, 0); 

	float inv_tan_angle = 1.0f / tanf(radians(fov)/2);
	return  make_scale(inv_tan_angle, inv_tan_angle, 1) * c_transform(m);
}

c_transform make_look_at_lh(const point3f& pos, const point3f& look, const vector3f& up)
{
	float m[4][4];
	
	// Initialize fourth column of viewing matrix
	m[0][3] = pos.x;
	m[1][3] = pos.y;
	m[2][3] = pos.z;
	m[3][3] = 1;

	// Initialize first three columns of viewing matrix
	vector3f dir = normalize(look - pos); 
	vector3f left = normalize(cross(normalize(up),dir)); 
	vector3f new_up = cross(dir, left); 

	m[0][0] = left.x; 
	m[1][0] = left.y; 
	m[2][0] = left.z; 
	m[3][0] = 0.0f; 
	
	m[0][1] = new_up.x;
	m[1][1] = new_up.y; 
	m[2][1] = new_up.z; 
	m[3][1] = 0.0f; 

	m[0][2] = dir.x; 
	m[1][2] = dir.y;
	m[2][2] = dir.z; 
	m[3][2] = 0.0f; 
	
	matrix44f cam_to_world(m);
	matrix44f world_to_cam = inverse(cam_to_world);
	
	return c_transform(world_to_cam, cam_to_world);
	
}

/*
void build_coord_system(const vector3f& v1, PARAM_OUT vector3f* v2, PARAM_OUT vector3f *v3)
{
	if (fabsf(v1[x]) > fabsf(v1[y]))
	{
		float inv_len = 1.0f / sqrtf(v1[x] * v1[x] + v1[z] * v1[z]); 
		*v2 = vector3f(-v1[z] * inv_len, 0.0f, v1[x] * inv_len);		
	}
	else 
	{
		float inv_len = 1.0f / sqrtf(v1[y] * v1[y] + v1[z] * v1[z]);
		*v2 = vector3f(0.0f, v1[z] * inv_len, -v1[y] * inv_len); 
	}
	
	*v3 = cross(v1, *v2);
}
*/