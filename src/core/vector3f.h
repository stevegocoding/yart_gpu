#ifndef __VECTOR3F_H__
#define __VECTOR3F_H__

#pragma once 

#include <math.h>
#include <assert.h>
#include <algorithm>

class c_normal3f;

class c_vector3f
{
public:
	/// Default constructor.
	c_vector3f(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	/// Explicit conversion from a normal.
	explicit c_vector3f(const c_normal3f& n);
	~c_vector3f(void) {}

	// Data members
public:
	/// x-coordinate of vector.
	float x;
	/// y-coordinate of vector.
	float y;
	/// z-coordinate of vector.
	float z;

	// Operators
public:
	/// Vector addition operator.
	c_vector3f operator+(const c_vector3f& v) const
	{
		return c_vector3f(x + v.x, y + v.y, z + v.z);
	}
	/// Vector addition assignment operator.
	c_vector3f& operator+=(const c_vector3f& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	/// Vector subtraction operator.
	c_vector3f operator-(const c_vector3f& v) const
	{
		return c_vector3f(x - v.x, y - v.y, z - v.z);
	}
	/// Vector subtraction assignment operator.
	c_vector3f& operator-=(const c_vector3f& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	/// Vector negation operator.
	c_vector3f operator-() const 
	{
		return c_vector3f(-x, -y, -z);
	}

	/// Vector scaling (by scalar) operator.
	c_vector3f operator*(float f) const
	{
		return c_vector3f(f*x, f*y, f*z);
	}
	/// Vector scaling (by scalar) assignment operator.
	c_vector3f& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Vector division (by scalar) operator.
	c_vector3f operator/(float f) const
	{
		assert(f != 0);
		float inv = 1.f / f;
		return c_vector3f(x*inv, y*inv, z*inv);
	}
	
	/// Vector division (by scalar) assignment operator.
	c_vector3f& operator/=(float f)
	{
		assert(f != 0);
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	float operator[](int i) const
	{
		assert(i >= 0 && i < 3);
		return (&x)[i];
	}

	float& operator[](int i)
	{
		assert(i >= 0 && i < 3);
		return (&x)[i];
	}

public:
	/// Computes the squared length of the vector. Might be used to avoid the square root operation.
	float length_squared() const { return x*x + y*y + z*z; }

	/// Computes the length of this vector. Includes square root operation.
	float length() const { return sqrtf(length_squared()); }
};


inline c_vector3f operator*(float f, const c_vector3f& v)
{
	return v * f;
}


inline c_vector3f normalize(const c_vector3f& v)
{
	return v / v.length();
}


inline float dot(const c_vector3f& v1, const c_vector3f& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline float abs_dot(const c_vector3f& v1, const c_vector3f& v2)
{
	return fabsf(dot(v1, v2));
}


inline c_vector3f cross(const c_vector3f& v1, const c_vector3f& v2)
{
	return c_vector3f((v1.y * v2.z) - (v1.z * v2.y),
		(v1.z * v2.x) - (v1.x * v2.z),
		(v1.x * v2.y) - (v1.y * v2.x));
}

inline void coordinate_system(const c_vector3f& v1, c_vector3f* pV2, c_vector3f* pV3)
{
	// Compute second vector by zeroing one component and swapping the others.
	if(fabsf(v1.x) > fabsf(v1.y))
	{
		float invLength = 1.f / sqrtf(v1.x * v1.x + v1.z * v1.z);
		*pV2 = c_vector3f(-v1.z * invLength, 0.f, v1.x * invLength);
	}
	else
	{
		float invLength = 1.f / sqrtf(v1.y * v1.y + v1.z * v1.z);
		*pV2 = c_vector3f(0.f, v1.z * invLength, -v1.y * invLength);
	}
	*pV3 = cross(v1, *pV2);
}

inline bool is_nan_vec(const c_vector3f& v)
{
	return ( _isnan((double)v[0]) || _isnan((double)v[1]) || _isnan((double)v[2]) ); 
}

typedef c_vector3f vector3f; 


#endif