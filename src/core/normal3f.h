#ifndef __normal3f_h__
#define __normal3f_h__

#pragma once 

#include <math.h>
#include <assert.h>
#include <vector3f.h>

class c_normal3f
{
public:
	/// Default constructor.
	c_normal3f(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	/// Explicit conversion from a vector. 
	explicit c_normal3f(const c_vector3f& v)	
		: x(v.x), y(v.y), z(v.z)
	{
	}
	~c_normal3f(void) {}

	// Data members
public:
	/// x-coordinate of normal.
	float x;
	/// y-coordinate of normal.
	float y;
	/// z-coordinate of normal.
	float z;

	// Operators
public:
	/// Normal addition operator.
	c_normal3f operator+(const c_normal3f& n) const
	{
		return c_normal3f(x + n.x, y + n.y, z + n.z);
	}
	/// Normal addition assignment operator.
	c_normal3f& operator+=(const c_normal3f& n)
	{
		x += n.x; y += n.y; z += n.z;
		return *this;
	}

	/// Normal subtraction operator.
	c_normal3f operator-(const c_normal3f& n) const
	{
		return c_normal3f(x - n.x, y - n.y, z - n.z);
	}
	/// Normal subtraction assignment operator.
	c_normal3f& operator-=(const c_normal3f& n)
	{
		x -= n.x; y -= n.y; z -= n.z;
		return *this;
	}

	/// Normal negation operator.
	c_normal3f operator-() const 
	{
		return c_normal3f(-x, -y, -z);
	}

	/// Normal scaling (by scalar) operator.
	c_normal3f operator*(float f) const
	{
		return c_normal3f(f*x, f*y, f*z);
	}
	/// Normal scaling (by scalar) assignment operator.
	c_normal3f& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Normal division (by scalar) operator.
	c_normal3f operator/(float f) const
	{
		assert(f != 0);
		float inv = 1.f / f;
		return c_normal3f(x*inv, y*inv, z*inv);
	}
	/// Normal division (by scalar) assignment operator.
	c_normal3f& operator/=(float f)
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
	/// Computes the squared length of the normal. Might be used to avoid the square root operation.
	float length_squared() const { return x*x + y*y + z*z; }

	/// Computes the length of this normal. Includes square root operation.
	float length() const { return sqrtf(length_squared()); }
};


inline c_normal3f operator*(float f, const c_normal3f& n)
{
	return n * f;
}

inline c_normal3f normalize(const c_normal3f& n)
{
	return n / n.length();
}

inline float dot(const c_normal3f& n1, const c_normal3f& n2)
{
	return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

inline float dot(const c_normal3f& n, const c_vector3f& v)
{
	return n.x * v.x + v.y * n.y + v.z * n.z;
}

inline float dot(const c_vector3f& v, const c_normal3f& n)
{
	return dot(n, v);
}

inline float abs_dot(const c_normal3f& n1, const c_normal3f& n2)
{
	return fabsf(dot(n1, n2));
}

inline float abs_dot(const c_normal3f& n, const c_vector3f& v)
{
	return fabsf(dot(n, v));
}

inline float abs_dot(const c_vector3f& v, const c_normal3f& n)
{
	return fabsf(dot(n, v));
}

typedef c_normal3f normal3f; 

#endif // __normal3f_h__
