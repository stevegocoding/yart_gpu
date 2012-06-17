#ifndef __point3f_h__
#define __point3f_h__

#include <math.h>
#include <assert.h>
#include "vector3f.h"
#include "normal3f.h"

class c_point3f
{
public:
	/// Default constructor.
	c_point3f(float _x=0, float _y=0, float _z=0)
		: x(_x), y(_y), z(_z)
	{
	}
	~c_point3f(void) {}

	// Data members
public:
	/// x-coordinate of point.
	float x;
	/// y-coordinate of point.
	float y;
	/// z-coordinate of point.
	float z;

	// Operators
public:
	/// Point addition operator.
	c_point3f operator+(const c_vector3f& v) const
	{
		return c_point3f(x + v.x, y + v.y, z + v.z);
	}
	/// Point addition assignment operator.
	c_point3f& operator+=(const c_vector3f& v)
	{
		x += v.x; y += v.y; z += v.z;
		return *this;
	}

	/// Point subtraction operator.
	c_point3f operator-(const c_vector3f& v) const
	{
		return c_point3f(x - v.x, y - v.y, z - v.z);
	}
	/// Point subtraction assignment operator.
	c_point3f& operator-=(const c_vector3f& v)
	{
		x -= v.x; y -= v.y; z -= v.z;
		return *this;
	}

	/// Point scaling (by scalar) operator.
	c_point3f operator*(float f) const
	{
		return c_point3f(f*x, f*y, f*z);
	}
	/// Point scaling (by scalar) assignment operator.
	c_point3f& operator*=(float f)
	{
		x *= f; y *= f; z *= f;
		return *this;
	}

	/// Point division (by scalar) operator.
	c_point3f operator/(float f) const
	{
		assert(f != 0);
		float inv = 1.f / f;
		return c_point3f(x*inv, y*inv, z*inv);
	}
	/// Point division (by scalar) assignment operator.
	c_point3f& operator/=(float f)
	{
		assert(f != 0);
		float inv = 1.f / f;
		x *= inv; y *= inv; z *= inv;
		return *this;
	}

	/// Vector generation operator. Creates an c_vector3f by subtracting the
	/// given point \a p from this point and returning the result.
	c_vector3f operator-(const c_point3f& p) const
	{
		return c_vector3f(x - p.x, y - p.y, z - p.z);
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
};

inline float distance_squared(const c_point3f& p1, const c_point3f& p2)
{
	return (p2 - p1).length_squared();
}

inline float distance(const c_point3f& p1, const c_point3f& p2)
{
	return (p2 - p1).length();
}

inline float dot(const c_point3f& p, const c_normal3f& n)
{
	return p.x * n.x + p.y * n.y + p.z * n.z;
}

inline float dot(const c_normal3f& n, const c_point3f& p)
{
	return dot(p, n);
}

typedef c_point3f point3f;


#endif // __point3f_h__
