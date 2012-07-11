#ifndef __functor_device_h__
#define __functor_device_h__

#include <vector_types.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include <assert.h>

// ---------------------------------------------------------------------
/*
	Functors 
*/ 
// ---------------------------------------------------------------------
template <typename T>
struct constant_add
{
	constant_add(T _val)
		: val(_val)
	{}

	__host__ __device__ 
	T operator() (T x) 
	{
		return x + val; 
	}

	T val;
};

template <typename T>
struct constant_sub
{
	constant_sub(T _val)
		: val(_val)
	{}

	__host__ __device__ 
	T operator() (T x) 
	{
		return x - val; 
	}

	T val;
};

template <typename T>
struct constant_mul
{
	constant_mul(T _val)
		: val(_val)
	{}

	__host__ __device__ 
	T operator() (T x) 
	{
		return x * val; 
	}

	T val;
};

struct is_valid
{
	__host__ __device__
		bool operator() (uint32 x)
	{
		return (x == 1);
	}
};

template <typename T>
struct op_minimum
{
	__host__ __device__ T operator()(T& a, T& b) {return a < b ? a : b;}
	__host__ __device__ T operator()(volatile T& a, volatile T& b) {return a < b ? a : b;}
};

template <>
struct op_minimum<float4>
{
	__host__ __device__
	float4 operator() (float4& a, float4& b)
	{
		return  make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
	} 
};

template <typename T> 
struct op_maximum
{
	__host__ __device__ T operator()(T& a, T& b) {return a > b ? a : b;}
	__host__ __device__ T operator()(volatile T& a, volatile T& b) {return a > b ? a : b;}
};

template <>
struct op_maximum<float4>
{
	__host__ __device__
	float4 operator() (float4& a, float4& b)
	{
		return  make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
	} 
};


#endif // __functor_device_h__
