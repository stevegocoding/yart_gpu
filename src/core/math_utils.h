#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

#pragma once 

#include <math.h> 
#include <algorithm>
#include <float.h>

using std::swap;
using std::sqrt;


#define M_PI 3.14159265358979323846f
#define M_INV_PI 0.31830988618379067154f
#define M_INV_TWOPI 0.15915494309189533577f
#define M_INFINITY FLT_MAX

inline bool quadratic(float A, float B, float C, float *t0, float *t1)
{
    // Find quadratic discriminant
    float discrim = B * B - 4.f * A * C;
    if (discrim <= 0.) 
        return false;
    float rootDiscrim = sqrtf(discrim);

    // Compute quadratic _t_ values
    float q;
    if (B < 0) 
        q = -.5f * (B - rootDiscrim);
    else       
        q = -.5f * (B + rootDiscrim);
    *t0 = q / A;
    *t1 = C / q;
    if (*t0 > *t1) swap(*t0, *t1);
    return true;
    
}

/*
inline float lerp(float t, float v1, float v2)
{
	return (1.0f - t) * v1 + t * v2; 
}

inline int clamp(int val, int low, int high)
{
    if (val < low) return low; 
    else 
        if (val > high) 
            return high; 
        else
            return val;
}

inline float clamp(float val, float low, float high)
{
    if (val < low) return low; 
    else 
        if (val > high) 
            return high; 
        else
            return val;
}
*/

inline int floor2int(float val)
{
	return (int)floorf(val); 
}

inline int round2int(float val)
{
	return floor2int(val + 0.5f); 
}

inline float _log2(float x)
{
	static float inv_log2 = 1.f / logf(2.0f);
	return logf(x) * inv_log2; 
}

inline int log2int(float val)
{
	return floor2int(_log2(val));
}

inline float radians(float deg)
{
	return (M_PI / 180.f) * deg;
}

inline float degrees(float rad)
{
	return (180.f / M_PI) * rad;
}


bool solve_linear_system2x2(const float A[2][2], 
	const float B[2], float *x0, float *x1);


#endif