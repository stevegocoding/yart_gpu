#pragma once

#include "prerequisites.h"
#include "math_utils.h"
#include "vector3f.h"
#include "point3f.h"

class c_ray
{
public:
	c_ray(const vector3f& _o = vector3f(), const vector3f& _d = vector3f(), 
		  float start = 0.0f, float end = std::numeric_limits<float>::infinity(), 
		  float _t = 0.0f, 
		  int _depth = 0,
		  float time = 0.0f)
		: o(_o)
		, d(_d)
		, t_min(start) 
		, t_max(end)
	{}

	bool has_nan() const;

    vector3f evaluate_t(const float t) const 
    {
        return o + d * t;
    }

	vector3f o;
	vector3f d;
	mutable float t_min, t_max;
};

class c_ray_differential : public c_ray
{
public:
	c_ray_differential() {} 

	point3f rx_o, ry_o; 
	vector3f rx_d, ry_d;
};