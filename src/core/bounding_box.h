#ifndef __bouding_box_h__
#define __bouding_box_h__

#pragma once

#include "math_utils.h"
#include "point3f.h"

class c_aabb 
{
public:
	
	c_aabb() 
	{
		pt_min = c_point3f(M_INFINITY, M_INFINITY, M_INFINITY); 
		pt_max = c_point3f(-M_INFINITY, -M_INFINITY, -M_INFINITY); 
	}

	explicit c_aabb(const c_point3f& p)
		: pt_min(p)
		, pt_max(p)
	{
	}

	c_aabb(const c_point3f& p1, const c_point3f& p2)
	{
		pt_min = c_point3f(std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z));
		pt_max = c_point3f(std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z)); 
	}
	
	~c_aabb() {}

	
public: 
	c_point3f pt_min; 
	c_point3f pt_max;
};

c_aabb union_aabb(const c_aabb& box, const c_point3f& pt);
c_aabb union_aabb(const c_aabb& box1, const c_aabb& box2);

#endif // __bouding_box_h__
