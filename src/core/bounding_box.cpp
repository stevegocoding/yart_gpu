#include "bounding_box.h"

c_aabb union_aabb(const c_aabb& box, const c_point3f& pt)
{
	c_aabb ret = box;
	ret.pt_min.x = std::min(box.pt_min.x, pt.x); 
	ret.pt_min.y = std::min(box.pt_min.y, pt.y); 
	ret.pt_min.z = std::min(box.pt_min.z, pt.z); 

	ret.pt_max.x = std::max(box.pt_max.x, pt.x); 
	ret.pt_max.y = std::max(box.pt_max.y, pt.y);
	ret.pt_max.z = std::max(box.pt_max.z, pt.z); 
	
	return ret; 
}


c_aabb union_aabb(const c_aabb& box1, const c_aabb& box2)
{
	c_aabb ret = box1; 
	ret.pt_min.x = std::min(box1.pt_min.x, box2.pt_min.x); 
	ret.pt_min.y = std::min(box1.pt_min.y, box2.pt_min.y); 
	ret.pt_min.z = std::min(box1.pt_min.z, box2.pt_min.z); 
	
	ret.pt_max.x = std::max(box1.pt_max.x, box2.pt_max.x); 
	ret.pt_max.y = std::max(box1.pt_max.y, box2.pt_max.y); 
	ret.pt_max.z = std::max(box1.pt_max.z, box2.pt_max.z); 

	return ret; 
}