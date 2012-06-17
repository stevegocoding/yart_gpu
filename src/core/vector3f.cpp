#include "vector3f.h"
#include "normal3f.h"

// Cannot move this into the header as we'd then have to include MNNormal3.h.
inline c_vector3f::c_vector3f(const c_normal3f& n)
	: x(n.x), y(n.y), z(n.z)
{
}