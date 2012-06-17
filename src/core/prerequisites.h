#ifndef __PREREQUISITES_H__
#define __PREREQUISITES_H__

#pragma once


//////////////////////////////////////////////////////////////////////////
// Class Forward Declarations and typedefs
//////////////////////////////////////////////////////////////////////////

// Geometry Classes
class c_ray;
class c_transform;
class c_camera; 
class c_render_target;

#pragma warning (disable : 4305) // double constant assigned to float
#pragma warning (disable : 4244) // int -> float conversion
#pragma warning (disable : 4267) // size_t -> unsigned int conversion

#define PARAM_OUT
#define PARAM_INOUT

#define SAFE_DELETE(p)  { if(p) delete (p); (p) = NULL; }
#define SAFE_DELETE_ARRAY(p)  { if(p) delete [] (p); (p) = NULL; }

#endif