#pragma once

#include "cuda_defs.h"

class c_render_target
{
public:
	uint32 res_x() const { return m_resolution_x; }
	uint32 res_y() const { return m_resolution_y; }
	
protected:
	uint32 m_resolution_x; 
	uint32 m_resolution_y; 
};