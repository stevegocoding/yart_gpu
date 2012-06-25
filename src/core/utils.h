#ifndef __utils_h__
#define __utils_h__

#pragma once

#include <vector_types.h>
#include <iostream>

void open_console_wnd();

void yart_log_message(const char *str, ...);


// ---------------------------------------------------------------------
/*
	Debug Utilities
*/ 
// ---------------------------------------------------------------------

void print_float3(std::ostream& os, float3& vec, int prec = 4, int width = 8);
void print_float4(std::ostream& os, float4& vec, int prec = 4, int width = 8);


#endif // __utils_h__
