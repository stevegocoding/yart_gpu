#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

// ---------------------------------------------------------------------
/*
It is important that this is 32-bit wide as some operations, e.g. CUDPP primitives, do
not support wider types.
*/ 
// ---------------------------------------------------------------------
typedef unsigned __int32 uint32;

/// Unsigned char type.
typedef unsigned char uchar; 