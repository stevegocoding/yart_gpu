#include <iostream>
#include <iomanip>
#include <sstream>

#include "cuda_utils.h"

using namespace std;

bool f_bEnableErrorChecks = false;

cudaError_t cuda_check_error(bool bforce /* = true */)
{
	if (bforce || f_bEnableErrorChecks)
		return cudaThreadSynchronize();
	else 
		return cudaSuccess; 
}


//////////////////////////////////////////////////////////////////////////
