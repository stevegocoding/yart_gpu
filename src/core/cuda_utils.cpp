#include "cuda_utils.h"


bool f_bEnableErrorChecks = false;

cudaError_t cuda_check_error(bool bforce /* = true */)
{
	if (bforce || f_bEnableErrorChecks)
		return cudaThreadSynchronize();
	else 
		return cudaSuccess; 
}