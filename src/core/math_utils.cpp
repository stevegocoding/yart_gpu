#include "math_utils.h"

bool solve_linear_system2x2(const float A[2][2], 
	const float B[2], float *x0, float *x1) 
{
	float det = A[0][0]*A[1][1] - A[0][1]*A[1][0];
	if (fabsf(det) < 1e-10f)
		return false;
	*x0 = (A[1][1]*B[0] - A[0][1]*B[1]) / det;
	*x1 = (A[0][0]*B[1] - A[1][0]*B[0]) / det;
	if (_isnan(*x0) || _isnan(*x1))
		return false;
	return true;

}