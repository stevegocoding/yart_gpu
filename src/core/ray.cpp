#include "ray.h"

bool c_ray::has_nan() const
{
    return ( is_nan_vec(o) || is_nan_vec(d) || _isnan(t_min) || _isnan(t_max) ); 
}