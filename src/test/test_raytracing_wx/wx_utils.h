#ifndef __wx_utils_h__
#define __wx_utils_h__

#pragma once 

#include <assert.h>
#include "wx/wx.h"

void wx_log_fatal(const wxChar *str, ...);
void wx_log_error(const wxChar *str, ...); 

#endif // __wx_utils_h__
