#include "wx_utils.h"

void wx_log_fatal(const wxChar *str, ...)
{
	va_list args;
	va_start(args, str);
	//vfprintf(stderr, str, args);
	wxVLogFatalError(str, args);
	va_end(args); 
}

void wx_log_error(const wxChar *str, ...)
{
	va_list args;
	va_start(args, str);
	//vfprintf(stderr, str, args);
	//fprintf(stderr, "\n");
	wxVLogError(str, args);
	va_end(args);
}