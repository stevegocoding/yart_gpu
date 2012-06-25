#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <Windows.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "utils.h"

using namespace std;

HANDLE g_handle_out; 

void open_console_wnd()
{
	AllocConsole();

	g_handle_out = GetStdHandle(STD_OUTPUT_HANDLE);
	int hCrt = _open_osfhandle((long) g_handle_out, _O_TEXT);
	FILE* hf_out = _fdopen(hCrt, "w");
	setvbuf(hf_out, NULL, _IONBF, 1);
	*stdout = *hf_out;
}

void yart_log_message(const char *str, ...)
{
	std::stringstream ss;
	va_list args;
	va_start(args, str);
	
	vfprintf(stdout, str, args);
	fprintf(stdout, "\n");

	va_end(args);
}

//////////////////////////////////////////////////////////////////////////


void print_float3(ostream& os, float3& vec, int prec, int width)
{
	std::ios::fmtflags old_flags = os.flags(); 
	os.setf(ios::left, ios::adjustfield); 

	os  << std::setprecision(prec) << std::setw(width) << std::setfill(' ') 
		<< vec.x << ' ' 
		<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
		<< vec.y << ' ' 
		<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
		<< vec.z << ' ' << std::endl; 

	os.setf(old_flags); 
}

void print_float4(ostream& os, float4& vec, int prec, int width)
{
	std::ios::fmtflags old_flags = os.flags(); 
	os.setf(ios::left, ios::adjustfield); 

	os  << std::setprecision(prec) << std::setw(width) << std::setfill(' ') 
		<< vec.x << ' ' 
		<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
		<< vec.y << ' ' 
		<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
		<< vec.z << ' ' 
		<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
		<< vec.w << ' ' << std::endl; 

	os.setf(old_flags); 
}