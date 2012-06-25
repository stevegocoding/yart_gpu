#ifndef __app_h__
#define __app_h__

#pragma once 

#include "wx/wx.h" 



class c_main_frame; 
class c_wx_app : public wxApp
{


private:
	c_main_frame *m_main_frame; 
	
	virtual bool OnInit();
	virtual int OnExit(); 
	virtual int OnRun();
	
};




#endif // __app_h__
