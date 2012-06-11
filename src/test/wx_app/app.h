#pragma once 

#include "wx/wx.h"
#include "wx/glcanvas.h"

class c_gl_canvas : public wxGLCanvas
{
public:
	c_gl_canvas(wxFrame *parent);
	void on_paint(wxPaintEvent& event); 


protected: 
	DECLARE_EVENT_TABLE();
	void render();
	
};

//////////////////////////////////////////////////////////////////////////

class c_wx_app : public wxApp
{
public:
	virtual bool OnInit();		
	
};

