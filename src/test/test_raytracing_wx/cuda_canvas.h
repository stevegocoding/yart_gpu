#ifndef __cuda_canvas_h__
#define __cuda_canvas_h__

#pragma once

#include "wx/wx.h"
#include "wx/glcanvas.h"

class c_main_frame;
class c_cuda_canvas : public wxGLCanvas
{
	DECLARE_EVENT_TABLE() 
public:
	c_cuda_canvas(c_main_frame *parent, 
				wxWindowID id = wxID_ANY, 
				const wxPoint& pos = wxDefaultPosition, 
				const wxSize& size = wxDefaultSize);
	
	virtual ~c_cuda_canvas();
	

	void init_gl_cuda(); 
	void destroy_gl_cuda(); 
	void render();

	wxSize get_screen_size() const { return m_screen_size; }
	void get_current_image(uchar4 *pixel_data);

private:

	// ---------------------------------------------------------------------
	/* Event Handlers 
	*/ 
	// ---------------------------------------------------------------------
	void OnPaint(wxPaintEvent& event);
	void OnEraseBackground(wxEraseEvent& event); 

	void set_enable_vsync(bool is_enable);
	
	c_main_frame *m_main_frame; 
	wxSize m_screen_size; 
	bool m_is_inited;
	
	GLuint m_gl_vbo; 
	GLuint m_gl_tex;
	struct cudaGraphicsResource *m_cuda_vbo_res;

	
	
	


};


#endif // __cuda_canvas_h__
