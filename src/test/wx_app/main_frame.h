#ifndef __main_frame_h__
#define __main_frame_h__

#pragma once

#include <vector_types.h>
#include "wx/wx.h"

class c_cuda_canvas; 
class c_renderer;

class c_main_frame : public wxFrame
{

public:
	c_main_frame(const wxString& title, const wxSize& size);
	virtual ~c_main_frame();

	void render(uchar4 *d_buf); 
	bool need_update() const; 

	/// Returns the ID of the chosen CUDA device.
	int get_cuda_device_id() const { return m_cuda_device_id; }
	


	bool check_for_cuda(); 
	
private:

	void on_close(wxCloseEvent& event); 
	void on_show_log(wxCommandEvent& event); 
	
	c_renderer *m_renderer;
	c_cuda_canvas *m_cuda_canvas; 
	int m_cuda_device_id; 
	
	wxLogWindow *m_log_wnd;
	
	
	DECLARE_EVENT_TABLE()
	
}; 




#endif // __main_frame_h__
