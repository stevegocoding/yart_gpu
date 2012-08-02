#ifndef __main_frame_h__
#define __main_frame_h__

#pragma once

#include <boost/shared_ptr.hpp>
#include <vector_types.h>
#include "wx/wx.h"
#include "wx/docview.h"			// For wxFileHistory 

class c_cuda_canvas; 
class c_renderer;
typedef boost::shared_ptr<c_renderer> renderer_ptr; 

class c_scene; 
typedef boost::shared_ptr<c_scene> scene_ptr; 

class wxFileHistory; 


class c_main_frame : public wxFrame
{

public:
	c_main_frame(const wxString& title, const wxSize& size);
	virtual ~c_main_frame();

	void reinit_renderer();
	void render(uchar4 *d_buf); 
	bool need_update() const; 

	/// Returns the ID of the chosen CUDA device.
	int get_cuda_device_id() const { return m_cuda_device_id; }
	bool check_for_cuda(); 
	void create_menu_bar(); 
	
private:
	
	void on_close(wxCloseEvent& event); 
	void on_show_log(wxCommandEvent& event); 
	void on_load_scene(wxCommandEvent& event); 
	void on_save_img(wxCommandEvent& event);

	bool load_scene_from_file(const wxString& file_name); 
	bool unload_scene();
	
	// ---------------------------------------------------------------------
	/* Renderer Objects 
	*/ 
	// ---------------------------------------------------------------------
	scene_ptr m_scene; 
	renderer_ptr m_renderer;
	c_cuda_canvas *m_cuda_canvas; 
	int m_cuda_device_id; 

	// ---------------------------------------------------------------------
	/* Stats
	*/ 
	// ---------------------------------------------------------------------
	bool m_single_frame; 

	// ---------------------------------------------------------------------
	/* wxWidgets Objects
	*/ 
	// ---------------------------------------------------------------------
	wxLogWindow *m_log_wnd;
	wxFileHistory *m_file_history; 
	wxMenu *m_menu_file;
	wxMenu *m_menu_recent; 
	
	DECLARE_EVENT_TABLE()
	
}; 




#endif // __main_frame_h__
