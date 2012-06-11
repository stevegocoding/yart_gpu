#include "app.h"

BEGIN_EVENT_TABLE(c_gl_canvas, wxGLCanvas)
	EVT_PAINT(c_gl_canvas::on_paint)
END_EVENT_TABLE()


c_gl_canvas::c_gl_canvas(wxFrame *parent)
: wxGLCanvas(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0, wxT("GLCanvas"))
{
}

void c_gl_canvas::on_paint(wxPaintEvent& event)
{
	render(); 
}

void c_gl_canvas::render()
{
	SetCurrent();
	wxPaintDC(this);
	
	glClearColor(0, 1.0f, 0, 0);
	
	glClear(GL_COLOR_BUFFER_BIT);

	glFlush();
	SwapBuffers();
}

//////////////////////////////////////////////////////////////////////////

IMPLEMENT_APP(c_wx_app)

bool c_wx_app::OnInit()
{
	wxFrame *frame = new wxFrame((wxFrame*)NULL, -1, wxT("Hello GL World"), wxPoint(50, 50), wxSize(200, 200));
	new c_gl_canvas(frame);
	
	frame->Show(true); 
	return true; 
}