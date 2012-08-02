#include "app.h"
#include "main_frame.h"
#include "utils.h"

#define APP_TITLE "YART GPU App"
#define SCREEN_W		1024
#define SCREEN_H		1024

IMPLEMENT_APP(c_wx_app) 

bool c_wx_app::OnInit()
{
	if (!wxApp::OnInit())
		return false;

	SetAppName(L"YART GPU");
	SetVendorName(L"Guangfu Shi");
	
	open_console_wnd();

	// Create main window.
	wxString strTitle = wxT(APP_TITLE);
	m_main_frame = new c_main_frame(strTitle, wxSize(SCREEN_W, SCREEN_H));
	m_main_frame->Show(true);
	m_main_frame->CenterOnScreen();
	SetTopWindow(m_main_frame);

	return true; 
}

int c_wx_app::OnRun()
{
	return wxApp::OnRun(); 
}

int c_wx_app::OnExit()
{
	return 0; 
}