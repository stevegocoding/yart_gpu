#include <cuda_runtime.h>

#include "main_frame.h"
#include "cuda_canvas.h"
#include "renderer.h"
#include "utils.h"

enum
{
	IDM_SHOWLOG 
};

BEGIN_EVENT_TABLE(c_main_frame, wxFrame)
	EVT_CLOSE(c_main_frame::on_close)
	EVT_MENU(IDM_SHOWLOG, c_main_frame::on_show_log)

END_EVENT_TABLE()

c_main_frame::c_main_frame(const wxString& title, const wxSize& size)
	: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, wxDefaultSize)
{
	// Create log window
	m_log_wnd = new wxLogWindow(this, L"Log", false, true); 
	wxWindow *txt_ctrl = m_log_wnd->GetFrame()->GetChildren().GetFirst()->GetData(); 
	txt_ctrl->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
	m_log_wnd->GetFrame()->SetSize(650, 500); 
	m_log_wnd->Show(); 
	
	// Create CUDA canvas
	if(!check_for_cuda())
	{
		yart_log_message("Failed to find compatible CUDA devices.");
		assert(false); 
	}
	m_cuda_canvas = new c_cuda_canvas(this, wxID_ANY, wxDefaultPosition, size);  
}

c_main_frame::~c_main_frame()
{
	
}

void c_main_frame::initialize_renderer()
{
	m_renderer.reset(new c_renderer(m_scene));
}

void c_main_frame::render(uchar4 *d_buf)
{
	wxSize screen_size = m_cuda_canvas->get_screen_size();
}

bool c_main_frame::need_update() const 
{
	return true;
}

bool c_main_frame::check_for_cuda()
{
	yart_log_message("Finding CUDA devices...");
	
	// Get CUDA device count.
	int num_devices = 0; 
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 0)
		yart_log_message("Found %d CUDA-enabled device(s)", num_devices);
	else 
	{
		yart_log_message("No CUDA-enabled devices found.");
		return false;
	}

	int gpu_arch_cores_per_sm[] = {-1, 8, 32}; 
	bool has_compatible = false;
	int id_to_use = 0;
	int best_power = 0; 

	// Enumerate the devices and select the most powerful one
	for (int d = 0; d < num_devices; ++d)
	{
		cudaDeviceProp props; 
		if (cudaGetDeviceProperties(&props, d) != cudaSuccess)
		{
			yart_log_message("%d: Failed to get device properties", d);
		}
		else 
		{
			int cores = props.multiProcessorCount;
			if (props.major <= 2)
				cores = gpu_arch_cores_per_sm[props.major]*props.multiProcessorCount; 

			yart_log_message("%d: %s (%d MPs, %d cores, %.0f MB global mem, %d.%d compute cap.)", 
				d, props.name, props.multiProcessorCount, cores, props.totalGlobalMem/(1024.f*1024.f),
				props.major, props.minor);
			
			bool is_compatible = (props.major >= 2 || props.minor >= 1);
			has_compatible |= is_compatible; 
			int compute_power = cores * props.clockRate; 
			if (is_compatible && compute_power > best_power)
			{
				id_to_use = d; 
				best_power = compute_power; 
			}
		}
	}

	if(has_compatible)
	{
		yart_log_message("-> Selecting CUDA device %d.", id_to_use);
		m_cuda_device_id = id_to_use;
		return true;
	}
	else
	{
		yart_log_message("Failed to detect compatible CUDA devices. Need compute cabability 1.1 or better.");
		return false;
	} 	
} 

void c_main_frame::on_close(wxCloseEvent& event)
{
	// m_cuda_canvas->destroy_gl_cuda();
	
	event.Skip(); 
}

void c_main_frame::on_show_log(wxCommandEvent& event)
{
	m_log_wnd->Show(); 
	m_log_wnd->GetFrame()->SetFocus(); 
}