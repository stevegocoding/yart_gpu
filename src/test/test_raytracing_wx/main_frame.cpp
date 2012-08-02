#include "wx/config.h" 

#include <cuda_runtime.h>

#include "main_frame.h"
#include "cuda_canvas.h"
#include "renderer.h"
#include "utils.h"
#include "assimp_loader.h" 

enum
{
	IDM_LOADING = 1, 
	IDM_SHOWLOG 
};

BEGIN_EVENT_TABLE(c_main_frame, wxFrame)
	EVT_CLOSE(c_main_frame::on_close)
	EVT_MENU(IDM_SHOWLOG, c_main_frame::on_show_log)

	EVT_MENU(IDM_LOADING, c_main_frame::on_load_scene) 
	
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

	// Create menu bar 
	create_menu_bar();

	// Create file history
	m_file_history = new wxFileHistory(6);
	m_file_history->Load(*wxConfig::Get()); 
	
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

void c_main_frame::reinit_renderer()
{
	m_renderer.reset(new c_renderer(m_scene));
}

void c_main_frame::render(uchar4 *d_buf)
{
	wxSize screen_size = m_cuda_canvas->get_screen_size();
	// bool err = m_renderer->render_scene(d_buf); 
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

const aiScene *ai_scene = NULL; 
bool c_main_frame::load_scene_from_file(const wxString& file_name)
{
	unload_scene();

	assimp_import_scene(std::string(file_name.mb_str()), &ai_scene);
	
	if (file_name.IsEmpty())
		m_file_history->AddFileToHistory(file_name); 
	
	return true; 
}

bool c_main_frame::unload_scene()
{
	
	return true; 
}

void c_main_frame::create_menu_bar()
{
	// File menu
	m_menu_file = new wxMenu(); 
	m_menu_file->Append(IDM_LOADING, _("&Load Scene"), _("Load a scene from file"));  
	m_menu_file->Append(wxID_EXIT); 

	wxMenuBar *menu_bar = new wxMenuBar();
	menu_bar->Append(m_menu_file, _("&File")); 
	SetMenuBar(menu_bar); 
}

void c_main_frame::on_close(wxCloseEvent& event)
{
	m_cuda_canvas->destroy_gl_cuda();
	m_file_history->Save(*wxConfig::Get()); 
	SAFE_DELETE(m_file_history); 

	event.Skip(); 
}

void c_main_frame::on_show_log(wxCommandEvent& event)
{
	m_log_wnd->Show(); 
	m_log_wnd->GetFrame()->SetFocus(); 
}

void c_main_frame::on_load_scene(wxCommandEvent& event)
{
	wxString strWildcard;
	strWildcard += _("Model Files|*.obj;*.3ds;*.lwo;*.lws;*.ply;*.dae;*.xml;*.dxf;*.nff;*.smd;*.vta;");
	strWildcard += _("*.md1;*.md2;*.md3;*.md5mesh;*.x;*.raw;*.ac;*.irrmesh;*.irr;*.mdl;*.mesh.xml;*.ms3d|");
	strWildcard += _("Wavefront Object (*.obj)|*.obj|");
	strWildcard += _("3D Studio Max 3DS (*.3ds)|*.3ds|");
	strWildcard += _("LightWave (*.lwo,*.lws)|*.lwo;*.lws|");
	strWildcard += _("Stanford Polygon Library (*.ply)|*.ply|");
	strWildcard += _("Collada (*.dae,*.xml)|*.dae;*.xml|");
	strWildcard += _("AutoCAD DXF (*.dxf)|*.dxf|");
	strWildcard += _("Neutral File Format (*.nff)|*.nff|");
	strWildcard += _("Valve Model (*.smd,*.vta)|*.smd;*.vta|");
	strWildcard += _("Quake Model (*.md1,*.md2,*.md3)|*.md1;*.md2;*.md3|");
	strWildcard += _("Doom 3 (*.md5mesh)|*.md5mesh|");
	strWildcard += _("DirectX X (*.x)|*.x|");
	strWildcard += _("Raw Triangles (*.raw)|*.raw|");
	strWildcard += _("AC3D (*.ac)|*.ac|");
	strWildcard += _("Irrlicht (*.irrmesh,*.irr)|*.irrmesh;*.irr|");
	strWildcard += _("3D GameStudio Model (*.mdl)|*.mdl|");
	strWildcard += _("Ogre (*.mesh.xml)|*.mesh.xml|");
	strWildcard += _("Milkshape 3D (*.ms3d)|*.ms3d|");
	strWildcard += _("All files (*.*)|*.*");
	
	wxFileDialog* pFD = new wxFileDialog(this, _("Select a model to load!"), wxEmptyString, wxEmptyString,
										strWildcard, wxFD_OPEN);

	if(pFD->ShowModal() == wxID_OK)
	{
		if(load_scene_from_file(pFD->GetPath()))
			reinit_renderer(); 
	}
}