#include "wx/config.h" 
#include "il/il.h" 
#include <cuda_runtime.h>

#include "main_frame.h"
#include "cuda_canvas.h"
#include "renderer.h"
#include "utils.h"
#include "wx_utils.h"
#include "assimp_loader.h" 
#include "scene.h"
#include "cuda_mem_pool.h"

enum
{
	IDM_LOADING = 1, 
	IDM_SAVEIMAGE, 
	IDM_BENCHMARK_RT, 
	IDM_SHOWLOG 
};

BEGIN_EVENT_TABLE(c_main_frame, wxFrame)
	EVT_CLOSE(c_main_frame::on_close)
	EVT_MENU(IDM_SHOWLOG, c_main_frame::on_show_log)
	EVT_MENU(IDM_SAVEIMAGE, c_main_frame::on_save_img)
	EVT_MENU(IDM_LOADING, c_main_frame::on_load_scene) 
	EVT_MENU(IDM_BENCHMARK_RT, c_main_frame::on_benchmark)
	
END_EVENT_TABLE()

c_main_frame::c_main_frame(const wxString& title, const wxSize& size)
	: wxFrame(NULL, wxID_ANY, title, wxDefaultPosition, size)
	, m_single_frame(true)
	, m_is_single_done(false)
{
	// Create log window
	m_log_wnd = new wxLogWindow(this, L"Log", false, true); 
	wxWindow *txt_ctrl = m_log_wnd->GetFrame()->GetChildren().GetFirst()->GetData(); 
	txt_ctrl->SetFont(wxFont(9, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL));
	m_log_wnd->GetFrame()->SetSize(650, 500); 
	m_log_wnd->Show(); 

	// Create file history
	m_file_history = new wxFileHistory(6);
	m_file_history->Load(*wxConfig::Get()); 

	// Create menu bar 
	create_menu_bar();

	// Create CUDA canvas
	if(!check_for_cuda())
	{
		yart_log_message("Failed to find compatible CUDA devices.");
		assert(false); 
	}

	m_cuda_canvas = new c_cuda_canvas(this, wxID_ANY, wxDefaultPosition, size);  

	m_is_ready = true; 
}

c_main_frame::~c_main_frame()
{
	
}

bool c_main_frame::reinit_renderer(bool update)
{
	if (!m_scene)
		return false; 
	
	m_renderer.reset(new c_renderer(m_scene));
	m_renderer->initialise(); 
	
	if (m_is_ready) 
	{
		m_is_single_done = false; 
		m_cuda_canvas->Refresh();
	}

	return true; 
}

void c_main_frame::render(uchar4 *d_buf)
{
	wxSize screen_size = m_cuda_canvas->get_screen_size();
	bool err = m_renderer->render_scene(d_buf);
	if (!err)
	{
		wx_log_error(L"Rendering Failed!"); 
		unload_scene();
	}

	m_is_single_done = true; 
}

bool c_main_frame::need_update() const 
{
	if (!m_is_ready)
		return false; 

	if (m_single_frame && !m_is_single_done)
		return m_renderer != NULL; 
	else if (!m_single_frame)
		return m_renderer != NULL; 
	else 
		return false;
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

	c_aabb bounds; 
	triangle_meshes2_array meshes;
	scene_material_array materials; 
	scene_light_array lights; 

	// Import the assimp scene 
	assimp_import_scene(std::string(file_name.mb_str()), &ai_scene);

	// Load the meshes
	assimp_load_meshes2(ai_scene, meshes, bounds);

	// Load material 
	assimp_load_materials(ai_scene, materials);
	
	// Release the assimp scene
	assimp_release_scene(ai_scene);
	
	// Create the lights 
	lights.push_back(make_point_light(c_vector3f(0.0f, 0.0f, 0.0f), c_vector3f(1.0f, 1.0f, 1.0f)));
	
	// Create camera 
	c_point3f eye_pos(0.0f, 0.0f, -5.0f);
	c_point3f look_at(0.f, 0.0f, 0.0f);
	c_vector3f up(0.0f, 1.0f, 0.0f); 
	float wnd[4] = {-1.333f, 1.333f, -1.0f, 1.0f}; 
	uint32 screen_w = m_cuda_canvas->get_screen_size().GetWidth(); 
	uint32 screen_h = m_cuda_canvas->get_screen_size().GetHeight(); 
	c_transform world_to_cam = make_look_at_lh(eye_pos, look_at, up);
	c_transform cam_to_world = inverse_transform(world_to_cam);
	c_transform proj = make_perspective_proj(60, 1e-2f, 1000.0f); 
	perspective_cam_ptr cam = make_perspective_cam(cam_to_world, wnd, 0, 0, 60, screen_w, screen_h);

	// Create the scene 
	m_scene.reset(new c_scene(meshes, bounds,cam, lights, materials));
	
	if (file_name.IsEmpty())
		m_file_history->AddFileToHistory(file_name); 
	
	return true; 
}

bool c_main_frame::unload_scene()
{
	if (!m_renderer)
		return false; 
	
	m_renderer.reset();
	m_scene.reset(); 
	
	return true; 
}

void c_main_frame::create_menu_bar()
{
	m_menu_recent = new wxMenu(); 
	m_file_history->UseMenu(m_menu_recent); 
	m_file_history->AddFilesToMenu(m_menu_recent); 

	wxMenu *menu_bench = new wxMenu();
	menu_bench->Append(IDM_BENCHMARK_RT, _("&Ray Tracing"));
	
	// File menu
	m_menu_file = new wxMenu(); 
	m_menu_file->Append(IDM_LOADING, _("&Load Scene"), _("Load a scene from file"));  
	m_menu_file->AppendSeparator();
	m_menu_file->Append(IDM_SAVEIMAGE, _("Save &Image"), _("Saves current image to file"));
	m_menu_file->AppendSeparator();
	m_menu_file->AppendSubMenu(m_menu_recent, _("&Recent Models")); 
	m_menu_file->Append(wxID_EXIT); 
	wxMenuBar *menu_bar = new wxMenuBar();
	menu_bar->Append(m_menu_file, _("&File")); 

	// Render menu
	m_menu_render = new wxMenu();
	m_menu_render->AppendSubMenu(menu_bench, _("&Benchmark")); 
	menu_bar->Append(m_menu_render, _("&Render")); 

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

void c_main_frame::on_save_img(wxCommandEvent& event)
{
	if (!m_cuda_canvas)
		return;
	
	wxString strWildcard;
	strWildcard += _("Portable Network Graphics (*.png)|*.png|");
	strWildcard += _("Windows Bitmap (*.bmp)|*.bmp|");
	strWildcard += _("Jpeg (*.jpg)|*.jpg|");
	strWildcard += _("All files (*.*)|*.*");

	// Show file dialog to get target file name.
	wxFileDialog* pFD = new wxFileDialog(this, _("Select destination for image"), wxEmptyString, wxEmptyString,
		strWildcard, wxFD_SAVE|wxFD_OVERWRITE_PROMPT);
	if(pFD->ShowModal() != wxID_OK)
		return;
	
	ILuint handle_img;

	// Generate the image and bind to handle 
	ilGenImages(1, &handle_img); 
	ilBindImage(handle_img); 

	// Set pixels from screen buffer 
	wxSize screen_size = m_cuda_canvas->get_screen_size(); 
	uchar4 *pixel_data = new uchar4[screen_size.GetWidth()*screen_size.GetHeight()]; 
	m_cuda_canvas->get_current_image(pixel_data); 
	ilTexImage(screen_size.GetWidth(), screen_size.GetHeight(), 1, 4, IL_RGBA, IL_UNSIGNED_BYTE, pixel_data); 
	
	SAFE_DELETE_ARRAY(pixel_data); 
	
	// Remove alpha channel.
	if (!ilConvertImage(IL_RGB, IL_UNSIGNED_BYTE))
	{
		wx_log_error(wxT("Failed to save image. Conversion failed."));
		ilDeleteImages(1, &handle_img); 
		return; 
	}
	
	// Save image to chosen file.
	//ilEnable(IL_FILE_OVERWRITE); 
	wchar_t *str_path = pFD->GetPath().wchar_str();
	std::wstring str(str_path); 
	if (!ilSaveImage(str.c_str()))
		wx_log_error(wxT("Failed to save image: %s."), pFD->GetPath().mb_str());

	ilDeleteImages(1, &handle_img);
}

void c_main_frame::on_benchmark(wxCommandEvent& event)
{
	if (event.GetId() == IDM_BENCHMARK_RT)
	{
		reinit_renderer();
	}
}