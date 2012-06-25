#include <GL/glew.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>

#include "cuda_canvas.h"
#include "main_frame.h"
#include "utils.h"

// Device context attribute list for wxGLCanvas
// "Note that all the attributes need to be set in the attribList. Just setting the 
//  interesting values of GL_SAMPLE_BUFFERS do not work causing glewInit() to fail."
// See: http://www.comp.nus.edu.sg/~ashwinna/docs/wxWidgetsInstallation.html
int f_devAttribs[] = {WX_GL_RGBA, 
	WX_GL_DOUBLEBUFFER, 
	WX_GL_DEPTH_SIZE, 0,
	0, 0};


c_cuda_canvas::c_cuda_canvas(c_main_frame *parent, 
							wxWindowID id /* = wxID_ANY */, 
							const wxPoint& pos /* = wxDefaultPosition */, 
							const wxSize& size /* = wxDefaultSize */)
							: wxGLCanvas(parent, (wxGLCanvas*)NULL, id, pos, size, wxFULL_REPAINT_ON_RESIZE, _T("CUDA GLCanvas"), f_devAttribs)
							, m_is_inited(false)
{
	// SetBackgroundColour(wxColour(0)); 
}

c_cuda_canvas::~c_cuda_canvas()
{
	destroy_gl_cuda();
}

void c_cuda_canvas::init_gl_cuda()
{
	cudaError_t err; 
	
	wxGLContext *gl_context = GetContext(); 
	assert(gl_context);

	SetCurrent();
	
	// Required for VBO stuff.
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0"))
		assert(false);
	
	// Disable VSync. Works only on Windows.
	set_enable_vsync(false);
	
	// Note that we do not call cudaSetDevice() since we call cudaGLSetGLDevice.
	int device_id = m_main_frame->get_cuda_device_id();
	err = cudaGLSetGLDevice(device_id);
	assert(err == cudaSuccess); 
	
	// Create video buffer object.
	// NOTE: For CUDA toolkit 2.3 I used a pixel buffer object here. This is no more required for toolkit 3.0. 
	glGenBuffers(1, &m_gl_vbo); 
	glBindBuffer(GL_ARRAY_BUFFER, m_gl_vbo); 
	glBufferData(GL_ARRAY_BUFFER, m_screen_size.x * m_screen_size.y * sizeof(GLubyte)*4, 0, GL_DYNAMIC_DRAW); 
	glBindBuffer(GL_ARRAY_BUFFER, 0); 
	
	// Register this buffer object with CUDA
	err = cudaGraphicsGLRegisterBuffer(&m_cuda_vbo_res, m_gl_vbo, cudaGraphicsMapFlagsNone); 
	assert(err == cudaSuccess);
	
	// Render target texture
	glGenTextures(1, &m_gl_tex);
	glBindTexture(GL_TEXTURE_2D, m_gl_tex); 
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_screen_size.x, m_screen_size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D,  GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
	glBindTexture(GL_TEXTURE_2D, 0); 
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glMatrixMode(GL_PROJECTION); 
	glLoadIdentity();
	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
	
	yart_log_message("OpenGL and CUDA initialized. Using device %d.", device_id);
 
	m_is_inited = false; 
}

void c_cuda_canvas::destroy_gl_cuda()
{
	cudaError_t err; 

	if (!m_is_inited)
		return; 
	
	err = cudaGraphicsUnregisterResource(m_cuda_vbo_res);
	
	glDeleteBuffers(1, &m_gl_vbo); 
	glDeleteTextures(1, &m_gl_tex);
	
	m_gl_vbo = 0; 
	m_gl_tex = 0; 	
}

void c_cuda_canvas::render()
{
	cudaError_t err = cudaSuccess; 

	if (!GetContext()) 
		yart_log_message("Failed to get OpenGL context!");
	
	wxPaintDC dc(this); 
	SetCurrent();

	glClear(GL_COLOR_BUFFER_BIT); 

	if (!m_main_frame->need_update())
	{
		glBindTexture(GL_TEXTURE_2D, m_gl_tex); 
	}
	else 
	{
		// Map VBO to get cuda device memory pointer
		size_t num_bytes; 
		uchar4 *d_buf_ptr = NULL;
		err = cudaGraphicsMapResources(1, &m_cuda_vbo_res, 0);
		assert(err == cudaSuccess); 
		err = cudaGraphicsResourceGetMappedPointer((void**)&d_buf_ptr, &num_bytes, m_cuda_vbo_res); 
		assert(err == cudaSuccess); 
		m_main_frame->render(d_buf_ptr);
		err = cudaGraphicsUnmapResources(1, &m_cuda_vbo_res, 0); 
		assert(err == cudaSuccess); 
		
		// Transfer VBO data to texture object 
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_gl_vbo);
		glBindTexture(GL_TEXTURE_2D, m_gl_tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_screen_size.x, m_screen_size.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	// Draw a full-screen quad with the texture
	glEnable(GL_TEXTURE_2D);
	

	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.0f, 0.0f); 
	glTexCoord2f(0, 0);	glVertex2f(0, 0);
	glColor3f(0.0f, 1.0f, 0.0f); 
	glTexCoord2f(1, 0);	glVertex2f(1, 0);
	glColor3f(0.0f, 0.0f, 1.0f); 
	glTexCoord2f(1, 1);	glVertex2f(1, 1);
	glColor3f(0.5f, 0.5f, 0.5f); 
	glTexCoord2f(0, 1);	glVertex2f(0, 1);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, 0);
	
	glFlush();
	SwapBuffers();
}


#ifdef _WIN32
// See http://www.devmaster.net/forums/showthread.php?t=443
typedef BOOL (APIENTRY *PFNWGLSWAPINTERVALFARPROC)(int);
PFNWGLSWAPINTERVALFARPROC wglSwapIntervalEXT = 0;

void c_cuda_canvas::set_enable_vsync(bool is_enable)
{
	const char *extensions = (const char*)glGetString( GL_EXTENSIONS );

	if(strstr( extensions, "WGL_EXT_swap_control" ) == 0)
	{
		yart_log_message("Disabling vertical synchronization not supported.");
		return;
	}
	else
	{
		wglSwapIntervalEXT = (PFNWGLSWAPINTERVALFARPROC)wglGetProcAddress( "wglSwapIntervalEXT" );

		int swapInterval = (is_enable ? 1 : 0);
		if(wglSwapIntervalEXT)
			wglSwapIntervalEXT(swapInterval);
	}
}
#else // _WIN32
void CUDACanvas::SetEnableVSync(bool bEnable)
{
	MNWarning("Disabling vertical synchronization not supported.");
}
#endif // _WIN32 