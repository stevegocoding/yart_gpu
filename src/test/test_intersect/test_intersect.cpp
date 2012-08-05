#include <fstream>

#include "obj_makers.h"
#include "cuda_mem_pool.h"
#include "camera.h"
#include "ray_tracer.h"
#include "utils.h"
#include "cuda_rng.h"
#include "kernel_data.h"
#include "scene.h"
#include "assimp_loader.h"
#include "renderer.h"

class c_renderer;
typedef boost::shared_ptr<c_renderer> renderer_ptr; 

uint32 g_screen_width = 512; 
uint32 g_screen_height = 512; 

boost::shared_array<float4> h_ray_origins;
boost::shared_array<float4> h_ray_dirs;

const std::string file_name = "../data/models/MNSimple.obj";
static const std::string mt_dat_file = "../data/mt/MersenneTwister.dat";
const aiScene *ai_scene = NULL;
scene_ptr scene;
renderer_ptr renderer; 

void initialise()
{
	open_console_wnd();
	
	std::ofstream ofs("debug.txt");

	// Init CUDA MT
	srand(1337); 
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	bool ret = rng.init(mt_dat_file); 
	assert(ret); 
	ret = rng.seed(1337);
	assert(ret); 

	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	mem_pool.initialise(256*1024*1024, 256*1024);

	triangle_meshes2_array meshes; 
	scene_material_array materials; 
	scene_light_array lights; 

	c_aabb bounds; 
	assimp_import_scene(file_name, &ai_scene);
	assimp_load_meshes2(ai_scene, meshes, bounds);
	assimp_load_materials(ai_scene, materials);
	assimp_release_scene(ai_scene);
	lights.push_back(make_point_light(c_vector3f(0.0f, 0.0f, 0.0f), c_vector3f(1.0f, 1.0f, 1.0f)));

	// Create camera 
	c_point3f eye_pos(0.0f, 0.0f, -0.5f);
	c_point3f look_at(0.f, 0.0f, 0.0f);
	c_vector3f up(0.0f, 1.0f, 0.0f); 
	float wnd[4] = {-1.333f, 1.333f, -1.0f, 1.0f}; 
	c_transform world_to_cam = make_look_at_lh(eye_pos, look_at, up);
	c_transform cam_to_world = inverse_transform(world_to_cam);
	c_transform proj = make_perspective_proj(60, 1e-2f, 1000.0f); 
	perspective_cam_ptr cam = make_perspective_cam(cam_to_world, wnd, 0, 0, 60, g_screen_width, g_screen_height);

	scene.reset(new c_scene(meshes, bounds,cam, lights, materials));

	renderer.reset(new c_renderer(scene)); 
	renderer->initialise(); 
}

void print_ray_pool(ray_pool_ptr ray_pool)
{
	std::ofstream ofs("debug_output.txt");
	
	for (size_t i = 0; i < ray_pool->get_num_chuks(); ++i)
	{
		c_ray_chunk *chunk = ray_pool->get_chunk(i); 
		
		size_t n = chunk->num_rays; 

		// Copy memory from device to host 
		h_ray_origins.reset(new float4[n]);
		h_ray_dirs.reset(new float4[n]);
		cudaMemcpy(h_ray_origins.get(), chunk->d_origins, sizeof(float4)*n, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ray_dirs.get(), chunk->d_dirs, sizeof(float4)*n, cudaMemcpyDeviceToHost);
		
		for (size_t j = 0; j < n; ++j)
		{
			print_float4(ofs, h_ray_dirs[j]);
		}

		h_ray_origins.reset();
		h_ray_dirs.reset();
	}
	
	ofs.close();
}

int main(int argc, char **argv)
{
	initialise();

	c_cuda_memory<uchar4> d_buf(g_screen_width*g_screen_width); 
	renderer->render_scene(d_buf.buf_ptr());
	
	// Print rays
	// print_ray_pool(g_ray_pool);

	renderer.reset(); 

	return 0; 
}