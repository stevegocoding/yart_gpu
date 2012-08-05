#include <iostream>
#include <fstream>
#include <string>

#include "obj_makers.h"
#include "cuda_mem_pool.h"
#include "camera.h"
#include "ray_tracer.h"
#include "cuda_utils.h"
#include "utils.h"
#include "cuda_rng.h"
#include "kernel_data.h"
#include "kdtree_kernel_data.h"
#include "kdtree_triangle.h"
#include "scene.h"
#include "assimp_loader.h"

static const std::string mt_dat_file = "../data/mt/MersenneTwister.dat"; 
const std::string file_name = "../data/models/MNSimple.obj";
const aiScene *ai_scene = NULL;
scene_ptr scene;

c_kdtree_triangle *kdtree_tri = NULL;

void initialize()
{
	open_console_wnd();
	
	// Initialize the memory pool 
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance(); 
	mem_pool.initialise(256*1024*1024, 256*1024); 

	// Initialize the MT random generator 
	// Init CUDA MT
	srand(1337); 
	c_cuda_rng& rng = c_cuda_rng::get_instance();
	bool ret = rng.init(mt_dat_file); 
	assert(ret); 
	ret = rng.seed(1337);
	assert(ret);  

	triangle_meshes2_array meshes; 
	c_aabb bounds; 
	assimp_import_scene(file_name, &ai_scene);
	assimp_load_meshes2(ai_scene, meshes, bounds); 
	assimp_release_scene(ai_scene); 
	scene_light_array lights;
	scene_material_array mats;
	scene.reset(new c_scene(meshes, bounds, perspective_cam_ptr(), lights, mats)); 
}

void cleanup()
{ 
}

void build_kdtree()
{
	c_triangle_data& tri_data = scene->get_triangle_data();
	kdtree_tri = new c_kdtree_triangle(tri_data); 
	kdtree_tri->build_tree(); 
}

int main(int argc, char **argv) 
{
	initialize();
	
	build_kdtree(); 

	cleanup(); 
	
	return 0; 
}