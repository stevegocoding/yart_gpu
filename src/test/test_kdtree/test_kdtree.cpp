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
#include "scene.h"
#include "assimp_loader.h"

const std::string file_name = "../data/models/cube.ply";
const aiScene *ai_scene = NULL;
scene_ptr scene;
c_triangle_data tri_data;  
 
void initialize()
{
	/*
	triangle_meshes2_array meshes; 
	assimp_import_scene(file_name, &ai_scene);
	assimp_load_meshes2(ai_scene, meshes); 
	assimp_release_scene(ai_scene);

	scene.reset(new c_scene(meshes));
	*/ 
}

int main(int argc, char **argv)
{
	return 0; 
}