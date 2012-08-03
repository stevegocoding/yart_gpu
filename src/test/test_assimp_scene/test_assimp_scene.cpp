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

extern "C"
void kernel_wrapper_test_triangle_data(const c_triangle_data& tri_data);

const std::string file_name = "../data/models/MNSimple.obj";
const aiScene *ai_scene = NULL;
scene_ptr scene;
c_triangle_data tri_data; 

void initialise()
{
	std::ofstream ofs("debug.txt");

	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	mem_pool.initialise(256*1024*1024, 256*1024);

	triangle_meshes2_array meshes; 
	scene_material_array materials; 
	c_aabb bounds; 
	assimp_import_scene(file_name, &ai_scene);
	assimp_load_meshes2(ai_scene, meshes, bounds);
	assimp_load_materials(ai_scene, materials);
	assimp_release_scene(ai_scene);
	
	// scene.reset(new c_scene(meshes, bounds));

	// Copy data back from device to host
	
	float4 *h_verts[3];
	for (int i = 0; i < 3; ++i)
	{
		h_verts[i] = new float4[tri_data.num_tris];
		
		cuda_safe_call_no_sync(cudaMemcpy(h_verts[i], tri_data.d_verts[i], tri_data.num_tris*sizeof(float4), cudaMemcpyDeviceToHost));
	}
	
	for (int i = 0; i < 3; ++i)
	{
		for (size_t f = 0; f < tri_data.num_tris; ++f) 
		{
			print_float4(ofs, h_verts[i][f]);
		}
	}
	
	for (int i = 0; i < 3; ++i)
	{
		SAFE_DELETE_ARRAY(h_verts[i]);
	}
	
	ofs.clear();
}

void cleanup()
{
}


int main(int argc, char **argv)
{ 
	initialise(); 

	kernel_wrapper_test_triangle_data(tri_data);

	cleanup();
	
	return 0; 
}
