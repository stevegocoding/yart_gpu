#ifndef __assimp_loader_h__
#define __assimp_loader_h__

#pragma once

#include <vector>
#include <boost/shared_ptr.hpp>

#include "prerequisites.h"
#include "assimp/assimp.hpp"
#include "assimp/assimp.h"
#include "assimp/aiPostProcess.h"
#include "assimp/aiScene.h"
#include "assimp/aiMesh.h"
#include "bounding_box.h"

class c_triangle_mesh; 
typedef boost::shared_ptr<c_triangle_mesh> triangle_mesh_ptr; 
typedef std::vector<triangle_mesh_ptr> triangle_meshes_array; 

class c_triangle_mesh2; 
typedef boost::shared_ptr<c_triangle_mesh2> triangle_mesh2_ptr; 
typedef std::vector<triangle_mesh2_ptr> triangle_meshes2_array; 

struct scene_material; 
typedef std::vector<scene_material> scene_material_array; 

void assimp_import_scene(const std::string& file_name, const aiScene **pp_scene);
void assimp_release_scene(const aiScene *scene); 
size_t assimp_load_meshes(const aiScene *scene, triangle_meshes_array& meshes); 
size_t assimp_load_meshes2(const aiScene *scene, triangle_meshes2_array& meshes, c_aabb& out_bounds);
size_t assimp_load_materials(const aiScene *scene, scene_material_array& mats); 

void assimp_calc_bounds(const aiMesh *mesh, c_aabb& out_bounds); 

#endif // __assimp_loader_h__
