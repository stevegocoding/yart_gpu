#include "assimp_loader.h" 
#include "triangle_mesh.h"
#include "scene.h"

#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <Windows.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

void color4_to_float4(const struct aiColor4D *c, float f[4])
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

void set_float4(float f[4], float a, float b, float c, float d)
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}

size_t assimp_load_meshes(const aiScene *scene, triangle_meshes_array& meshes)
{
	for (size_t i = 0; i < scene->mNumMeshes; ++i)
	{
		vertices_array verts; 
		face_indices_array faces; 
		normals_array normals;
		tangents_array tangents; 
		uvs_array uvs;

		size_t num_verts = scene->mMeshes[i]->mNumVertices;
		size_t num_faces = scene->mMeshes[i]->mNumFaces; 

		bool has_normals = scene->mMeshes[i]->HasNormals(); 
		bool has_tangents = scene->mMeshes[i]->HasTangentsAndBitangents(); 
		bool has_uvs = scene->mMeshes[i]->HasTextureCoords(0); 
		
		// Vertex Data
		for (size_t v = 0; v < scene->mMeshes[i]->mNumVertices; ++v)
		{
			point3f vert(scene->mMeshes[i]->mVertices[v].x, scene->mMeshes[i]->mVertices[v].y, scene->mMeshes[i]->mVertices[v].z);
			verts.push_back(vert);

			if (has_normals)
			{
				vector3f normal(scene->mMeshes[i]->mNormals[v].x, scene->mMeshes[i]->mNormals[v].y, scene->mMeshes[i]->mNormals[v].z);
				normals.push_back(normal); 
			}
			if (has_tangents)
			{
				vector3f tangent(scene->mMeshes[i]->mTangents[v].x, scene->mMeshes[i]->mTangents[v].y, scene->mMeshes[i]->mTangents[v].z);
				tangents.push_back(tangent); 
			}
			if (has_uvs)
			{
				uv _uv(scene->mMeshes[i]->mTextureCoords[0][v].x, scene->mMeshes[i]->mTextureCoords[0][v].y);
				uvs.push_back(_uv);
			}
		}
		
		// Triangle Face Data
		for (size_t f = 0; f < scene->mMeshes[i]->mNumFaces; ++f)
		{
			tri_indices face(scene->mMeshes[i]->mFaces[f].mIndices[0], scene->mMeshes[i]->mFaces[f].mIndices[1], scene->mMeshes[i]->mFaces[f].mIndices[2]);
			faces.push_back(face);
		} 
		
		triangle_mesh_ptr mesh = triangle_mesh_ptr(new c_triangle_mesh(verts, normals, tangents, uvs, faces));
		meshes.push_back(mesh);
	}

	return scene->mNumMeshes;
}

//////////////////////////////////////////////////////////////////////////

void print_mesh(std::ostream& os, triangle_mesh2_ptr mesh)
{
	int prec = 4; 
	int width = 8;

	os << "Mesh: " << std::endl; 
	for (int i = 0; i < 3; ++i)
	{
		os << "index: " << i << std::endl;
		point3f *verts = mesh->get_verts(i); 
		
		for (size_t f = 0; f < mesh->get_num_faces(); ++f)
		{
			std::ios::fmtflags old_flags = os.flags(); 
			os.setf(std::ios::left, std::ios::adjustfield); 

			os  << std::setprecision(prec) << std::setw(width) << std::setfill(' ') 
				<< verts[f].x << ' ' 
				<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
				<< verts[f].y << ' ' 
				<< std::setprecision(prec) << std::setw(width) << std::setfill(' ')
				<< verts[f].z << ' ' << std::endl; 

			os.setf(old_flags); 

		}
	} 
}

size_t assimp_load_meshes2(const aiScene *scene, triangle_meshes2_array& meshes, c_aabb& out_bounds)
{
	std::ofstream ofs("mesh_debug.txt");
	
	unsigned int tri_idx = 0; 
	for (size_t i = 0; i < scene->mNumMeshes; ++i)
	{
		size_t num_tris = scene->mMeshes[i]->mNumFaces; 
		size_t num_verts = scene->mMeshes[i]->mNumVertices;
		bool has_normals = scene->mMeshes[i]->HasNormals(); 
		bool has_uvs = scene->mMeshes[i]->HasTextureCoords(0); 

		verts_pos_array pos_array[3]; 
		verts_normal_array normals_array[3];
		verts_uv_array uvs_array[3]; 
		face_mat_idx_array mats_idx_array; 

		for (size_t j = 0; j < 3; ++j)
		{
			pos_array[j].reset(new point3f[num_tris]);
			if (has_normals)
				normals_array[j].reset(new normal3f[num_tris]);
			if (has_uvs)
				uvs_array[j].reset(new vector3f[num_tris]); 
		}
		
		mats_idx_array.reset(new unsigned int[num_tris]);
		
		// Extract triangles data
		aiMesh *ai_mesh = scene->mMeshes[i]; 
		for (size_t f = 0; f < ai_mesh->mNumFaces; ++f)
		{
			aiFace face = ai_mesh->mFaces[f]; 
			
			// Only Triangles!
			assert(face.mNumIndices == 3);
			
			mats_idx_array[f] = ai_mesh->mMaterialIndex; 

			// Add the triangle
			for (size_t t = 0; t < 3; ++t)
			{
				// WARNING: Do not transform the vertices here. This *will* lead to
				// rounding errors that would destroy the triangle connectivity.
				pos_array[t][f] = *(point3f*)&ai_mesh->mVertices[face.mIndices[t]]; 
				
				if (has_normals)
					normals_array[t][f] = *(normal3f*)&ai_mesh->mNormals[face.mIndices[t]]; 
				
				if (has_uvs)
					uvs_array[t][f] = *(vector3f*)&ai_mesh->mTextureCoords[face.mIndices[t]]; 
			}
		}

		triangle_mesh2_ptr mesh = triangle_mesh2_ptr(new c_triangle_mesh2(pos_array, normals_array, uvs_array, mats_idx_array, num_tris, num_verts));
		meshes.push_back(mesh); 

		print_mesh(ofs, mesh);
	}

	// Calculate the bounding box
	for (size_t i = 0; i < scene->mNumMeshes; ++i)
	{
		aiMesh *mesh = scene->mMeshes[i];
		c_aabb mesh_bounds;
		for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
		{
			c_point3f pt = *(c_point3f*)&mesh->mVertices[v]; 
			mesh_bounds = union_aabb(mesh_bounds, pt);
		}
		out_bounds = union_aabb(out_bounds, mesh_bounds); 
	}

	return scene->mNumMeshes;
}

size_t assimp_load_materials(const aiScene *scene, scene_material_array& mats)
{
	// According to ASSIMP doc: If the AI_SCENE_FLAGS_INCOMPLETE flag is not set there 
	// will always be at least ONE material. See
	// http://assimp.sourceforge.net/lib_html/structai_material.html
	
	assert(scene->mNumMaterials > 0);
	
	for (size_t i = 0; i < scene->mNumMaterials; ++i)
	{
		aiMaterial *mat = scene->mMaterials[i];
	
		// 
		aiString name; 
		aiColor4D diff_clr;
		aiColor4D spec_clr;
		float spec_exp = 0.2f; 
		int ret = 0; 

		ret = aiGetMaterialString(mat, AI_MATKEY_NAME, &name); 

		ret = aiGetMaterialColor(mat, AI_MATKEY_COLOR_DIFFUSE, &diff_clr); 
		assert(ret == AI_SUCCESS);

		ret = aiGetMaterialColor(mat, AI_MATKEY_COLOR_SPECULAR, &spec_clr); 
		assert(ret == AI_SUCCESS);

		unsigned int max = 1; 
		ret = aiGetMaterialFloat(mat, AI_MATKEY_SHININESS, &spec_exp);
		assert(ret == AI_SUCCESS); 
		
		std::string mat_name(name.data);
		spec_exp *= 0.25f; 
		scene_material scene_mat(mat_name, *(c_vector3f*)&diff_clr, *(c_vector3f*)&spec_clr, spec_exp, false);
		mats.push_back(scene_mat); 
	}
	
	return mats.size(); 
}

/*
scene_ptr load_scene_from_file(const std::string& file_name)
{
	Assimp::Importer importer;
	
	// Remove lines and points
	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE|aiPrimitiveType_POINT);
	
	const aiScene *scene = importer.ReadFile(file_name, 
		//aiProcess_GenNormals |		// Generate normals
		aiProcess_Triangulate |			// Only triangles
		aiProcess_MakeLeftHanded
		// aiProcess_PreTransformVertices |
		);
	
	assert(scene);
	
	std::vector<triangle_mesh_ptr> meshes; 
	std::vector<light_ptr> lights; 
	assimp_load_meshes(scene, meshes);
	// load_lights(scene, lights);

	aiReleaseImport(scene);

	scene_ptr s = scene_ptr(new c_scene(meshes, lights));
	return s; 
}
*/


void assimp_import_scene(const std::string& file_name, const aiScene **pp_scene)
{
	assert(*pp_scene == NULL); 
	
	/*
	Assimp::Importer importer;

	// Remove lines and points
	importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE|aiPrimitiveType_POINT);

	const aiScene *scene = importer.ReadFile(file_name, 
		//aiProcess_GenNormals |		// Generate normals
		aiProcess_Triangulate |			// Only triangles
		aiProcess_MakeLeftHanded
		// aiProcess_PreTransformVertices |
		);
	
	assert(scene);
	*/

	const aiScene *scene = aiImportFile(file_name.c_str(), 
		aiProcess_Triangulate
		// | aiProcess_MakeLeftHanded
		); 
	assert(scene);

	*pp_scene = scene;  
}

void assimp_release_scene(const aiScene *scene)
{
	assert(scene);

	aiReleaseImport(scene); 
}


void assimp_calc_bounds(const aiMesh *mesh, c_aabb& out_bounds)
{
	for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
	{
		c_point3f pt = *(c_point3f*)&mesh->mVertices[v]; 
		out_bounds = union_aabb(out_bounds, pt);
	}
}