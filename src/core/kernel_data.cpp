#include "kernel_data.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"
#include "triangle_mesh.h"
#include "scene.h"

#include <iostream>

void init_device_triangle_data(c_triangle_data *tri_data, c_scene *scene)
{
	assert(tri_data && scene); 

	tri_data->num_tris = scene->get_num_tri_total();
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();  
	
	// Request texture memory
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(tri_data->d_verts[i]), tri_data->num_tris*sizeof(float4), "scene_data"));
	}
	/*
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(tri_data->d_normals[i]), tri_data->num_tris*sizeof(float4), "scene_data")); 
	}

	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(tri_data->d_texcoords[i]), tri_data->num_tris*sizeof(float2), "scene_data"));
	}
	*/

	// Temporary buffer, used to convert from (x, y, z), (x, y, z), ... to (x, x, ...), (y, y, ...), ...
	
	/*
	float *temp_buf = new float[tri_data->num_tris];
	float2 *temp_buf_float2 = new float2[3*tri_data->num_tris]; 
	float4 *temp_normal_buf = new float4[3*tri_data->num_tris];
	float4 *temp_pos_buf = new float4[3*tri_data->num_tris];
	*/
	
	std::vector<float4> temp_pos_buf[3];
	std::vector<float4> temp_normal_buf[3];
	std::vector<float2> temp_uv_buf[3]; 

	for (int i = 0; i < 3; ++i)
	{
		temp_pos_buf[i].reserve(tri_data->num_tris);
		temp_normal_buf[i].reserve(tri_data->num_tris);
		temp_uv_buf[i].reserve(tri_data->num_tris);
	}
	
	// tri_data->d_material_idx = NULL;
	
	for (size_t m = 0; m < scene->get_num_meshes(); ++m)
	{
		triangle_mesh2_ptr mesh = scene->get_triangle_mesh(m); 
		size_t num_tris = mesh->get_num_faces(); 

		// Triangle vertices.
		for (size_t i = 0; i < 3; ++i)
		{
			// Convert to float4  
			float3 *src = (float3*)mesh->get_verts(i);

			for (size_t f = 0; f < num_tris; ++f)
				temp_pos_buf[i].push_back(make_float4(*src++)); 
			
			// cuda_safe_call_no_sync(cudaMemcpy(d_pdest, temp_buf_float4, num_tris*sizeof(float4), cudaMemcpyHostToDevice)); 
		} 
		
		/*
		if (mesh->has_normal())
		{ 
			// Triangle normals 
			for (size_t i = 0; i < 3; ++i)
			{
				// float4 *pdest = temp_normal_buf + i*num_tris;
				float3 *src = (float3*)mesh->get_normals(i);

				for (size_t f = 0; f < num_tris; ++f)
					// *pdest++ = make_float4(*src++); 
					temp_normal_buf[i].push_back(make_float4(*src++));

				// cuda_safe_call_no_sync(cudaMemcpy(d_pdest, temp_buf_float4, num_tris*sizeof(float4), cudaMemcpyHostToDevice)); 
			}
		} 
		
		if (mesh->has_uvs())
		{
			// Texture coordinates 
			for (size_t v = 0; v < 3; ++v)
			{
				// float2 *pdest = temp_buf_float2;
				vector3f *src = mesh->get_texcoords(v);

				for (size_t j = 0; j < num_tris; ++j)
				{
					float2 uv = make_float2((*src)[0],  (*src)[1]);
					temp_uv_buf[v].push_back(uv);
					src++;
				}

				// cudaMemcpy(tri_data->d_texcoords, temp_buf_float2, num_tris*eof(float2), cudaMemcpyHostToDevice); 
			}
		} 
		*/
	} 
	
	for (size_t i = 0; i < 3; ++i)
	{
		
		cuda_safe_call_no_sync(cudaMemcpy(tri_data->d_verts[i], (void*)&(temp_pos_buf[i][0]), tri_data->num_tris*sizeof(float4), cudaMemcpyHostToDevice));
		
		
		// float4 *normal_src = temp_normal_buf + i * tri_data->num_tris; 
		// cuda_safe_call_no_sync(cudaMemcpy(tri_data->d_normals[i], (void*)&(temp_normal_buf[i][0]), tri_data->num_tris*sizeof(float4), cudaMemcpyHostToDevice));  
	}
	
	/*
	SAFE_DELETE_ARRAY(temp_buf);
	SAFE_DELETE_ARRAY(temp_buf_float2);
	SAFE_DELETE_ARRAY(temp_pos_buf);
	SAFE_DELETE_ARRAY(temp_normal_buf);
	*/
	
	
}

void release_device_triangle_data(c_triangle_data *tri_data)
{
	assert(tri_data);
	
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.release(tri_data->d_verts[i])); 
		//cuda_safe_call_no_sync(mem_pool.release(tri_data->d_normals[i]));
		//cuda_safe_call_no_sync(mem_pool.release(tri_data->d_texcoords[i])); 
	}	
}

//////////////////////////////////////////////////////////////////////////

void c_shading_points_array::alloc_mem(uint32 _max_points)
{
	assert(_max_points > 0); 
	
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	max_pts = CUDA_ALIGN_EX(_max_points, mem_pool.get_allcated_size()/sizeof(float));
	num_pts = 0; 

	cuda_safe_call_no_sync(mem_pool.request((void**)&d_pixels_buf, max_pts*sizeof(uint32), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_tri_idices, max_pts*sizeof(int), "shading_pts"));
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_isect_pts, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_geo_normals, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_shading_normals, max_pts*sizeof(float4), "shading_pts")); 
	cuda_safe_call_no_sync(mem_pool.request((void**)&d_isect_bary_coords, max_pts*sizeof(float2), "shading_pts")); 
}

void c_shading_points_array::destroy()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();
	cuda_safe_call_no_sync(mem_pool.release(d_pixels_buf));
	cuda_safe_call_no_sync(mem_pool.release(d_tri_idices));
	cuda_safe_call_no_sync(mem_pool.release(d_isect_pts));
	cuda_safe_call_no_sync(mem_pool.release(d_geo_normals));
	cuda_safe_call_no_sync(mem_pool.release(d_shading_normals));
	cuda_safe_call_no_sync(mem_pool.release(d_isect_bary_coords));	
	
	max_pts = 0; 
	num_pts = 0; 
}