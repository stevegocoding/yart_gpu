#include "scene.h"
#include "triangle_mesh.h"
#include "cuda_defs.h"
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

c_scene::~c_scene() 
{
	
}


size_t c_scene::get_num_tri_total() const
{
	size_t total = 0; 
	for (size_t i = 0; i < m_meshes.size(); ++i)
	{
		total += m_meshes[i]->get_num_faces();
	}
	return total; 
}


void c_scene::init_triangle_data(triangle_meshes2_array& meshes, const c_aabb& scene_bounds) 
{ 
	c_aabb bounds = scene_bounds; 
	m_tri_data.num_tris = get_num_tri_total();
	m_tri_data.aabb_min = *(float3*)&bounds.pt_min;
	m_tri_data.aabb_max = *(float3*)&bounds.pt_max; 
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();  
	
	// Request texture memory
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(m_tri_data.d_verts[i]), m_tri_data.num_tris*sizeof(float4), "scene_data"));
	}
	
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(m_tri_data.d_normals[i]), m_tri_data.num_tris*sizeof(float4), "scene_data")); 
	}

	/*
	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.request_tex((void**)&(m_tri_data.d_texcoords[i]), m_tri_data.num_tris*sizeof(float2), "scene_data"));
	}
	*/

	// Temporary buffer, used to convert from (x, y, z), (x, y, z), ... to (x, x, ...), (y, y, ...), ...
	
	/*
	float *temp_buf = new float[m_tri_data.num_tris];
	float2 *temp_buf_float2 = new float2[3*m_tri_data.num_tris]; 
	float4 *temp_normal_buf = new float4[3*m_tri_data.num_tris];
	float4 *temp_pos_buf = new float4[3*m_tri_data.num_tris];
	*/
	
	std::vector<float4> temp_pos_buf[3];
	std::vector<float4> temp_normal_buf[3];
	std::vector<float2> temp_uv_buf[3]; 
	std::vector<unsigned int> temp_mat_idx_buf; 

	for (int i = 0; i < 3; ++i)
	{
		temp_pos_buf[i].reserve(m_tri_data.num_tris);
		temp_normal_buf[i].reserve(m_tri_data.num_tris);
		temp_uv_buf[i].reserve(m_tri_data.num_tris);
	}
	temp_mat_idx_buf.reserve(m_tri_data.num_tris); 
	
	for (size_t m = 0; m < get_num_meshes(); ++m)
	{
		triangle_mesh2_ptr mesh = get_triangle_mesh(m); 
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

		for (size_t f = 0; f < num_tris; ++f)
		{
			temp_mat_idx_buf.push_back(mesh->get_mat_idx(f)); 
		}
		 
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
		
		/*
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

				// cudaMemcpy(m_tri_data.d_texcoords, temp_buf_float2, num_tris*eof(float2), cudaMemcpyHostToDevice); 
			}
		} 
		*/
	} 
	
	// Triangle vertices
	for (size_t i = 0; i < 3; ++i)
	{
		
		cuda_safe_call_no_sync(cudaMemcpy(m_tri_data.d_verts[i], (void*)&(temp_pos_buf[i][0]), m_tri_data.num_tris*sizeof(float4), cudaMemcpyHostToDevice));
		cuda_safe_call_no_sync(cudaMemcpy(m_tri_data.d_normals[i], (void*)&(temp_normal_buf[i][0]), m_tri_data.num_tris*sizeof(float4), cudaMemcpyHostToDevice));
		
		// float4 *normal_src = temp_normal_buf + i * m_tri_data.num_tris; 
		// cuda_safe_call_no_sync(cudaMemcpy(m_tri_data.d_normals[i], (void*)&(temp_normal_buf[i][0]), m_tri_data.num_tris*sizeof(float4), cudaMemcpyHostToDevice));  
	}

	// Material Indices 
	cuda_safe_call_no_sync(mem_pool.request_tex((void**)&m_tri_data.d_material_idx, m_tri_data.num_tris*sizeof(uint32), "scene_data"));
	cuda_safe_call_no_sync(cudaMemcpy(m_tri_data.d_material_idx, 
									(void*)&temp_mat_idx_buf[0], 
									m_tri_data.num_tris*sizeof(unsigned int), 
									cudaMemcpyDeviceToHost));
	
	/*
	SAFE_DELETE_ARRAY(temp_buf);
	SAFE_DELETE_ARRAY(temp_buf_float2);
	SAFE_DELETE_ARRAY(temp_pos_buf);
	SAFE_DELETE_ARRAY(temp_normal_buf);
	*/	
}

void c_scene::release_triangle_data()
{
	c_cuda_mem_pool& mem_pool = c_cuda_mem_pool::get_instance();

	for (size_t i = 0; i < 3; ++i)
	{
		cuda_safe_call_no_sync(mem_pool.release(m_tri_data.d_verts[i])); 
		cuda_safe_call_no_sync(mem_pool.release(m_tri_data.d_normals[i]));
		//cuda_safe_call_no_sync(mem_pool.release(tri_data->d_texcoords[i])); 
	}
	cuda_safe_call_no_sync(mem_pool.release(m_tri_data.d_material_idx));
}