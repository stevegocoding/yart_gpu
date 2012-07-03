#include "cuda_defs.h"
#include "cuda_utils.h"
#include "kernel_data.h"
#include "cuda_mem_pool.h"

texture<float4, 1, cudaReadModeElementType> tex_tri_v0, tex_tri_v1, tex_tri_v2; 

__constant__ c_triangle_data const_tri_data; 

/*
texture<float4, 1, cudaReadModeElementType> tex_tri_n0;
texture<float2, 1, cudaReadModeElementType> tex_tri_texcoord0; 
texture<float2, 1, cudaReadModeElementType> tex_tri_texcoord1; 
texture<float2, 1, cudaReadModeElementType> tex_tri_texcoord2; 
*/

__global__ void kernel_test_triangle_data()
{
	uint32 thread_idx = blockIdx.x * blockDim.x + threadIdx.x; 

	printf("Thread Index: %d \r\n", thread_idx);
	
	
	for (uint32 i = 0; i < const_tri_data.num_tris; ++i)
	{
		float3 v0 = make_float3(tex1Dfetch(tex_tri_v0, i)); 
		float3 v1 = make_float3(tex1Dfetch(tex_tri_v1, i)); 
		float3 v2 = make_float3(tex1Dfetch(tex_tri_v2, i));
		
		printf("V0: %f, %f, %f | V1:  %f, %f, %f | V2: %f, %f, %f \r\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
	}
}


extern "C"
void kernel_wrapper_test_triangle_data(const c_triangle_data& tri_data)
{
	cuda_safe_call_no_sync(cudaMemcpyToSymbol("const_tri_data", &tri_data, sizeof(c_triangle_data)));

	// Set texture parameters and bind the linear memory to the texture.
	cudaChannelFormatDesc cd_float4 = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); 
	//cudaChannelFormatDesc cd_float2 = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);

	// Bind triangle data to texture reference

	// Vertex
	size_t offset = 0;
	tex_tri_v0.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(&offset, tex_tri_v0, tri_data.d_verts[0], cd_float4, tri_data.num_tris*sizeof(float4)));
	tex_tri_v1.normalized = false; 
	cuda_safe_call_no_sync(cudaBindTexture(&offset, tex_tri_v1, tri_data.d_verts[1], cd_float4, tri_data.num_tris*sizeof(float4)));
	tex_tri_v2.normalized = false;
	cuda_safe_call_no_sync(cudaBindTexture(&offset, tex_tri_v2, tri_data.d_verts[2], cd_float4, tri_data.num_tris*sizeof(float4)));
	
	
	/*
	// Normals
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_n0, const_tri_data.d_normals[0], cd_float4, const_tri_data.num_tris*sizeof(float4)));
	
	// Texture coordinates 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_texcoord0, const_tri_data.d_texcoords[0], cd_float2, const_tri_data.num_tris*sizeof(float2))); 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_texcoord1, const_tri_data.d_texcoords[1], cd_float2, const_tri_data.num_tris*sizeof(float2))); 
	cuda_safe_call_no_sync(cudaBindTexture(NULL, tex_tri_texcoord2, const_tri_data.d_texcoords[2], cd_float2, const_tri_data.num_tris*sizeof(float2))); 
	*/

	dim3 block_size = dim3(1, 1, 1);
	dim3 grid_size = dim3(1, 1, 1);
	
	kernel_test_triangle_data<<<grid_size, block_size>>>();
}

extern "C"
void rt_cleanup_kernel_data()
{
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v0));
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v1));
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_v2));

	/*
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_n0));
	
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_texcoord0)); 
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_texcoord1));
	cuda_safe_call_no_sync(cudaUnbindTexture(tex_tri_texcoord2));
	*/
};