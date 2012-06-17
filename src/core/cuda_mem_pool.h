#ifndef __cuda_mem_pool_h__
#define __cuda_mem_pool_h__

#pragma once 

#include <list>
#include <cuda_runtime.h>
#include "prerequisites.h"

/// Keeps track of assigned device memory segments. 
class c_assigned_segment 
{
public: 
	c_assigned_segment(size_t _offset, size_t _size, void *buf, const std::string& cat)
		: offset(_offset)
		, size_bytes(_size)
		, d_buf(buf)
		, mem_category(cat)
	{
	}
	
	// Byte offset. Relative to owning chunk base offset.
	// Has to be aligned.
	size_t offset; 
	
	// Number of bytes
	size_t size_bytes;

	// Pointer to the device memory segment.
	void *d_buf;

	std::string mem_category;
};

//////////////////////////////////////////////////////////////////////////

/// Keeps track of allocated device or pinned host memory.
class c_mem_chunk
{

public:
	c_mem_chunk(size_t _size, bool pinned_host, bool removeable, cudaError_t& out_error);
	~c_mem_chunk();

	// Returns a pointer to the requested memory space within this chunk, if any.
	// Else NULL is returned. In the first case the free range list is updated.
	void* request(size_t size_bytes, size_t alignment, const std::string& cat);

	// Releases the given buffer if it is assigned within this chunk. Number of
	// free'd bytes is returned, else 0.
	size_t release(void* d_buffer);
	
	// Returns the assigned size within this segment.
	size_t get_assigned_size() const;
	
	/*
	// Sets whether the chunk is removeable or not.
	void SetRemoveable(bool b) { isRemoveable = b; }
	
	// Returns true when this chunk is obsolete and can be destroyed.
	bool IsObsolete(time_t tCurrent) const;
	
	// Tests this chunk for errors, e.g. non disjoint segments.
	bool Test(FILE* stream) const;
	*/

	// If true, the chunk can be killed when not used for some time. Default: true.
	bool is_removeable; 

	// If true, this is a pinned host memory chunk. Else it's a device memory chunk.
	bool is_host_pinned;

	// Size in bytes.
	size_t chunk_size; 

	// Memory of *sizeChunk* bytes.
	void *d_buffer;

	// Assigned segments. Ordered by offsets.
	std::list<c_assigned_segment> assigned_segs;

	// Time of last use (time(), seconds). Used to detect obsolete chunks.
	time_t last_use; 
};

//////////////////////////////////////////////////////////////////////////

class c_mem_pool 
{
	typedef std::list<c_mem_chunk*> chunks_list;

public:
	c_mem_pool();
	~c_mem_pool();

	cudaError_t initialise(size_t init_size, size_t pinned_host_size);

	cudaError_t request(void **d_buf, size_t size, const std::string& cat = "general", size_t alignment = 64); 
	
	cudaError_t request_tex(void **d_buf, size_t size, const std::string& cat = "general"); 
	
	cudaError_t release(void *d_buf); 

	
private:
	
	cudaError_t pre_alloc(size_t initial_size, size_t pinned_host_size);
	
	// Free all the memory. 
	void free(); 

	// Allocates a new chunk of device memory. Used for pool resizing.
	cudaError_t alloc_chunk(size_t size_bytes); 
	
	// Number of bytes currently assigned.
	size_t m_num_bytes_assigned;
	
	// Texture alignment requirement for current device.
	size_t m_tex_alignment; 

	/// Initial segment size in bytes.
	size_t m_initial_size;

	bool m_is_inited; 
	
	// Device memory chunks.
	chunks_list m_device_chunks; 

	c_mem_chunk *m_pinned_host_chunk; 
};



#endif // __cuda_mem_pool_h__