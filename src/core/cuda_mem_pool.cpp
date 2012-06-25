#include <assert.h>
#include <ctime>
#include "cuda_mem_pool.h"
#include "cuda_utils.h"

c_mem_chunk::c_mem_chunk(size_t _size, bool pinned_host, bool removeable, cudaError_t& out_error)
	: chunk_size(_size)
	, is_host_pinned(pinned_host)
	, is_removeable(removeable)
{
	if (is_host_pinned)
	{
		// Allocate pinned (non-pageable) host memory.
		out_error = cudaMallocHost(&d_buffer, chunk_size); 
		if (out_error != cudaSuccess)
			return; 
	}
	else 
	{
		// Allocate device memory
		out_error = cudaMalloc(&d_buffer, chunk_size); 
		if (out_error != cudaSuccess)
			return; 
	}

	assigned_segs.clear(); 
}

c_mem_chunk::~c_mem_chunk()
{
	if (is_host_pinned)
	{
		cudaFreeHost(d_buffer); 
	}
	else 
	{
		cudaFree(d_buffer); 
	}
}

void *c_mem_chunk::request(size_t size_bytes, size_t alignment, const std::string& cat)
{
	assert(size_bytes > 0); 
	
	// Find first fit
	std::list<c_assigned_segment>::iterator it;
	size_t offset, free_to;
	c_assigned_segment *prev = NULL; 

	// If there is already more than ONE segments
	for (it = assigned_segs.begin(); it != assigned_segs.end(); ++it)
	{
		c_assigned_segment *p = (c_assigned_segment*)&(*it);

		// Check space *before* current segment.
		offset = 0; 
		if (prev != NULL) 
			offset = CUDA_ALIGN_BYTES(prev->offset + prev->size_bytes, alignment);
		free_to = p->offset;
		
		if (free_to > offset && free_to - offset >= size_bytes)
		{
			// Found fit. To keep order, place fit right before assigned segment.
			void *buf = ((unsigned char*)d_buffer + offset);
			assigned_segs.insert(it, c_assigned_segment(offset, size_bytes, buf, cat));
			last_use = time(NULL);
			return buf;
		}
		prev = p; 
	}

	// Now check space after the last segment or from the beginning
	offset = 0; 
	if (assigned_segs.size())
	{
		c_assigned_segment *last = &assigned_segs.back(); 
		offset = CUDA_ALIGN_BYTES(last->offset + last->size_bytes, alignment); 
	}
	free_to = chunk_size;
	if (free_to > offset && free_to - offset >= size_bytes)
	{
		// Found fit. Just attach at end.
		void *buf = ((unsigned char*)d_buffer + offset);
		assigned_segs.push_back(c_assigned_segment(offset, size_bytes, buf, cat));
		last_use = time(NULL);
		return buf; 
	}

	// Nothing found
	return NULL; 
}

size_t c_mem_chunk::release(void* d_buffer)
{
	std::list<c_assigned_segment>::iterator it; 
	for (it = assigned_segs.begin(); it != assigned_segs.end(); ++it)
	{
		c_assigned_segment *p = (c_assigned_segment*)&(*it); 
		
		if (p->d_buf == d_buffer)
		{	
			size_t freed_bytes = p->size_bytes; 
			assigned_segs.erase(it); 
			last_use = time(NULL); 
			return freed_bytes;
		}
	}
	
	return 0; 
}

bool c_mem_chunk::is_obsolete(time_t current) const
{
	return is_removeable && assigned_segs.size() == 0 && (current - last_use >= 5);
}

size_t c_mem_chunk::get_assigned_size() const 
{
	size_t total = 0; 

	std::list<c_assigned_segment>::const_iterator it; 
	for (it = assigned_segs.begin(); it != assigned_segs.end(); ++it)
	{
		c_assigned_segment *p = (c_assigned_segment*)&(*it);
		total += p->size_bytes; 
	}
	return total; 
}

//////////////////////////////////////////////////////////////////////////

c_cuda_mem_pool::c_cuda_mem_pool()
	: m_is_inited(false)
	, m_num_bytes_assigned(0)
	, m_pinned_host_chunk(NULL)
{
}

c_cuda_mem_pool::~c_cuda_mem_pool()
{
	free(); 
}

cudaError_t c_cuda_mem_pool::initialise(size_t init_size, size_t pinned_host_size)
{
	if (m_is_inited)
	{
		assert(false); 
		return cudaErrorInitializationError; 
	}
	
	cudaError_t err = cudaSuccess; 

	// Get texture alignment requirement (bytes).

	cudaDeviceProp props; 
	int device; 
	err = cudaGetDevice(&device); 
	if (err != cudaSuccess)
		return err; 
	
	err = cudaGetDeviceProperties(&props, device); 
	m_tex_alignment = props.textureAlignment; 

	// Store initial chunk size, but do not allocate yet.
	m_initial_size = init_size; 
	
	// Allocate pinned host memory chunk if required.
	m_pinned_host_chunk = NULL; 
	if (pinned_host_size > 0)
	{
		m_pinned_host_chunk = new c_mem_chunk(pinned_host_size, true, false, err); 
		if (err != cudaSuccess)
		{
			SAFE_DELETE(m_pinned_host_chunk); 
			return err; 
		}
	}

	m_is_inited = true; 
	
	return err; 
}

cudaError_t c_cuda_mem_pool::request(void **d_buf, size_t size, const std::string& cat /* = "general" */, size_t alignment /* = 64 */)
{
	if (!m_is_inited)
	{
		assert(false); 
		return cudaErrorNotReady; 
	}
	if (size == 0)
	{
		assert(false);
		return cudaErrorInvalidValue; 
	}
	
	cudaError_t err = cudaSuccess;
	
	while (true)
	{
		// Find a free range. For now, just use the first fit.
		chunks_list::iterator it; 
		bool done = false;
		
		for (it = m_device_chunks.begin(); it != m_device_chunks.end(); ++it)
		{
			c_mem_chunk *chunk = *it;
			
			// Try to use this chunk.
			void *buf = chunk->request(size, alignment, cat); 
			if (buf)
			{
				// Found space.
				*d_buf = buf; 
				m_num_bytes_assigned += size; 
				err = cudaSuccess; 
				done = true; 
				break; 
			}
		}
		
		if (done)
			break; 

		// When getting here, there is no more space left in our chunks.
		// Therefore allocate a new device memory chunk.

		// Get the amount of free memory first. Ensure we still have enough memory left.
		// Not available in device emu mode!
		size_t free, total; 
		CUresult res = cuMemGetInfo(&free, &total); 
		if (res != CUDA_SUCCESS)
		{
			err = cudaErrorUnknown; 
			break; 
		}
		
		// No more device memory available 
		if (free < size)
		{
			err = cudaErrorMemoryValueTooLarge; 
			yart_log_message("There is no more device memory");
			break; 
		}
		
		// Avoid allocating too much memory by reserving 100 MB for other use.
		const size_t reserved = 100 * 1024 * 1024; 
		size_t free_for_us = 0; 
		if (free > reserved)
			free_for_us = free - reserved; 
		
		// Use a maximum chunk size. Doubling the chunks does not lead to good results as it
		// would fill the whole memory in a few steps. This chunk size can only be enlarged
		// if a given request needs more memory.
		size_t new_size; 
		if (m_device_chunks.size() == 0)
			new_size = m_initial_size; 
		else 
		{
			c_mem_chunk *last = m_device_chunks.back(); 
			new_size = std::min(free_for_us, std::max(std::min(last->chunk_size*2, (size_t)100*1024*1024), size));
		}
		
		if (free_for_us == 0)
		{
			// No more memory available for us.
			assert(false);
		}
		
		err = alloc_chunk(new_size); 
		if (err != cudaSuccess)
			break; 

		// Use the new chunk.
		c_mem_chunk *new_chunk = m_device_chunks.back();
		void *buf = new_chunk->request(size, alignment, cat);
		if (buf)
		{
			*d_buf = buf; 
			m_num_bytes_assigned += size; 
			err = cudaSuccess; 
			break; 
		}

		err = cudaErrorMemoryValueTooLarge;
		break;
	}
	
	return err; 
}

cudaError_t c_cuda_mem_pool::request_tex(void **d_buf, size_t size, const std::string& cat /* = "general" */)
{
	return request(d_buf, size, cat, m_tex_alignment);
}

cudaError_t c_cuda_mem_pool::release(void *d_buf)
{
	if (!m_is_inited)
	{
		return cudaSuccess; 
	}
	
	cudaError_t err = cudaErrorInvalidDevicePointer; 
	
	// Find the associated segment.
	chunks_list::iterator it; 
	for (it = m_device_chunks.begin(); it != m_device_chunks.end(); ++it)
	{
		c_mem_chunk *chunk = *it; 
		size_t freed =  chunk->release(d_buf);
		if (freed)
		{
			m_num_bytes_assigned -= freed; 
			err = cudaSuccess;
			break; 
		}
	}
	
	return err;
}

void c_cuda_mem_pool::free()
{
	if (!m_is_inited)
		return; 
	
	chunks_list::iterator it; 
	for (it = m_device_chunks.begin(); it != m_device_chunks.end(); ++it)
	{
		SAFE_DELETE(*it); 
	}
	m_device_chunks.clear(); 

	SAFE_DELETE(m_pinned_host_chunk);
	m_is_inited = false; 
}

cudaError_t c_cuda_mem_pool::alloc_chunk(size_t size_bytes)
{
	assert(size_bytes > 0);
	
	cudaError_t err = cudaSuccess; 
	bool removeable = (m_device_chunks.size() > 0);
	c_mem_chunk *new_chunk = new c_mem_chunk(size_bytes, false, removeable, err);
	m_device_chunks.push_back(new_chunk); 
	return err; 
}

size_t c_cuda_mem_pool::get_allcated_size() const 
{
	if (!m_is_inited)
		return 0; 

	size_t count = 0; 

	chunks_list::const_iterator it; 
	for (it = m_device_chunks.begin(); it != m_device_chunks.end(); ++it)
	{
		count += (*it)->chunk_size; 
	}
	
	return count; 
}

c_cuda_mem_pool& c_cuda_mem_pool::get_instance()
{
	static c_cuda_mem_pool mem_pool; 
	return mem_pool;
}

//////////////////////////////////////////////////////////////////////////


