#ifndef __cuda_primitives_h__
#define __cuda_primitives_h__

#pragma once

#include <assert.h>
#include <map>
#include "prerequisites.h"
#include "cudpp/cudpp.h"
 
class c_cuda_primitives
{
public:
	~c_cuda_primitives();

	static c_cuda_primitives& get_instance();

	// ---------------------------------------------------------------------
	/*
		Destroys all CUDPP plans. 
		Call this just before \c cudaThreadExit() to avoid errors.
	*/ 
	// ---------------------------------------------------------------------
	void destroy_plans();

	// ---------------------------------------------------------------------
	/*
		Compacts given array \a d_in using binary array \a d_isValid.
		
		Note that the number of compacted elements is returned in form of device memory.
		This was done to try to avoid the read back to host memory when possible. This
		operation may not be used inplace. 
		
	*/ 
	// ---------------------------------------------------------------------
	void compact(const void *d_in, const unsigned *d_is_valid, size_t num_elems, void *d_out_compacted, size_t *d_out_new_count);

	// ---------------------------------------------------------------------
	/*
	/// \brief	Performs scan operation on given array \a d_in. Result is stored in \a d_out.
	/// 		
	/// 		The performed scan has the parameters ADD, FWD. Inplace scan is supported.
	/// 		
	/// 		\warning Use for 32-bit data types only, e.g. ::uint.  
	///
	/// \param [in]		d_in	The data to scan (device memory). 
	/// \param	numElems		Number of \em elements in \a d_in. 
	/// \param	bInclusive		Whether to use inclusive or exclusive scan. 
	/// \param [out]	d_out	The result of scanning \a d_in (device memory). According to CUDPP
	/// 						developers, this can be equal to \a d_in (inplace operation). 
	*/ 
	// ---------------------------------------------------------------------
	void scan(const void *d_in, size_t num_elems, bool inclusive, void *d_out);
	
private:
	
	// Types of parallel primitives supported by c_cuda_primitives class.
	enum e_cuda_prim_type
	{
		// Scan (ADD, FWD, EXCLUSIVE).
		cuda_prim_scan_add_exc,
		
		// Scan (ADD, FWD, INCLUSIVE).
		cuda_prim_scan_add_inc, 
		
		// Compact 
		cuda_prim_compact,

		// Segmented scan (ADD, FWD, INCLUSIVE).
		cuda_prim_seg_scan_add_inc,

		// Segmented scan (ADD, FWD, EXCLUSIVE).
		cuda_prim_seg_scan_add_exc,
		
		// Sort plan (KEY-VALUE-PAIR radix sort).
		cuda_prim_sort_key_val 
	};

	// Holds primitive information including CDUPP plan handle. 
	class c_cuda_primitive
	{
	public:
		c_cuda_primitive(const CUDPPConfiguration& config, const CUDPPHandle& plan, size_t max_num_elems)
			: m_config(config)
			, m_plan(plan)
			, m_max_num_elems(max_num_elems)
		{
			
		}

	public:
		
		CUDPPConfiguration m_config;
		CUDPPHandle m_plan; 
		size_t m_max_num_elems; 
	};

	c_cuda_primitives();
	c_cuda_primitives(const c_cuda_primitives& other);


	void init_cudpp();
	void destroy_cudpp();
	
	void create_plans(size_t max_elems);
	void create_plan(e_cuda_prim_type type, const CUDPPConfiguration& config, size_t max_elems);

	bool check_plan_size(e_cuda_prim_type type, size_t required_max_elems);

	std::map<e_cuda_prim_type, c_cuda_primitive*> m_prims_map; 

	// CUDPP Library Handle
	CUDPPHandle m_cudpp_handle;
};


#endif // __cuda_primitives_h__
