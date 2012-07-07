#include "cuda_primitives.h"
#include "utils.h"

c_cuda_primitives& c_cuda_primitives::get_instance()
{
	static c_cuda_primitives prims;
	return prims;
}

c_cuda_primitives::c_cuda_primitives()
{
	init_cudpp();
	create_plans(1024*1024);
}

c_cuda_primitives::c_cuda_primitives(const c_cuda_primitives& other)
{
	
}

c_cuda_primitives::~c_cuda_primitives()
{
	destroy_plans();
	destroy_cudpp();
}

void c_cuda_primitives::init_cudpp()
{
	CUDPPResult res = cudppCreate(&m_cudpp_handle);
	if (res != CUDPP_SUCCESS)
		yart_log_message("Failed to create CUDPP handle!");
}

void c_cuda_primitives::destroy_cudpp()
{
	CUDPPResult res = cudppDestroy(m_cudpp_handle);
	if (res != CUDPP_SUCCESS)
		yart_log_message("Failed to destory CUDPP handle!");
	
}

void c_cuda_primitives::create_plans(size_t max_elems)
{
	m_prims_map.clear();

	// Create scan plan.
	CUDPPConfiguration scan_config; 
	scan_config.op = CUDPP_ADD;
	scan_config.datatype = CUDPP_UINT; 
	scan_config.algorithm = CUDPP_SCAN; 
	scan_config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
	create_plan(cuda_prim_scan_add_exc, scan_config, max_elems);
	
	scan_config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE; 
	create_plan(cuda_prim_scan_add_exc, scan_config, max_elems);

	// Create compact plan.
	CUDPPConfiguration compact_config;
	compact_config.datatype = CUDPP_UINT;
	compact_config.algorithm = CUDPP_COMPACT;
	compact_config.options = CUDPP_OPTION_FORWARD;
	create_plan(cuda_prim_compact, compact_config, max_elems);

	// Create segmented scan plan (inclusive).
	CUDPPConfiguration segscan_config;
	segscan_config.op = CUDPP_ADD;
	segscan_config.datatype = CUDPP_UINT;
	segscan_config.algorithm = CUDPP_SEGMENTED_SCAN;
	segscan_config.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_INCLUSIVE;
	create_plan(cuda_prim_seg_scan_add_inc, segscan_config, max_elems);

	// Create segmented scan plan (exclusive).
	segscan_config.options = CUDPP_OPTION_FORWARD|CUDPP_OPTION_EXCLUSIVE;
	create_plan(cuda_prim_seg_scan_add_exc, segscan_config, max_elems);

	// Create key-value-pair sort plan.
	CUDPPConfiguration sort_config;
	sort_config.datatype = CUDPP_UINT;
	sort_config.algorithm = CUDPP_SORT_RADIX;
	sort_config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;
	create_plan(cuda_prim_sort_key_val, sort_config, max_elems);
}

void c_cuda_primitives::destroy_plans()
{
	std::map<e_cuda_prim_type, c_cuda_primitive*>::iterator it;
	for (it = m_prims_map.begin(); it != m_prims_map.end(); ++it)
	{
		c_cuda_primitive *prim = it->second; 
		if (CUDPP_SUCCESS != cudppDestroyPlan(prim->m_plan))
			yart_log_message("Failed to destroy CUDPP plan (op: %d).", it->first);

		SAFE_DELETE(prim);
	}
	
	m_prims_map.clear();
}


void c_cuda_primitives::create_plan(e_cuda_prim_type type, const CUDPPConfiguration& config, size_t max_elems)
{
	// Check if plan is already created. 
	if (m_prims_map.find(type) != m_prims_map.end())
	{
		// Already in map. So recreate.
		c_cuda_primitive *prim = m_prims_map[type];

		if (CUDPP_SUCCESS != cudppDestroyPlan(prim->m_plan))
			yart_log_message("Failed to destroy CUDPP plan (op: %d).", type);
		
		if (CUDPP_SUCCESS != cudppPlan(m_cudpp_handle, &prim->m_plan, config, max_elems, 1, 0))
			yart_log_message("Failed to create CUDPP plan (op: %d).", type);
	}
	else 
	{
		CUDPPHandle plan;
		if (CUDPP_SUCCESS != cudppPlan(m_cudpp_handle, &plan, config, max_elems, 1, 0))
			yart_log_message("Failed to create CUDPP plan (op: %d).", type);
		
		c_cuda_primitive *prim = new c_cuda_primitive(config, plan, max_elems);
		m_prims_map[type] = prim;
	}
}

void c_cuda_primitives::compact(const void *d_in, const unsigned *d_is_valid, size_t num_elems, void *d_out_compacted, size_t *d_out_new_count)
{
	assert(num_elems > 0 && d_in && d_is_valid && d_out_compacted && d_out_new_count);
	check_plan_size(cuda_prim_compact, num_elems);

	c_cuda_primitive *prim = m_prims_map[cuda_prim_compact];
	if (CUDPP_SUCCESS != cudppCompact(prim->m_plan, d_out_compacted, d_out_new_count, d_in, d_is_valid, num_elems))
		yart_log_message("Failed to run CUDPP compact.");
}

void c_cuda_primitives::scan(const void *d_in, size_t num_elems, bool inclusive, void *d_out)
{
	assert(d_in && num_elems > 0 && d_out);

	if (inclusive)
	{
		check_plan_size(cuda_prim_scan_add_inc, num_elems);
		c_cuda_primitive *prim = m_prims_map[cuda_prim_scan_add_inc];
		if (CUDPP_SUCCESS != cudppScan(prim->m_plan, d_out, d_in, num_elems))
		{
			yart_log_message("Failed to run CUDPP inclusive scan.");
		}
	}
	else 
	{
		check_plan_size(cuda_prim_scan_add_exc, num_elems);
		c_cuda_primitive* prim = m_prims_map[cuda_prim_scan_add_exc];
		if (CUDPP_SUCCESS != cudppScan(prim->m_plan, d_out, d_in, num_elems))
			yart_log_message("Failed to run CUDPP exclusive scan.");
		
	}
	
}

bool c_cuda_primitives::check_plan_size(e_cuda_prim_type type, size_t required_max_elems)
{
	assert(m_prims_map.find(type) != m_prims_map.end());
	
	// Check if the current plan is large enough.
	c_cuda_primitive *prim  = m_prims_map[type];
	if (prim->m_max_num_elems >= required_max_elems)
		return false;
	
	// New size
	size_t new_max = std::max(required_max_elems, 2*prim->m_max_num_elems);

	// Recreate 
	yart_log_message("Recreating CUDPP plan (op: %d, old size: %d; new size: %d).\n", type, prim->m_max_num_elems, new_max); 
	create_plan(type, prim->m_config, new_max);
	return true; 

}