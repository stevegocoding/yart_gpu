#ifndef __kdtree_debug_h__
#define __kdtree_debug_h__

#pragma once  

#include <iostream> 
#include "kdtree_kernel_data.h"

void print_node_list(std::ostream& os, c_kd_node_list *node_list);
void print_final_node_list(std::ostream& os, c_kd_final_node_list *final_node_list); 
void print_chunks_list(std::ostream& os, c_kd_chunk_list *chunks_list);

void print_device_array(std::ostream& os, uint32 *d_array, uint32 count); 

#endif // __kdtree_debug_h__
