#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

#include <shrQATest.h>
#include <shrUtils.h>
#include <sdkHelper.h>


float *host_a = NULL; 
float *host_b = NULL; 
float *host_c = NULL; 
float *device_a = NULL; 
float *device_b = NULL; 
float *device_c = NULL; 

void random_init(float *a, int n)
{
	static int dd = 0; 
	
	dd++;
	
	for (int i = 0; i < n; ++i)
	{
		// a[i] = rand() / (float)RAND_MAX; 
		a[i] = (float)dd; 
	}
}

void cleanup_resource()
{
	if (device_a)
		cudaFree(device_a);
	if (device_b)
		cudaFree(device_b);
	if (device_c)
		cudaFree(device_c);

	if (host_a)
		delete[] host_a; 
	if (host_b)
		delete[] host_b; 
	if (host_c)
		delete[] host_c; 
}

extern "C"
void ivk_krnl_vec_add(const float *a, const float *b, float *c, int n);

int main(int argc, char **argv)
{
	
	int N = 5000;
	size_t size = N * sizeof(float); 

	// Allocate host memory
	host_a = new float[N];
	host_b = new float[N]; 
	host_c = new float[N];
	
	// Init data
	random_init(host_a, N); 
	random_init(host_b, N);

	// Allocate device memory
	cudaMalloc((void**)&device_a, size);
	cudaMalloc((void**)&device_b, size); 
	cudaMalloc((void**)&device_c, size); 

	// Copy host data to device
	cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

	// Invoke kernel

	ivk_krnl_vec_add(device_a, device_b, device_c, N);
	
	// Copy the result back
	cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);
	
	// Verify the result 
	for (int i = 0; i < N; ++i)
	{
		std::cout << host_c[i] << std::endl;
	}

	return 0; 
	
}