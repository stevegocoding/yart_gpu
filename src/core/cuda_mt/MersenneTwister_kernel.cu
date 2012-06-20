/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 *
 * MNeumann (April 2010): Removed shrUtil dependency and added external declarations
 *						  to enable usage for MNRT.
 * 
 */

//#include <shrUtils.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_utils.h"
#include "MersenneTwister.h"
//#include "MNCudaUtil.h"



__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];

////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of NPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void RandomGPU(
    float *d_Random,
    int NPerRng
){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN];

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        //Initialize current state
        mt[0] = config.seed;
        for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

        iState = 0;
        mti1 = mt[0];
        for(iOut = 0; iOut < NPerRng; iOut++){
            //iState1 = (iState +     1) % MT_NN
            //iStateM = (iState + MT_MM) % MT_NN
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN) iState1 -= MT_NN;
            if(iStateM >= MT_NN) iStateM -= MT_NN;
            mti  = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
            mt[iState] = x;
            iState = iState1;

            //Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & config.mask_b;
            x ^= (x << MT_SHIFTC) & config.mask_c;
            x ^= (x >> MT_SHIFT1);

            //Convert to (0, 1] float and write to global memory
            d_Random[iRng + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979f
__device__ inline void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int NPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N)
        for(int iOut = 0; iOut < NPerRng; iOut += 2)
            BoxMuller(
                d_Random[iRng + (iOut + 0) * MT_RNG_COUNT],
                d_Random[iRng + (iOut + 1) * MT_RNG_COUNT]
            );
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" bool MersenneTwisterGPUInit(const char *fname)
///
/// \brief	Loads Mersenne Twister configuration from given source file. 
///
/// \author	Mathias Neumann
/// \date	11.04.2010
///
/// \param	fname	Filename of the configuration file. 
///
/// \return	true if it succeeds, false if it fails. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
bool MersenneTwisterGPUInit(const char *fname)
{
    FILE *fd = fopen(fname, "rb");
    if(!fd)
	{
		//MNFatal("Failed to open %s for Mersenne Twister configuration.", fname);
		return false;
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) )
	{
		//MNFatal("Failed to load %s for Mersenne Twister configuration.", fname);
		return false;
    }
    fclose(fd);

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void MersenneTwisterGPUSeed(unsigned int seed)
///
/// \brief	Seeds Mersenne Twister for current GPU context. 
///
/// \author	Mathias Neumann
/// \date	11.04.2010
///
/// \param	seed	The seed. 
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void MersenneTwisterGPUSeed(unsigned int seed)
{
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
	cudaError_t err = cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));
	assert(err == cudaSuccess);

    free(MT);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// \fn	extern "C" void MersenneTwisterGPU(float* d_outRand, int nPerRNG)
///
/// \brief	Performs Mersenne Twister RNG to generate a predefined number of uniform random
/// 		numbers to use in other kernels. 
///
/// \author	Mathias Neumann
/// \date	11.04.2010
///
/// \param [out]	d_outRand	The generated uniform random numbers. 
/// \param	nPerRNG				The random numbers per generator. Will generate 
///								nPerRNG * MT_RNG_COUNT numbers.
////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C"
void MersenneTwisterGPU(float* d_outRand, int nPerRNG)
{
	// 32 * 128 = MT_RNG_COUNT = 4096. See SDK 3.0 sample.
	RandomGPU<<<32, 128>>>(d_outRand, nPerRNG);
	// MNCUDA_CHECKERROR;
	CUDA_CHECK_ERROR;

	//BoxMullerGPU<<<32, 128>>>(d_outRand, nPerRNG);
}