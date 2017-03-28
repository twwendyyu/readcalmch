/*
 ============================================================================
 Name        : readcalmch.cu
 Author      : Ting-Wen Yu
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

void initialize1DArray(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; ++i)
		data[i] = 0.0;
}

void initialize2DArray(float **data2D, unsigned *size_2D)
{
	unsigned height = size_2D[0];
	unsigned width = size_2D[1];

	for (unsigned i = 0; i < height; ++i){
		for (unsigned j = 0; j < width; ++j)
			data2D[i][j] = 0.0;
	}
}

void fprintf1DArray(float *data, unsigned size)
{
	FILE *fptr = fopen("1DArray.txt","w");
	for (unsigned i = 0; i < size; ++i)
		fprintf(fptr,"[%d]\t%f\n",i,data[i]);
	fclose(fptr);
}

void fprintf2DArray(float **data2D, unsigned *size_2D)
{
	unsigned height = size_2D[0];
	unsigned width = size_2D[1];

	FILE *fptr = fopen("2DArray.txt","w");
	for (unsigned i = 0; i < height; ++i){
		for (unsigned j = 0; j < width; ++j)
			fprintf(fptr,"[%d][%d]\t%f\n",i,j,data2D[i][j]);
	}
	fclose(fptr);
}

float *covert2Dto1DArray (float **data2D, unsigned *size_2D)
{
	unsigned height = size_2D[0];
	unsigned width = size_2D[1];

	float *data1D = new float[height*width];
	for (unsigned i = 0; i < height; ++i){
		for (unsigned j = 0; j < width; ++j)
			data1D[ i*width +j ] = data2D[i][j];
	}
	return data1D;
}

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void reciprocalKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = data[idx]+idx;
}

/**
 * CUDA kernel that conduct reduction with some conditions
 */
__global__ void ifReductionKernel(float *data, unsigned vectorSize) {
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	if (idx < vectorSize)
		data[idx] = data[idx]+idx;
}

/**
 * CUDA kernel that computes reflectance values for each photon
 */
__global__ void calRefPerPhotonKernel(float *data, float *result, unsigned height, unsigned width) {

	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned size = height*width;

	if (idx < size){
		result[idx] = data[idx]+idx;
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	static const int BLOCK_SIZE = 256;
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
	return rc;
}

/**
 * Host function that copies, reshape the data and launches the work on GPU
 */
float *gpuRefPerDet(float **data2D, unsigned *size_2D)
{
	unsigned height = size_2D[0];
	unsigned width  = size_2D[1];
	unsigned size = height * width;
	unsigned sizeOfgpuResult = size;

	float *data = covert2Dto1DArray (data2D, size_2D);
	fprintf1DArray(data, size);

	float *rc = new float[size];
	float *gpuData;
	float *gpuResult;

	/* gpuRefPerPhoton */
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuResult, sizeof(float)*sizeOfgpuResult));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));

	static const int BLOCK_SIZE = 512;
	int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	calRefPerPhotonKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, gpuResult, height, width);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuResult, sizeof(float)*sizeOfgpuResult, cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(gpuData));

	/* gpuRefPerDet */


	//blockCount = (sizeOfgpuResult+BLOCK_SIZE-1)/BLOCK_SIZE;
	//ifReductionKernel <<< blockCount, BLOCK_SIZE>>> ();

	//CUDA_CHECK_RETURN(cudaFree(gpuResult));

	return rc;
}

int main(void)
{
	/* 1D */
	int WORK_SIZE = 128;
	float *data = new float[WORK_SIZE];
	initialize1DArray (data, WORK_SIZE);

	float *recGpu = gpuReciprocal(data, WORK_SIZE);
	fprintf1DArray (recGpu, WORK_SIZE);

	/* 2D */
	unsigned *WORK_SIZE_2D = new unsigned[2];
	WORK_SIZE_2D[0] = 32; /* array height */
	WORK_SIZE_2D[1] = 4;   /* array width */

	float **data2D = new float*[WORK_SIZE_2D[0]];
	for(int i = 0; i < WORK_SIZE_2D[0]; ++i)
		data2D[i] = new float[WORK_SIZE_2D[1]];
	initialize2DArray (data2D, WORK_SIZE_2D);
	fprintf2DArray (data2D, WORK_SIZE_2D);

	float *refPerDet = gpuRefPerDet(data2D, WORK_SIZE_2D);



	/* Sum up in host */
	float gpuSum = std::accumulate (recGpu, recGpu+WORK_SIZE, 0.0);
	float result = std::accumulate (refPerDet, refPerDet+ WORK_SIZE_2D[0]*WORK_SIZE_2D[1], 0.0);

	/* Verify the results */
	std::cout<<"gpuSum = "<<gpuSum<<std::endl;
	std::cout<<"result = "<<result<<std::endl;

	/* Free memory */
	delete[] data;
	delete[] recGpu;
	for(int i = 0; i < WORK_SIZE_2D[0]; ++i)
		delete[] data2D[i];
	delete[] data2D;
	delete[] WORK_SIZE_2D;
	delete[] refPerDet;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

