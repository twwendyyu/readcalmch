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
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

typedef struct _MCHInfo {
	char	fname_mch[30];
	char	fname_inp[30];
	char	magicheader[4];
	unsigned int	version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton, seedbyte;
	unsigned int 	junk[5];
	float 	unitmm, normalizer;
	float 	na, n0, theta; 	// load from .inp
	float 	*mua;			// load from .inp
}MCHInfo;

typedef struct _MCHData {
	float *rawData;
	unsigned int *detid;
	float *weight;
	float *result;
}MCHData;

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
float *gpuRefPerPhoton(float **data2D, unsigned *size_2D)
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
	CUDA_CHECK_RETURN(cudaFree(gpuResult));

	return rc;
}
void initloadpara(MCHInfo *info, MCHData *data){

	FILE *fptr_mch, *fptr_inp;

	// set constant
	info->n0 = 1.457; 	printf("n0\t\t%f\n",info->n0);
	info->na = 0.22; 	printf("na\t\t%f\n",info->na);
	info->theta = asin(info->na/info->n0); printf("theta\t\t%f\n",info->theta);

	// specify .mch fname
	printf("Enter .mch file name:");
	scanf("%s",&(info->fname_mch));
	printf("Loading from %s ...\n",info->fname_mch);

	// load from fptr_mch
	fptr_mch = fopen(info->fname_mch,"rb");

	fread(info->magicheader,sizeof(char),4,fptr_mch);				printf("version\t\t%c%c%c%c\n",info->magicheader[0],info->magicheader[1],info->magicheader[2],info->magicheader[3]);
	fread(&(info->version),sizeof(unsigned int),1,fptr_mch);		printf("version\t\t%d\n",info->version);
	fread(&(info->maxmedia),sizeof(unsigned int),1,fptr_mch);		printf("mexmedia\t%d\n",info->maxmedia);
	fread(&(info->detnum),sizeof(unsigned int),1,fptr_mch);			printf("detnum\t\t%d\n",info->detnum);
	fread(&(info->colcount),sizeof(unsigned int),1,fptr_mch);		printf("colcount\t%d\n",info->colcount);
	fread(&(info->totalphoton),sizeof(unsigned int),1,fptr_mch);	printf("totalphoton\t%d\n",info->totalphoton);
	fread(&(info->detected),sizeof(unsigned int),1,fptr_mch);		printf("detected\t%d\n",info->detected);
	fread(&(info->savedphoton),sizeof(unsigned int),1,fptr_mch);	printf("savedphoton\t%d\n",info->savedphoton);
	fread(&(info->unitmm),sizeof(float),1,fptr_mch);				printf("unitmm\t\t%f\n",info->unitmm);
	fread(&(info->seedbyte),sizeof(unsigned int),1,fptr_mch);		printf("seedbyte\t%d\n",info->seedbyte);
	fread(&(info->normalizer),sizeof(float),1,fptr_mch);			printf("normalizer\t%f\n",info->normalizer);
	fread(info->junk,sizeof(unsigned int),5,fptr_mch);				printf("junk\t\t%d%d%d%d%d\n",info->junk[0],info->junk[1],info->junk[2],info->junk[3],info->junk[4]);

	//allocate memory
	unsigned int sizeOfData = info->savedphoton;
	data->detid = (unsigned int*) malloc (sizeof(unsigned int)*sizeOfData);
	data->weight = (float*) malloc (sizeof(float)*sizeOfData);

	unsigned int sizeOfResult = info->detnum;
	data->result = (float*) malloc (sizeof(float)*sizeOfResult);

	unsigned int sizeOfRawData = info->savedphoton*info->colcount;
	data->rawData = (float*) malloc (sizeof(float)*sizeOfRawData);
	fread(data->rawData ,sizeof(float), sizeOfRawData,fptr_mch); /* did not scaled back to 1 mm yet */


	// specify .inp fname
	printf("Enter .inp file name:");
	scanf("%s",&(info->fname_inp));
	printf("Loading from %s ...\n",info->fname_inp);

	// load from fptr_inp
	fptr_inp = fopen(info->fname_inp,"r");
	char junkc[50];
	for (int i = 0; i < 10; ++i){
		fgets(junkc, 50, fptr_inp); //discard from line 1 to 10
		printf("%s\n",junkc);
	}
	unsigned int sizeOfMua = info->maxmedia;
	double junkf1, junkf2, junkf3, junkf4;
	info->mua = (float*) malloc (sizeof(float)*sizeOfMua);
	for(int i = 0; i < sizeOfMua; ++i)
	{
		printf("line %d:",i);
		fscanf(fptr_inp,"%lf %lf %lf %lf",&(junkf1), &(junkf2), &(junkf3), &(junkf4));
		info->mua[i] = (float)junkf3; //casting double into float, and stored in mua[i]
		printf("\t%e\n",info->mua[i]);
	}

	// close
	fclose(fptr_mch);
	fclose(fptr_inp);

}
void calref_photon(MCHInfo *info,MCHData *data){

}

int main(void)
{
	MCHInfo info;
	MCHData data;

	initloadpara(&info,&data);
	calref_photon(&info,&data);	// void calref_photon(MCHInfo *info, MCHData *data);//also partition rawData into detid, weight arrays //__host__ calRefPerPhotonKernel
	//sortbykey(&info,&data);		// void sortbykey(MCHInfo *info, MCHData *data); 	//__host__ void thrust::sort_by_key
	//calref_det(&info,&data);		// void calref(MCHInfo *info, MCHData *data);		//__host__ thrust::pair<float*,float*> thrust::reduce_by_key
	//printresult(&info,&data);		// void printresult(MCHInfo *info, MCHData *data);

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

