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
#define BLOCK_SIZE 512


typedef struct _MCHInfo {
	char	fname_mch[30];
	char	fname_inp[30];
	char	magicheader[4];
	unsigned int	version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton, seedbyte;
	unsigned int 	junk[5];
	float 	unitmm, normalizer;
	float 	na, n0, theta; 	// load from .inp
	float 	*mua;			// load from .inp
	unsigned int sizeOfRawData, sizeOfData, sizeOfResult;
}MCHInfo;

typedef struct _MCHData {
	float *rawdata;			//array length: sizeOfRawData
	unsigned int *detid;	//array length:	sizeOfData
	float *weight;			//array length: sizeOfData
	float *result;			//array length: sizeOfResult
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
 * Host function that copies the data and launches the work on GPU
 */
float *gpuReciprocal(float *data, unsigned size)
{
	float *rc = new float[size];
	float *gpuData;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gpuData, sizeof(float)*size));
	CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, sizeof(float)*size, cudaMemcpyHostToDevice));
	
	const int blockCount = (size+BLOCK_SIZE-1)/BLOCK_SIZE;
	reciprocalKernel<<<blockCount, BLOCK_SIZE>>> (gpuData, size);

	CUDA_CHECK_RETURN(cudaMemcpy(rc, gpuData, sizeof(float)*size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpuData));
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
	info->sizeOfData = info->savedphoton;
	data->detid = (unsigned int*) malloc (sizeof(unsigned int)*info->sizeOfData);
	data->weight = (float*) malloc (sizeof(float)*info->sizeOfData);

	info->sizeOfResult = info->detnum;
	data->result = (float*) malloc (sizeof(float)*info->sizeOfResult);

	info->sizeOfRawData = info->savedphoton*info->colcount;
	data->rawdata = (float*) malloc (sizeof(float)*info->sizeOfRawData);
	fread(data->rawdata ,sizeof(float), info->sizeOfRawData,fptr_mch); /* did not scaled back to 1 mm yet */


	// specify .inp fname
	printf("Enter .inp file name:");
	scanf("%s",&(info->fname_inp));
	printf("Loading from %s ...\n",info->fname_inp);

	// load from fptr_inp
	fptr_inp = fopen(info->fname_inp,"r");
	char junkc[50];
	for (int i = 0; i < 10; ++i){
		fgets(junkc, 50, fptr_inp); //discard from line 1 to 10
		//printf("%s\n",junkc);
	}
	unsigned int sizeOfMua = info->maxmedia;
	double junkf1, junkf2, junkf3, junkf4;
	info->mua = (float*) malloc (sizeof(float)*sizeOfMua);
	for(int i = 0; i < sizeOfMua; ++i)
	{
		printf("mua %d:",i);
		fscanf(fptr_inp,"%lf %lf %lf %lf",&(junkf1), &(junkf2), &(junkf3), &(junkf4));
		info->mua[i] = (float)junkf3; //casting double into float, and stored in mua[i]
		printf("\t%e\n",info->mua[i]);
	}

	// close
	fclose(fptr_mch);
	fclose(fptr_inp);

}
/**
 * CUDA kernel that computes reflectance values for each photon
 */
__global__ void calRefPerPhotonKernel(unsigned int size, unsigned int colcount, unsigned int maxmedia, float *rawdata, float *detid, float *weight, float *mua, float unitmm, float theta) {

	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x; //i.e. rowcount

	if (idx < size){
		detid[idx] = rawdata[idx*colcount];
		weight[idx] = 0.0;

		float temp = 0.0;
		//if (acosf(abs(rawdata[(idx+1)*colcount-1])) <= theta){
			for (unsigned int i = 0; i < maxmedia; ++i)
				temp += (-1.0)*unitmm*mua[i]*rawdata[idx*colcount + (2+i)];
			weight[idx] = __expf(temp);
		//}
	}
}
void calref_photon(MCHInfo *info,MCHData *data){

	float *gRawdata, *gDetid, *gWeight, *gMua;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gRawdata, sizeof(float)*info->sizeOfRawData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gDetid, sizeof(float)*info->sizeOfData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gWeight, sizeof(float)*info->sizeOfData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gMua, sizeof(float)*info->maxmedia));

	CUDA_CHECK_RETURN(cudaMemcpy(gRawdata, data->rawdata, sizeof(float)*info->sizeOfRawData, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gMua, info->mua, sizeof(float)*info->maxmedia, cudaMemcpyHostToDevice));

	unsigned int blockCount = (info->sizeOfData + BLOCK_SIZE-1)/BLOCK_SIZE;
	calRefPerPhotonKernel<<<blockCount, BLOCK_SIZE>>> (info->sizeOfData, info->colcount, info->maxmedia, gRawdata, gDetid, gWeight, gMua, info->unitmm, info->theta);

	CUDA_CHECK_RETURN(cudaMemcpy(data->detid, gDetid, sizeof(float)*info->sizeOfData, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(data->weight, gWeight, sizeof(float)*info->sizeOfData, cudaMemcpyDeviceToHost));

	fprintf1DArray(data->weight, info->sizeOfData);

	CUDA_CHECK_RETURN(cudaFree(gRawdata));
	CUDA_CHECK_RETURN(cudaFree(gDetid));
	CUDA_CHECK_RETURN(cudaFree(gWeight));
	CUDA_CHECK_RETURN(cudaFree(gMua));

}

int main(void)
{
	MCHInfo info;
	MCHData data;

	initloadpara(&info,&data);
	calref_photon(&info,&data);	// void calref_photon(MCHInfo *info, MCHData *data);//also partition rawdata into detid, weight arrays //__host__ calRefPerPhotonKernel
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

