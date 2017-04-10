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
#include <iomanip>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define BLOCK_SIZE 512
#define MAX_WORKLOAD 250000

using namespace std;

typedef struct _MCHInfo {
	int 	isprintinfo;
	char	fname[30];
	char	magicheader[4];
	unsigned int	version, maxmedia, detnum, colcount, totalphoton, detected, savedphoton, seedbyte;
	unsigned int 	junk[5];
	float 	unitmm, normalizer;
	float 	na, n0, theta; 	// load from .inp
	float 	*mua;			// load from .inp
}MCHInfo;

typedef struct _MCHData {
	unsigned sizeOfRawData, sizeOfData, sizeOfResult;
	float 	*rawdata;		//array length: sizeOfRawData
	int		*detid;			//array length:	sizeOfData
	float 	*weight;		//array length: sizeOfData
	float 	*result;		//array length: sizeOfResult
}MCHData;

template <typename T>
void arraymapping_1d(T *origin, T *copy, unsigned size){
	for (unsigned i = 0; i < size; ++i)
		copy[i] = origin[i];
}

template <typename T>
void fprintf1DArray(char fname[], T *data, unsigned size)
{
	ofstream myfile;

	myfile.open(fname);
	for (unsigned i = 0; i < size; ++i)
		myfile << fixed << setprecision(16) << data[i] << endl;

	myfile.close();
}

void initloadpara(int argc, char* argv[], MCHInfo *info, MCHData *data){

	FILE *fptr_mch, *fptr_inp;
	//default
	info->isprintinfo = 1;

	//load from argv
	int i = 0;
	while(i < argc){
		if(argv[i][0] == '-'){

			switch(argv[i][1]){
				case 'h':
					printf("p[1|0]\tPrint all the details of input arguments.\n");
					printf("n[float]\tRefraction index of outside medium.(must be entered).\n");
					printf("a[float]\tNumerical aperature (must be entered).\n");
					printf("f[string]\tFile name of .inp and .mch (must be entered).\n");
					exit(0);
				case 'p':
					info->isprintinfo = atoi(argv[i+1]);
					i++;
					break;
				case 'n':
					info->n0 = atof(argv[i+1]);
					i++;
					break;
				case 'a':
					info->na = atof(argv[i+1]);
					i++;
					break;
				case 'f':
					strcpy(info->fname,argv[i+1]);
					i++;
					break;
				default:
					printf("Did not assign from argv.\n Use '-h' to see the must options.\n");
					i++;
			}
		}
		i++;
	}

	// set constant
	info->theta = asin(info->na/info->n0);
	if (info->isprintinfo){
		printf("n0\t\t%f\n",info->n0);
		printf("na\t\t%f\n",info->na);
		printf("theta\t\t%f\n",info->theta);
	}

	// specify .mch fname
	char fname_mch[30];
	sprintf(fname_mch,"%s.mch",info->fname);
	if (info->isprintinfo) printf("Loading from %s ...\n",fname_mch);

	// load from fptr_mch
	fptr_mch = fopen(fname_mch,"rb");

	fread(info->magicheader,sizeof(char),4,fptr_mch);
	fread(&(info->version),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->maxmedia),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->detnum),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->colcount),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->totalphoton),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->detected),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->savedphoton),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->unitmm),sizeof(float),1,fptr_mch);
	fread(&(info->seedbyte),sizeof(unsigned int),1,fptr_mch);
	fread(&(info->normalizer),sizeof(float),1,fptr_mch);
	fread(info->junk,sizeof(unsigned int),5,fptr_mch);

	if (info->isprintinfo){
		printf("version\t\t%c%c%c%c\n",info->magicheader[0],info->magicheader[1],info->magicheader[2],info->magicheader[3]);
		printf("version\t\t%d\n",info->version);
		printf("mexmedia\t%d\n",info->maxmedia);
		printf("detnum\t\t%d\n",info->detnum);
		printf("colcount\t%d\n",info->colcount);
		printf("totalphoton\t%d\n",info->totalphoton);
		printf("detected\t%d\n",info->detected);
		printf("savedphoton\t%d\n",info->savedphoton);
		printf("unitmm\t\t%f\n",info->unitmm);
		printf("seedbyte\t%d\n",info->seedbyte);
		printf("normalizer\t%f\n",info->normalizer);
		printf("junk\t\t%d%d%d%d%d\n",info->junk[0],info->junk[1],info->junk[2],info->junk[3],info->junk[4]);
	}

	//allocate memory
	data->sizeOfData = info->savedphoton;
	data->detid = (int*) malloc (sizeof(int)*data->sizeOfData);
	data->weight = (float*) malloc (sizeof(float)*data->sizeOfData);

	data->sizeOfResult = info->detnum;
	data->result = (float*) malloc (sizeof(float)*data->sizeOfResult);

	data->sizeOfRawData = info->savedphoton*info->colcount;
	data->rawdata = (float*) malloc (sizeof(float)*data->sizeOfRawData);
	fread(data->rawdata ,sizeof(float), data->sizeOfRawData,fptr_mch); /* did not scaled back to 1 mm yet */


	// specify .inp fname
	char fname_inp[30];
	sprintf(fname_inp,"%s.inp",info->fname);
	if (info->isprintinfo) printf("Loading from %s ...\n",fname_inp);

	// load from fptr_inp
	fptr_inp = fopen(fname_inp,"r");
	char junkc[50];
	for (int i = 0; i < 10; ++i)
		fgets(junkc, 50, fptr_inp); //discard from line 1 to 10

	unsigned sizeOfMua = info->maxmedia;
	double junkf1, junkf2, junkf3, junkf4;
	info->mua = (float*) malloc (sizeof(float)*sizeOfMua);
	for(int i = 0; i < sizeOfMua; ++i)
	{
		if (info->isprintinfo) printf("mua %d:",i);
		fscanf(fptr_inp,"%lf %lf %lf %lf",&(junkf1), &(junkf2), &(junkf3), &(junkf4));
		info->mua[i] = (float)junkf3; //casting double into float, and stored in mua[i]
		if (info->isprintinfo) printf("\t%e\n",info->mua[i]);
	}

	// close
	fclose(fptr_mch);
	fclose(fptr_inp);

}
/**
 * CUDA kernel that computes reflectance values for each photon
 */
__global__ void calRefPerPhotonKernel(unsigned size, unsigned int colcount, unsigned int maxmedia, float *rawdata, int *detid, float *weight, float *mua, float unitmm, float theta) {

	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x; //i.e. rowcount

	if (idx < size){
		detid[idx] = (int)rawdata[idx*colcount];
		weight[idx] = 0.0;

		float temp = 0.0;
		if (acosf(abs(rawdata[(idx+1)*colcount-1])) <= theta){
			for (unsigned i = 0; i < maxmedia; ++i)
				temp += (-1.0)*unitmm*mua[i]*rawdata[idx*colcount + (2+i)];
			weight[idx] = __expf(temp);
		}
	}
}
void calref_photon(MCHInfo *info,MCHData *data){

	float *gRawdata, *gWeight, *gMua;
	int *gDetid;

	CUDA_CHECK_RETURN(cudaMalloc((void **)&gRawdata, sizeof(float)*data->sizeOfRawData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gDetid, sizeof(int)*data->sizeOfData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gWeight, sizeof(float)*data->sizeOfData));
	CUDA_CHECK_RETURN(cudaMalloc((void **)&gMua, sizeof(float)*info->maxmedia));

	CUDA_CHECK_RETURN(cudaMemcpy(gRawdata, data->rawdata, sizeof(float)*data->sizeOfRawData, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(gMua, info->mua, sizeof(float)*info->maxmedia, cudaMemcpyHostToDevice));

	unsigned int blockCount = (data->sizeOfData + BLOCK_SIZE-1)/BLOCK_SIZE;
	calRefPerPhotonKernel<<<blockCount, BLOCK_SIZE>>> (data->sizeOfData, info->colcount, info->maxmedia, gRawdata, gDetid, gWeight, gMua, info->unitmm, info->theta);

	CUDA_CHECK_RETURN(cudaMemcpy(data->detid, gDetid, sizeof(int)*data->sizeOfData, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(data->weight, gWeight, sizeof(float)*data->sizeOfData, cudaMemcpyDeviceToHost));


	CUDA_CHECK_RETURN(cudaFree(gRawdata));
	CUDA_CHECK_RETURN(cudaFree(gDetid));
	CUDA_CHECK_RETURN(cudaFree(gWeight));
	CUDA_CHECK_RETURN(cudaFree(gMua));

}

void sortbykey(MCHInfo *info, MCHData *data){

	//copy values from pointer to static array
	int keys[data->sizeOfData];
	arraymapping_1d<int>(data->detid, keys, data->sizeOfData);

	float values[data->sizeOfData];
	arraymapping_1d<float>(data->weight, values, data->sizeOfData);

	//call main function
	const int N = data->sizeOfData;
	thrust::sort_by_key(thrust::host, keys, keys + N, values);

	//copy values from static array to pointer
	arraymapping_1d<int >(keys, data->detid, data->sizeOfData);
	arraymapping_1d<float>(values, data->weight, data->sizeOfData);


}
void calref_det(MCHInfo *info, MCHData *data){

	//copy values from pointer to static array
	int keysIn[data->sizeOfData];
	arraymapping_1d<int>(data->detid, keysIn, data->sizeOfData);

	float valuesIn[data->sizeOfData];
	arraymapping_1d<float>(data->weight, valuesIn, data->sizeOfData);

	int keysOut[data->sizeOfResult];

	float valuesOut[data->sizeOfResult];

	//call main function
	const int N = data->sizeOfData;
	thrust::reduce_by_key(thrust::host, keysIn, keysIn + N, valuesIn, keysOut, valuesOut);

	//copy values from static array to pointer
	arraymapping_1d<float>(valuesOut, data->result, data->sizeOfResult);

}
void printresult(MCHInfo *info, MCHData *data){

	// print result./totalphoton
	double temp[data->sizeOfResult];
	for (unsigned i = 0; i < data->sizeOfResult; ++i)
		temp[i] = data->result[i]/info->totalphoton;

	char fname[30];
	sprintf(fname,"%s.txt",info->fname);
	fprintf1DArray(fname, temp, data->sizeOfResult);
	if (info->isprintinfo) printf("Print to %s ...\n",fname);
}
void clearmch(MCHInfo *info, MCHData *data){
	if(info->mua){
		free(info->mua);
		info->mua = NULL;
	}
	if(data->rawdata){
		free(data->rawdata);
		data->rawdata = NULL;
	}
	if(data->detid){
		free(data->detid);
		data->detid = NULL;
	}
	if(data->weight){
		free(data->weight);
		data->weight = NULL;
	}
	if(data->result){
		free(data->result);
		data->result = NULL;
	}
}

void segmentdata(int batchid, int batchnum, MCHData *data_batch, MCHInfo *info, MCHData *data){
	//copy segmented data from data.rawdata to data_batch.rawdata, and change the corresponding info

	int rownum;

	if (batchid == batchnum-1)
		rownum = info->savedphoton - (batchnum-1)*MAX_WORKLOAD;
	else
		rownum = MAX_WORKLOAD;

	//rawdata
	data_batch->sizeOfRawData = rownum*info->colcount;
	unsigned startid = (unsigned)batchid*MAX_WORKLOAD*info->colcount;
	data_batch->rawdata = &(data->rawdata[startid]);

	//data
	data_batch->sizeOfData = rownum;
	data_batch->detid = (int*) malloc (sizeof(int)*data_batch->sizeOfData);
	data_batch->weight = (float*) malloc (sizeof(float)*data_batch->sizeOfData);

	//result
	data_batch->sizeOfResult = data->sizeOfResult;
	data_batch->result = (float*) malloc (sizeof(float)*data_batch->sizeOfResult);

}

void gatherbatchdata(MCHData *data_batch, MCHData *data){
	//add data_batch.result to data.result
	for (int i = 0; i < data_batch->sizeOfResult; ++i)
		data->result[i] += data_batch->result[i];

}

void clearbatch(MCHData *data){
	/*if(data->rawdata){
			free(data->rawdata);
			data->rawdata = NULL;
		}*/
		if(data->detid){
			free(data->detid);
			data->detid = NULL;
		}
		if(data->weight){
			free(data->weight);
			data->weight = NULL;
		}
		if(data->result){
			free(data->result);
			data->result = NULL;
		}
}

int main(int argc, char *argv[])
{
	MCHInfo info;
	MCHData data;

	initloadpara(argc, argv, &info, &data);

	if(info.savedphoton > MAX_WORKLOAD){

		MCHData data_batch;//intialize in segmentdata

		int batchnum = (info.savedphoton/MAX_WORKLOAD) +1;
		int batchid = 0;

		while (batchid < batchnum){

			segmentdata(batchid,batchnum,&data_batch,&info,&data);
			calref_photon(&info, &data_batch);
			sortbykey(&info, &data_batch);
			calref_det(&info, &data_batch);
			gatherbatchdata(&data_batch,&data);
			clearbatch(&data_batch);
			batchid++;
		}

	}else{
		calref_photon(&info, &data);
		sortbykey(&info, &data);
		calref_det(&info, &data);
	}

	printresult(&info, &data);
	clearmch(&info, &data);

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

