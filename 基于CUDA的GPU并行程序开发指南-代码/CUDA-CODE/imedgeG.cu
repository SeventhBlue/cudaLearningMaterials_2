#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"


#define	CEIL(a,b)				((a+b-1)/b)
#define PI						3.1415926
#define EDGE					0
#define NOEDGE					255
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))


typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

uch *TheImg, *CopyImg;				// Where images are stored in CPU
int	ThreshLo=50, ThreshHi=100;		// "Edge" vs. "No Edge" thresholds

// Where images and temporary results are stored in GPU
uch		*GPUImg,   *GPUResultImg;
double  *GPUBWImg, *GPUGaussImg, *GPUGradient, *GPUTheta;	


struct ImgProp{
	ui Hpixels;
	ui Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;

#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)



// Kernel that calculates a B&W image from an RGB image
// resulting image has a double type for each pixel position
__global__
void BWKernel(double *ImgBW, uch *ImgGPU, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	double R, G, B;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYsrcIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;

	B = (double)ImgGPU[MYsrcIndex];
	G = (double)ImgGPU[MYsrcIndex + 1];
	R = (double)ImgGPU[MYsrcIndex + 2];
	ImgBW[MYpixIndex] = (R+G+B)/3.0;
}


__device__
double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 } };
// Kernel that calculates a Gauss image from the B&W image
// resulting image has a double type for each pixel position
__global__
void GaussKernel(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	int row, col, indx, i, j;
	double G=0.00;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	int MYrow = MYbid / BlkPerRow;
	int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)){
		ImgGauss[MYpixIndex] = 0.0;
		return;
	}else{
		G = 0.0;
		for (i = -2; i <= 2; i++){
			for (j = -2; j <= 2; j++){
				row = MYrow + i;
				col = MYcol + j;
				indx = row*Hpixels + col;
				G += (ImgBW[indx] * Gauss[i + 2][j + 2]);
			}
		}
		ImgGauss[MYpixIndex] = G / 159.00;
	}
}


__device__
double Gx[3][3] = { { -1, 0, 1 },
					{ -2, 0, 2 },
					{ -1, 0, 1 } };
__device__
double Gy[3][3] = { { -1, -2, -1 },
					{ 0, 0, 0 },
					{ 1, 2, 1 } };
// Kernel that calculates Gradient, Theta from the Gauss image
// resulting image has a double type for each pixel position
__global__
void SobelKernel(double *ImgGrad, double *ImgTheta, double *ImgGauss, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	int row, col, indx, i, j;
	double GX,GY;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	int MYrow = MYbid / BlkPerRow;
	int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<1) || (MYrow>Vpixels - 2) || (MYcol<1) || (MYcol>Hpixels - 2)){
		ImgGrad[MYpixIndex] = 0.0;
		ImgTheta[MYpixIndex] = 0.0;
		return;
	}else{
		GX = 0.0;   GY = 0.0;
		for (i = -1; i <= 1; i++){
			for (j = -1; j <= 1; j++){
				row = MYrow + i;
				col = MYcol + j;
				indx = row*Hpixels + col;
				GX += (ImgGauss[indx] * Gx[i + 1][j + 1]);
				GY += (ImgGauss[indx] * Gy[i + 1][j + 1]);
			}
		}
		ImgGrad[MYpixIndex] = sqrt(GX*GX + GY*GY);
		ImgTheta[MYpixIndex] = atan(GX / GY)*180.0 / PI;
	}
}


// Kernel that calculates the threshold image from Gradient, Theta
// resulting image has an RGB for each pixel, same RGB for each pixel
__global__
void ThresholdKernel(uch *ImgResult, double *ImgGrad, double *ImgTheta, ui Hpixels, ui Vpixels, ui ThreshLo, ui ThreshHi)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	unsigned char PIXVAL;
	double L, H, G, T;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	int MYrow = MYbid / BlkPerRow;
	int MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYresultIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<1) || (MYrow>Vpixels - 2) || (MYcol<1) || (MYcol>Hpixels - 2)){
		ImgResult[MYresultIndex] = NOEDGE;
		ImgResult[MYresultIndex + 1] = NOEDGE;
		ImgResult[MYresultIndex + 2] = NOEDGE;
		return;
	}else{
		L = (double)ThreshLo;		H = (double)ThreshHi;
		G = ImgGrad[MYpixIndex];
		PIXVAL = NOEDGE;
		if (G <= L){						// no edge
			PIXVAL = NOEDGE;
		}else if (G >= H){					// edge
			PIXVAL = EDGE;
		}else{
			T = ImgTheta[MYpixIndex];
			if ((T<-67.5) || (T>67.5)){
				// Look at left and right: [row][col-1]  and  [row][col+1]
				PIXVAL = ((ImgGrad[MYpixIndex - 1]>H) || (ImgGrad[MYpixIndex + 1]>H)) ? EDGE : NOEDGE;
			}
			else if ((T >= -22.5) && (T <= 22.5)){
				// Look at top and bottom: [row-1][col]  and  [row+1][col]
				PIXVAL = ((ImgGrad[MYpixIndex - Hpixels]>H) || (ImgGrad[MYpixIndex + Hpixels]>H)) ? EDGE : NOEDGE;
			}
			else if ((T>22.5) && (T <= 67.5)){
				// Look at upper right, lower left: [row-1][col+1]  and  [row+1][col-1]
				PIXVAL = ((ImgGrad[MYpixIndex - Hpixels + 1]>H) || (ImgGrad[MYpixIndex + Hpixels - 1]>H)) ? EDGE : NOEDGE;
			}
			else if ((T >= -67.5) && (T<-22.5)){
				// Look at upper left, lower right: [row-1][col-1]  and  [row+1][col+1]
				PIXVAL = ((ImgGrad[MYpixIndex - Hpixels - 1]>H) || (ImgGrad[MYpixIndex + Hpixels + 1]>H)) ? EDGE : NOEDGE;
			}
		}
		ImgResult[MYresultIndex] = PIXVAL;
		ImgResult[MYresultIndex + 1] = PIXVAL;
		ImgResult[MYresultIndex + 2] = PIXVAL;
	}
}


/*
// helper function that wraps CUDA API calls, reports any error and exits
void chkCUDAErr(cudaError_t error_id)
{
	if (error_id != CUDA_SUCCESS){
		printf("CUDA ERROR :::%\n", cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}
*/


// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin(char* fn)
{
	static uch *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL){	printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }

	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		ip.Hbytes = RowBytes;
	//save header for re-use
	memcpy(ip.HeaderInfo, HeaderInfo,54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}


// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uch *Img, char* fn)
{
	FILE* f = fopen(fn, "wb");
	if (f == NULL){ printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	//write header
	fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
	//write data
	fwrite(Img, sizeof(uch), IMAGESIZE, f);
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%u", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}


int main(int argc, char **argv)
{
	// clock_t CPUStartTime, CPUEndTime, CPUElapsedTime;
	// GPU code run times
	float totalTime, totalKernelTime, tfrCPUtoGPU, tfrGPUtoCPU;
	float kernelExecTimeBW, kernelExecTimeGauss, kernelExecTimeSobel, kernelExecTimeThreshold;
	cudaError_t cudaStatus;
	cudaEvent_t time1, time2, time2BW, time2Gauss, time2Sobel, time3, time4;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks;
	ui GPUDataTfrBW, GPUDataTfrGauss, GPUDataTfrSobel, GPUDataTfrThresh,GPUDataTfrKernel, GPUDataTfrTotal;
	cudaDeviceProp GPUprop;
	void *GPUptr;			// Pointer to the bulk-allocated GPU memory
	ul GPUtotalBufferSize;
	ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];

	strcpy(ProgName, "imedgeG");
	switch (argc){
		case 6:  ThreshHi  = atoi(argv[5]);
		case 5:  ThreshLo  = atoi(argv[4]);
		case 4:  ThrPerBlk = atoi(argv[3]);
		case 3:  strcpy(InputFileName, argv[1]);
				 strcpy(OutputFileName, argv[2]);
				 break;
		default: printf("\n\nUsage:   %s InputFilename OutputFilename [ThrPerBlk] [ThreshLo] [ThreshHi]", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp 256", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp 256 50 100",ProgName);
				 exit(EXIT_FAILURE);
	}
	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
		exit(EXIT_FAILURE);
	}
	if ((ThreshLo<0) || (ThreshHi>255) || (ThreshLo>ThreshHi)){
		printf("\nInvalid Thresholds: Threshold must be between [0...255] ...\n");
		printf("\n\nNothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
	}
	// Create CPU memory to store the input and output images
	TheImg = ReadBMPlin(InputFileName); // Read the input image if memory can be allocated
	if (TheImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	CopyImg = (uch *)malloc(IMAGESIZE);
	if (CopyImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		free(TheImg);
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		goto EXITERROR;
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto EXITERROR;
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui) GPUprop.maxGridSize[0] * (ui) GPUprop.maxGridSize[1] * (ui )GPUprop.maxGridSize[2]/1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks>=5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks>=5) ?  'M':'K');
	MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	cudaEventCreate(&time1);		cudaEventCreate(&time2);	
	cudaEventCreate(&time2BW);		cudaEventCreate(&time2Gauss);	cudaEventCreate(&time2Sobel);	
	cudaEventCreate(&time3);		cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer
	// Allocate GPU buffer for the input and output images and the imtermediate results
	GPUtotalBufferSize = 4 * sizeof(double)*IMAGEPIX + 2 * sizeof(uch)*IMAGESIZE;
	cudaStatus = cudaMalloc((void**)&GPUptr, GPUtotalBufferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		goto EXITERROR;
	}
	GPUImg			= (uch *)GPUptr;
	GPUResultImg	= GPUImg + IMAGESIZE;
	GPUBWImg		= (double *)(GPUResultImg + IMAGESIZE);
	GPUGaussImg		= GPUBWImg + IMAGEPIX;
	GPUGradient		= GPUGaussImg + IMAGEPIX;
	GPUTheta		= GPUGradient + IMAGEPIX;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		goto EXITCUDAERROR;
	}
	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	//dim3 dimBlock(ThrPerBlk);
	//dim3 dimGrid(ip.Hpixels*BlkPerRow);
	BlkPerRow = CEIL(ip.Hpixels, ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUBWImg, GPUImg, ip.Hpixels);
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR;
	cudaEventRecord(time2BW, 0);		// Time stamp after BW image calculation
	GPUDataTfrBW = sizeof(double)*IMAGEPIX + sizeof(uch)*IMAGESIZE;

	GaussKernel <<< NumBlocks, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, ip.Hpixels, ip.Vpixels);
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR; 
	cudaEventRecord(time2Gauss, 0);		// Time stamp after Gauss image calculation
	GPUDataTfrGauss = 2*sizeof(double)*IMAGEPIX;

	SobelKernel <<< NumBlocks, ThrPerBlk >>> (GPUGradient, GPUTheta, GPUGaussImg, ip.Hpixels, ip.Vpixels);
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR; 
	cudaEventRecord(time2Sobel, 0);		// Time stamp after Gradient, Theta computation
	GPUDataTfrSobel = 3 * sizeof(double)*IMAGEPIX;

	ThresholdKernel <<< NumBlocks, ThrPerBlk >>> (GPUResultImg, GPUGradient, GPUTheta, ip.Hpixels, ip.Vpixels, ThreshLo, ThreshHi);
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR;
	GPUDataTfrThresh = sizeof(double)*IMAGEPIX + sizeof(uch)*IMAGESIZE;
	GPUDataTfrKernel = GPUDataTfrBW + GPUDataTfrGauss + GPUDataTfrSobel + GPUDataTfrThresh;
	GPUDataTfrTotal = GPUDataTfrKernel + 2 * IMAGESIZE;
	cudaEventRecord(time3, 0);

	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CopyImg, GPUResultImg, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		goto EXITCUDAERROR;
	}
	cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1);	cudaEventSynchronize(time2);
	cudaEventSynchronize(time2BW);	cudaEventSynchronize(time2Gauss);	cudaEventSynchronize(time2Sobel);
	cudaEventSynchronize(time3);	cudaEventSynchronize(time4);

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecTimeBW, time2, time2BW);
	cudaEventElapsedTime(&kernelExecTimeGauss, time2BW, time2Gauss);
	cudaEventElapsedTime(&kernelExecTimeSobel, time2Gauss, time2Sobel);
	cudaEventElapsedTime(&kernelExecTimeThreshold, time2Sobel, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);
	totalKernelTime = kernelExecTimeBW + kernelExecTimeGauss + kernelExecTimeSobel + kernelExecTimeThreshold;

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk
	printf("\n\n----------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n",
		GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("----------------------------------------------------------------------------\n");
	printf("%s %s %s %u %d %d  [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, ThrPerBlk, ThreshLo, ThreshHi, NumBlocks, BlkPerRow);
	printf("----------------------------------------------------------------------------\n");
	printf("              CPU->GPU Transfer =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE,tfrCPUtoGPU));
	printf("              GPU->CPU Transfer =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU));
	printf("----------------------------------------------------------------------------\n");
	printf("       BW Kernel Execution Time =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecTimeBW, DATAMB(GPUDataTfrBW), DATABW(GPUDataTfrBW, kernelExecTimeBW));
	printf("    Gauss Kernel Execution Time =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecTimeGauss, DATAMB(GPUDataTfrGauss), DATABW(GPUDataTfrGauss, kernelExecTimeGauss));
	printf("    Sobel Kernel Execution Time =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecTimeSobel, DATAMB(GPUDataTfrSobel), DATABW(GPUDataTfrSobel, kernelExecTimeSobel));
	printf("Threshold Kernel Execution Time =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecTimeThreshold, DATAMB(GPUDataTfrThresh), DATABW(GPUDataTfrThresh, kernelExecTimeThreshold));
	printf("----------------------------------------------------------------------------\n");
	printf("         Total Kernel-only time =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", totalKernelTime, DATAMB(GPUDataTfrKernel), DATABW(GPUDataTfrKernel, totalKernelTime));
	printf("   Total time with I/O included =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB(GPUDataTfrTotal), DATABW(GPUDataTfrTotal, totalTime));
	printf("----------------------------------------------------------------------------\n");

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUptr);
	cudaEventDestroy(time1);	cudaEventDestroy(time2);
	cudaEventDestroy(time2BW);	cudaEventDestroy(time2Gauss);	cudaEventDestroy(time2Sobel);
	cudaEventDestroy(time3);	cudaEventDestroy(time4);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	free(TheImg);
	free(CopyImg);
	return(EXIT_SUCCESS);
KERNELERROR:
	fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
EXITCUDAERROR:
	cudaFree(GPUptr);
EXITERROR:
	free(TheImg);
	free(CopyImg);
	return(EXIT_FAILURE);

}



