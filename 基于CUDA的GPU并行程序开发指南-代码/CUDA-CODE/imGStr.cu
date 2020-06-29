#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"


#define	CEIL(a,b)		((a+b-1)/b)
#define SWAP(a,b,t)		t=b; b=a; a=t;
#define PI				3.1415926
#define EDGE			0
#define NOEDGE			255
#define DATAKB(bytes)			(bytes/1024)
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))
#define MAXSTREAMS		32

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



__global__
void Hflip3S(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui RowBytes, ui StartRow)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = StartRow + blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

	ui MYmirrorcol = Hpixels - 1 - MYcol;
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	ui MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


__global__
void BWKernel2S(double *ImgBW, uch *ImgGPU, ui Hpixels, ui Vpixels, ui RowBytes, ui StartRow)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui R, G, B;

	ui MYrow = StartRow + blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

	ui MYsrcIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;

	B = (ui)ImgGPU[MYsrcIndex];
	G = (ui)ImgGPU[MYsrcIndex + 1];
	R = (ui)ImgGPU[MYsrcIndex + 2];
	ImgBW[MYpixIndex] = (double)(R + G + B) * 0.333333;
}


__constant__
double GaussC[5][5] = { { 2, 4, 5, 4, 2 },
						{ 4, 9, 12, 9, 4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9, 12, 9, 4 },
						{ 2, 4, 5, 4, 2 } };



// Processes multiple rows (as many as blockIdx.y, starting at MYrow)
__global__
void GaussKernel3S(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels, ui StartRow)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int row, col, indx, i, j;
	double G;

	ui MYrow = StartRow+blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)) {
		ImgGauss[MYpixIndex] = 0.0;
		return;
	}else{
		G = 0.0;
		for (i = -2; i <= 2; i++) {
			for (j = -2; j <= 2; j++) {
				row = MYrow + i;
				col = MYcol + j;
				indx = row*Hpixels + col;
				G += (ImgBW[indx] * GaussC[i + 2][j + 2]);  // use constant memory
			}
		}
		ImgGauss[MYpixIndex] = G * 0.0062893;   // (1/159)=0.0062893
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


__global__
void SobelKernel2S(double *ImgGrad, double *ImgTheta, double *ImgGauss, ui Hpixels, ui Vpixels, ui StartRow)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int indx;
	double GX,GY;

	ui MYrow = StartRow + blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<1) || (MYrow>Vpixels - 2) || (MYcol<1) || (MYcol>Hpixels - 2)){
		ImgGrad[MYpixIndex] = 0.0;
		ImgTheta[MYpixIndex] = 0.0;
		return;
	}else{
		indx=(MYrow-1)*Hpixels + MYcol-1;
		GX = (-ImgGauss[indx-1]+ImgGauss[indx+1]);
		GY = (-ImgGauss[indx-1]-2*ImgGauss[indx]-ImgGauss[indx+1]);
		
		indx+=Hpixels;
		GX += (-2*ImgGauss[indx-1]+2*ImgGauss[indx+1]);
		
		indx+=Hpixels;
		GX += (-ImgGauss[indx-1]+ImgGauss[indx+1]);
		GY += (ImgGauss[indx-1]+2*ImgGauss[indx]+ImgGauss[indx+1]);
		ImgGrad[MYpixIndex] = sqrt(GX*GX + GY*GY);
		ImgTheta[MYpixIndex] = atan(GX / GY)*57.2957795;      // 180.0/PI = 57.2957795;
	}
}


// Kernel that calculates the threshold image from Gradient, Theta
// resulting image has an RGB for each pixel, same RGB for each pixel
__global__
void ThresholdKernel2S(uch *ImgResult, double *ImgGrad, double *ImgTheta, ui Hpixels, ui Vpixels, ui RowBytes, ui ThreshLo, ui ThreshHi, ui StartRow)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	
	unsigned char PIXVAL;
	double L, H, G, T;

	ui MYrow = StartRow + blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

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



// helper function that wraps CUDA API calls, reports any error and exits
void chkCUDAErr(cudaError_t error_id)
{
	if (error_id != CUDA_SUCCESS){
		printf("CUDA ERROR :::%s\n", cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}
}

// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin(char* fn)
{
	static uch *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL) { printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }

	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
										   // extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		ip.Hbytes = RowBytes;
	//save header for re-use
	memcpy(ip.HeaderInfo, HeaderInfo, 54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn,
		ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
									  // read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}



// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate PINNED memory to store the 1D image and return its pointer.
uch *ReadBMPlinPINNED(char* fn)
{
	static uch *Img;
	void *p;
	cudaError_t  AllocErr;
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
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, IPH, IPV, IMAGESIZE);
	// allocate PINNED memory to store the main 1D image
	//Img  = (uch *)malloc(IMAGESIZE);
	AllocErr=cudaMallocHost((void**)&p, IMAGESIZE);
	if (AllocErr == cudaErrorMemoryAllocation){
		Img=NULL;      // Cannot allocate memory
		return Img;
	}else{
		Img=(uch *)p;
	}

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


// Print a separator between messages
void PrintSep()
{
	printf("-----------------------------------------------------------------------------------------\n");
}

int main(int argc, char **argv)
{
	char			Operation = 'E';
	float			totalTime, Time12, Time23, Time34;	// GPU code run times
	cudaError_t		cudaStatus;
	cudaEvent_t		time1, time2, time3, time4;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow, ThrPerBlk=256;
	cudaDeviceProp	GPUprop;
	void			*GPUptr;			// Pointer to the bulk-allocated GPU memory
	ul				GPUtotalBufferSize;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];
	int				deviceOverlap, SMcount;
	ul				ConstMem, GlobalMem;
	ui				NumberOfStreams=1,RowsPerStream;
    cudaStream_t	stream[MAXSTREAMS];
	void			*p;					// temporary pointer for the pinned memory
	ui				i;

	strcpy(ProgName, "imGStr");
	switch (argc){
		case 6:  NumberOfStreams = atoi(argv[5]);
		case 5:  ThrPerBlk = atoi(argv[4]);
		case 4:  Operation = toupper(argv[3][0]);
		case 3:  strcpy(InputFileName, argv[1]);
				 strcpy(OutputFileName, argv[2]);
				 break;
		default: printf("\n\nUsage:   %s InputFilename OutputFilename [H/E] [ThrPerBlk] [NumberOfStreams:0-32]", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp E 256 3   --- which means 3 streams", ProgName);
				 printf("\n\n   (Note: 0 means Synchronous (no streaming) ... H=Hor. flip , E=Edge detection\n");
				 exit(EXIT_FAILURE);
	}
	
	// Operation is 'H' for Horizontal flip and 'E' for Edge Detection
	if ((Operation != 'E') && (Operation != 'H')) {
		printf("Invalid operation '%c'. Must be 'H', or 'E' ... \n", Operation);
		exit(EXIT_FAILURE);
	}

	// Parse the "Threads per block" parameter
	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
		exit(EXIT_FAILURE);
	}

	// Determine the number of streams
	if (NumberOfStreams > 32) {
		printf("Invalid NumberOfStreams option (%u). Must be between 0 and 32. \n", NumberOfStreams);
		printf("0 means NO STREAMING (i.e., synchronous)\n");
		exit(EXIT_FAILURE);
	}

	if (NumberOfStreams == 0) {
		// 0 means, synchronous. No streams.
		//printf("NumberOfStreams=0 ... Executing in non-streaming (synchronous) mode.\n");
		TheImg = ReadBMPlin(InputFileName);		// Read the input image into a regular memory
		if (TheImg == NULL) {
			printf("Cannot allocate memory for the input image...\n");
			exit(EXIT_FAILURE);
		}
		CopyImg = (uch *)malloc(IMAGESIZE);
		if (CopyImg == NULL) {
			printf("Cannot allocate memory for the input image...\n");
			free(TheImg);
			exit(EXIT_FAILURE);
		}
	}else{
		// Create CPU memory to store the input and output images
		TheImg = ReadBMPlinPINNED(InputFileName);		// Read the input image into a PINNED memory
		if (TheImg == NULL){
			printf("Cannot allocate PINNED memory for the input image...\n");
			exit(EXIT_FAILURE);
		}
		// Allocate pinned memory for the CopyImg
		cudaStatus=cudaMallocHost((void**)&p, IMAGESIZE);
		if (cudaStatus == cudaErrorMemoryAllocation){
			printf("Cannot allocate PINNED memory for the CopyImg ...\n");
			cudaFreeHost(TheImg);
			exit(EXIT_FAILURE);
		}else{
			CopyImg=(uch *)p;
		}
	}


	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		//cudaFreeHost(TheImg);
		//cudaFreeHost(CopyImg);
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		//cudaFreeHost(TheImg);
		//cudaFreeHost(CopyImg);
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui) GPUprop.maxGridSize[0] * (ui) GPUprop.maxGridSize[1] * (ui )GPUprop.maxGridSize[2]/1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks>=5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks>=5) ?  'M':'K');
	MaxThrPerBlk  = (ui)GPUprop.maxThreadsPerBlock;
	deviceOverlap = GPUprop.deviceOverlap;   // Shows whether the device can transfer in both directions simultaneously
	SMcount       = GPUprop.multiProcessorCount;
	ConstMem      = (ul) GPUprop.totalConstMem;
	GlobalMem     = (ul) GPUprop.totalGlobalMem;
	

	// CREATE EVENTS
	cudaEventCreate(&time1);		
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	// CREATE STREAMS  
	if(NumberOfStreams != 0){
		for (i = 0; i < NumberOfStreams; i++) {
			chkCUDAErr(cudaStreamCreate(&stream[i]));
		}
	}
	//printf("%u streams created\n",NumberOfStreams);
	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer
	// Allocate GPU buffer for the input and output images and the imtermediate results
	GPUtotalBufferSize = 4 * sizeof(double)*IMAGEPIX + 2 * sizeof(uch)*IMAGESIZE;
	cudaStatus = cudaMalloc((void**)&GPUptr, GPUtotalBufferSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory\n");
		//cudaFreeHost(TheImg);
		//cudaFreeHost(CopyImg);
		exit(EXIT_FAILURE);	
	}
	GPUImg			= (uch *)GPUptr;
	GPUResultImg	= GPUImg + IMAGESIZE;
	GPUBWImg		= (double *)(GPUResultImg + IMAGESIZE);
	GPUGaussImg		= GPUBWImg + IMAGEPIX;
	GPUGradient		= GPUGaussImg + IMAGEPIX;
	GPUTheta		= GPUGradient + IMAGEPIX;


	BlkPerRow = CEIL(IPH, ThrPerBlk);
	RowsPerStream = ((NumberOfStreams == 0) ? IPV : CEIL(IPV, NumberOfStreams));
	dim3 dimGrid2D(BlkPerRow, IPV);						// to process the entire stream
	dim3 dimGrid2DS(BlkPerRow, RowsPerStream);			// to process the rows of one stream
	dim3 dimGrid2DS1(BlkPerRow, 1);
	dim3 dimGrid2DS2(BlkPerRow, 2);
	dim3 dimGrid2DS4(BlkPerRow, 4);
	dim3 dimGrid2DS6(BlkPerRow, 6);
	dim3 dimGrid2DS10(BlkPerRow, 10);
	dim3 dimGrid2DSm1(BlkPerRow, RowsPerStream - 1);
	dim3 dimGrid2DSm2(BlkPerRow, RowsPerStream - 2);
	dim3 dimGrid2DSm3(BlkPerRow, RowsPerStream - 3);
	dim3 dimGrid2DSm4(BlkPerRow, RowsPerStream - 4);
	dim3 dimGrid2DSm5(BlkPerRow, RowsPerStream - 5);
	dim3 dimGrid2DSm6(BlkPerRow, RowsPerStream - 6);
	dim3 dimGrid2DSm10(BlkPerRow, RowsPerStream - 10);

	uch  *CPUstart, *GPUstart;
	ui   StartByte, StartRow;
	ui   RowsThisStream;
	switch (NumberOfStreams) {
		case 0:	 chkCUDAErr(cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice));
				 cudaEventRecord(time2, 0);		// Time stamp at the beginning of kernel execution
				 switch(Operation){
					case 'E': BWKernel2S <<< dimGrid2D, ThrPerBlk >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, 0);
							  GaussKernel3S <<< dimGrid2D, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV, 0); 
							  SobelKernel2S <<< dimGrid2D, ThrPerBlk >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, 0);
							  ThresholdKernel2S <<< dimGrid2D, ThrPerBlk >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi,0);
							  break;
					case 'H': Hflip3S <<< dimGrid2D, ThrPerBlk >>> (GPUResultImg, GPUImg, IPH, IPV, IPHB, 0);
							  break;
				 }
				 cudaEventRecord(time3, 0);		// Time stamp at the end of kernel execution
				 chkCUDAErr(cudaMemcpy(CopyImg, GPUResultImg, IMAGESIZE, cudaMemcpyDeviceToHost));
				 break;
		case 1:  chkCUDAErr(cudaMemcpyAsync(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice, stream[0]));
				 cudaEventRecord(time2, 0);		// Time stamp at the beginning of kernel execution
				 switch(Operation) {
					case 'E': BWKernel2S <<< dimGrid2D, ThrPerBlk, 0, stream[0] >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, 0);
							  GaussKernel3S <<< dimGrid2D, ThrPerBlk, 0, stream[0] >>> (GPUGaussImg, GPUBWImg, IPH, IPV, 0);
							  SobelKernel2S <<< dimGrid2D, ThrPerBlk, 0, stream[0] >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, 0);
							  ThresholdKernel2S <<< dimGrid2D, ThrPerBlk, 0, stream[0] >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi, 0);
							  break;
					case 'H': Hflip3S <<< dimGrid2D, ThrPerBlk, 0, stream[0] >>> (GPUResultImg, GPUImg, IPH, IPV, IPHB, 0);
							  break;
				 }
				 cudaEventRecord(time3, 0);		// Time stamp at the end of kernel execution
				 chkCUDAErr(cudaMemcpyAsync(CopyImg, GPUResultImg, IMAGESIZE, cudaMemcpyDeviceToHost, stream[0]));
				 break;
		default: // Check to see if it is horizontal flip
				 if (Operation == 'H') {
					 for (i = 0; i < NumberOfStreams; i++) {
						 StartRow = i*RowsPerStream;
						 StartByte = StartRow*IPHB;
						 CPUstart = TheImg + StartByte;
						 GPUstart = GPUImg + StartByte;
						 RowsThisStream = (i != (NumberOfStreams - 1)) ? RowsPerStream : (IPV - (NumberOfStreams - 1)*RowsPerStream);
						 chkCUDAErr(cudaMemcpyAsync(GPUstart, CPUstart, RowsThisStream*IPHB, cudaMemcpyHostToDevice, stream[i]));
						 cudaEventRecord(time2, 0);			// time2 will time stamp at the end of CPU --> GPU transfer
						 Hflip3S <<< dimGrid2DS, ThrPerBlk, 0, stream[i] >>> (GPUResultImg, GPUImg, IPH, IPV, IPHB, StartRow);
						 cudaEventRecord(time3, 0);			// time2 will time stamp at the end of kernel exec
						 CPUstart = CopyImg + StartByte;
						 GPUstart = GPUResultImg + StartByte;
						 chkCUDAErr(cudaMemcpyAsync(CPUstart, GPUstart, RowsThisStream*IPHB, cudaMemcpyDeviceToHost, stream[i]));
					 }
					 break;
				 }
				 // If not horizontal flip, do edge detection (STREAMING)
				 // Pre-process: 10 rows of B&W, 6 rows of Gauss, 4 rows of Sobel, 2 rows of Threshold
				 for (i = 0; i < (NumberOfStreams-1); i++) {
					StartRow = (i+1)*RowsPerStream-5;
					StartByte = StartRow*IPHB;
					CPUstart = TheImg + StartByte;
					GPUstart = GPUImg + StartByte;
					// Transfer 10 rows between chunk boundaries
					chkCUDAErr(cudaMemcpy(GPUstart, CPUstart, 10*IPHB, cudaMemcpyHostToDevice));
					// Pre-process 10 rows for B&W
					BWKernel2S <<< dimGrid2DS10, ThrPerBlk >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, StartRow);
					// Calculate 6 rows of Gauss, starting @ the last 3 rows for every stream, except the very last one
					StartRow += 2;
					GaussKernel3S <<< dimGrid2DS6, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV, StartRow);
					// Calculate 4 rows of Sobel starting @last-1 row of every stream, except the very last one
					StartRow ++;
					SobelKernel2S <<< dimGrid2DS4, ThrPerBlk >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, StartRow);
					// Calculate 2 rows of Threshold starting @last row of every stream, except the very last one
					//StartRow++;
					//ThresholdKernel2S <<< dimGrid2DS2, ThrPerBlk >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi, StartRow);
				 }
				 cudaEventRecord(time2, 0);			// time2 will time stamp at the end of the pre-processing
				 // Stream data from CPU --> GPU
				 // Streaming B&W
				 // Streaming Gaussian
				 // Streaming Sobel
				 for (i = 0; i < NumberOfStreams; i++) {
					 if (i == 0) {
						 RowsThisStream = RowsPerStream - 5;
					 }else if (i == (NumberOfStreams - 1)) {	 
						 RowsThisStream = IPV - (NumberOfStreams - 1)*RowsPerStream - 5;
					 }else{
						 RowsThisStream = RowsPerStream - 10;
					 }
					 StartRow = ((i == 0) ? 0 : i*RowsPerStream + 5);
					//	printf("Stream=%u ... RowsThisStream=%u\n", i, RowsThisStream);
					 StartByte = StartRow*IPHB;
					 CPUstart = TheImg + StartByte;
					 GPUstart = GPUImg + StartByte;
					 chkCUDAErr(cudaMemcpyAsync(GPUstart, CPUstart, RowsThisStream * IPHB, cudaMemcpyHostToDevice, stream[i]));
					 if (i==0){
						 BWKernel2S <<< dimGrid2DSm5, ThrPerBlk, 0, stream[i] >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, StartRow);
						 GaussKernel3S <<< dimGrid2DSm3, ThrPerBlk, 0, stream[i] >>> (GPUGaussImg, GPUBWImg, IPH, IPV, StartRow);
						 SobelKernel2S <<< dimGrid2DSm2, ThrPerBlk, 0, stream[i] >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, StartRow);
						 //ThresholdKernel2S <<< dimGrid2DSm1, ThrPerBlk, 0, stream[i] >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi, StartRow);
					 }else if (i == (NumberOfStreams - 1)) {
						 BWKernel2S <<< dimGrid2DSm5, ThrPerBlk, 0, stream[i] >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, StartRow);
						 StartRow -= 2;
						 GaussKernel3S <<< dimGrid2DSm3, ThrPerBlk, 0, stream[i] >>> (GPUGaussImg, GPUBWImg, IPH, IPV, StartRow);
						 StartRow--;
						 SobelKernel2S <<< dimGrid2DSm2, ThrPerBlk, 0, stream[i] >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, StartRow);
					 }else {
						 BWKernel2S <<< dimGrid2DSm10, ThrPerBlk, 0, stream[i] >>> (GPUBWImg, GPUImg, IPH, IPV, IPHB, StartRow);
						 StartRow -= 2;
						 GaussKernel3S <<< dimGrid2DSm6, ThrPerBlk, 0, stream[i] >>> (GPUGaussImg, GPUBWImg, IPH, IPV, StartRow);
						 StartRow--;
						 SobelKernel2S <<< dimGrid2DSm4, ThrPerBlk, 0, stream[i] >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV, StartRow);
						 //ThresholdKernel2S <<< dimGrid2DSm2, ThrPerBlk, 0, stream[i] >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi, StartRow);
					 }
				 }
				 //for (i = 0; i < NumberOfStreams; i++) cudaStreamSynchronize(stream[i]);
				 cudaEventRecord(time3, 0);			// time3 will time stamp at the end of BW+Gauss+Sobel
				 // Streaming Threshold
				 for (i = 0; i < NumberOfStreams; i++) {
					 StartRow = i*RowsPerStream;
					 ThresholdKernel2S <<< dimGrid2DS, ThrPerBlk, 0, stream[i] >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, IPHB, ThreshLo, ThreshHi, StartRow);
				 }
				 //for (i = 0; i < NumberOfStreams; i++) cudaStreamSynchronize(stream[i]);

				 // Stream data from GPU --> CPU
				 for (i = 0; i < NumberOfStreams; i++) {
					 StartRow = i*(RowsPerStream-5);
					 StartByte = StartRow*IPHB;
					 CPUstart = CopyImg + StartByte;
					 GPUstart = GPUResultImg + StartByte;
					 RowsThisStream = (i != (NumberOfStreams - 1)) ? (RowsPerStream - 5) : (IPV - (NumberOfStreams - 1)*(RowsPerStream - 5));
					 chkCUDAErr(cudaMemcpyAsync(CPUstart, GPUstart, RowsThisStream*IPHB, cudaMemcpyDeviceToHost, stream[i]));
				 }
	}

	cudaEventRecord(time4, 0);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);
	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&Time12, time1, time2);
	cudaEventElapsedTime(&Time23, time2, time3);
	cudaEventElapsedTime(&Time34, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		cudaFreeHost(TheImg);
		cudaFreeHost(CopyImg);
		cudaFree(GPUptr);
		exit(EXIT_FAILURE);
	}

	WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk
	printf("\n\n");	PrintSep();
	printf("%s        ComputeCapab=%d.%d  [max %s blocks; %d thr/blk; %d SMs] \n", GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk, SMcount);
	printf("Total Global Mem=%u MB   Total Constant Mem=%u KB \n",  DATAMB(GlobalMem), DATAKB(ConstMem));
	PrintSep();
	ui NumBlocks = IPV*BlkPerRow;
	ui GPUDataTransfer;
	printf("%s %s %s %c %u %u  [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, Operation, ThrPerBlk, NumberOfStreams, NumBlocks, BlkPerRow);
	PrintSep();
	switch (Operation) {
		case 'E':	GPUDataTransfer = 2 * IMAGESIZE;
					break;
		case 'H':	ui GPUDataTfrBW = sizeof(double)*IMAGEPIX + sizeof(uch)*IMAGESIZE;
					ui GPUDataTfrGauss = 2 * sizeof(double)*IMAGEPIX;
					ui GPUDataTfrSobel = 3 * sizeof(double)*IMAGEPIX;
					ui GPUDataTfrThresh = sizeof(double)*IMAGEPIX + sizeof(uch)*IMAGESIZE;
					ui GPUDataTfrKernel = GPUDataTfrBW + GPUDataTfrGauss + GPUDataTfrSobel + GPUDataTfrThresh;
					GPUDataTransfer = GPUDataTfrKernel + 2 * IMAGESIZE;
					break;
	}
	if(NumberOfStreams==0){
		printf("Synchronous Mode. NO STREAMING\n");
		PrintSep();
		printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", Time12, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, Time12));
		printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", Time23, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, Time23));
		printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", Time34, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, Time34));
		PrintSep();
		printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2 * IMAGESIZE + GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
		PrintSep();
	}else{
		printf("Streaming Mode. NumberOfStreams=%u   (deviceOverlap=%s)\n", NumberOfStreams, deviceOverlap ? "TRUE" : "FALSE");
		printf("This device is %s capable of simultaneous CPU-to-GPU and GPU-to-CPU data transfers\n", deviceOverlap ? "" : "NOT");
		PrintSep();
		switch (Operation) {
			case 'E': printf("Pre-processing                        =%7.2f ms  \n", Time12);
					  printf("CPU--> GPU Transfer + BW+Gauss+Sobel  =%7.2f ms\n", Time23);
					  printf("Threshold + GPU--> CPU Transfer       =%7.2f ms\n", Time34);
					  break;
			case 'H': printf("CPU--> GPU Transfer                   =%7.2f ms  \n", Time12);
					  printf("Flip kernel                           =%7.2f ms\n", Time23);
					  printf("GPU--> CPU Transfer                   =%7.2f ms\n", Time34);
				
					  break;
		}
		PrintSep();
		printf("Total time elapsed                    =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2 * IMAGESIZE + GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
		PrintSep();
	}

	// Deallocate CPU, GPU memory
	cudaFree(GPUptr);

	// DESTROY EVENTS
	cudaEventDestroy(time1);	
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);

	// DESTROY STREAMS
	if (NumberOfStreams != 0) {
		for (i = 0; i < NumberOfStreams; i++) {
			chkCUDAErr(cudaStreamDestroy(stream[i]));
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		cudaFreeHost(TheImg);
		cudaFreeHost(CopyImg);
		exit(EXIT_FAILURE);
	}
	cudaFreeHost(TheImg);
	cudaFreeHost(CopyImg);
	return(EXIT_SUCCESS);
}

