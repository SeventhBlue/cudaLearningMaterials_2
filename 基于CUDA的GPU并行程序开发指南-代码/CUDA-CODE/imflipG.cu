#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

uch *TheImg, *CopyImg;					// Where images are stored in CPU
uch *GPUImg, *GPUCopyImg, *GPUResult;	// Where images are stored in GPU

struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;

#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)



// Kernel that flips the given image vertically
// each thread only flips a single pixel (R,G,B)
__global__
void Vflip(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	ui BlkPerRow = (Hpixels + ThrPerBlk - 1) / ThrPerBlk;  // ceil
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	ui MYdstIndex = MYdstOffset + 3 * MYcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


// Kernel that flips the given image horizontally
// each thread only flips a single pixel (R,G,B)
__global__
void Hflip(uch *ImgDst, uch *ImgSrc, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	ui BlkPerRow = (Hpixels + ThrPerBlk -1 ) / ThrPerBlk;  // ceil
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorcol = Hpixels - 1 - MYcol;
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	ui MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


// Kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
__global__
void PixCopy(uch *ImgDst, uch *ImgSrc, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	if (MYgtid > FS) return;				// outside the allocated memory
	ImgDst[MYgtid] = ImgSrc[MYgtid];
}


/*
// helper function that wraps CUDA API calls, reports any error and exits
void chkCUDAErr(cudaError_t error_id)
{
	if (error_id != CUDA_SUCCESS)
	{
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
	char Flip = 'H';
	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t cudaStatus, cudaStatus2;
	cudaEvent_t time1, time2, time3, time4;
	char InputFileName[255], OutputFileName[255], ProgName[255];
	ui BlkPerRow, ThrPerBlk=256, NumBlocks, GPUDataTransfer;
	cudaDeviceProp GPUprop;
	ul SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];


	strcpy(ProgName, "imflipG");
	switch (argc){
	case 5:  ThrPerBlk=atoi(argv[4]);
	case 4:  Flip = toupper(argv[3][0]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename [V/H/C/T] [ThrPerBlk]", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp H", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp V  128",ProgName);
			 printf("\n\nH=horizontal flip, V=vertical flip, T=Transpose, C=copy image\n\n");
			 exit(EXIT_FAILURE);
	}
	if ((Flip != 'V') && (Flip != 'H') && (Flip != 'C') && (Flip != 'T')) {
		printf("Invalid flip option '%c'. Must be 'V','H', 'T', or 'C'... \n", Flip);
		exit(EXIT_FAILURE);
	}
	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
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
		free(TheImg);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer
	// Allocate GPU buffer for the input and output images
	cudaStatus = cudaMalloc((void**)&GPUImg, IMAGESIZE);
	cudaStatus2 = cudaMalloc((void**)&GPUCopyImg, IMAGESIZE);
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	//dim3 dimBlock(ThrPerBlk);
	//dim3 dimGrid(ip.Hpixels*BlkPerRow);

	BlkPerRow = (IPH + ThrPerBlk -1 ) / ThrPerBlk;
	NumBlocks = IPV*BlkPerRow; 
	switch (Flip){
		case 'H': Hflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2*IMAGESIZE;
				  break;
		case 'V': Vflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV);
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2*IMAGESIZE;
				  break;
		case 'T': Hflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
				  Vflip <<< NumBlocks, ThrPerBlk >>> (GPUImg, GPUCopyImg, IPH, IPV);
				  GPUResult = GPUImg;
				  GPUDataTransfer = 4*IMAGESIZE;
				  break;
		case 'C': NumBlocks = (IMAGESIZE+ThrPerBlk-1) / ThrPerBlk;
				  PixCopy <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IMAGESIZE);
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2*IMAGESIZE;
				  break;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time3, 0);

	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk
	printf("\n\n--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n", 
			GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");
	printf("%s %s %s %c %u   [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName,
			Flip, ThrPerBlk, NumBlocks, BlkPerRow);
	printf("--------------------------------------------------------------------------\n");
	printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
	printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
	printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU));
	printf("--------------------------------------------------------------------------\n");
	printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2 * IMAGESIZE + GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
	printf("--------------------------------------------------------------------------\n\n");

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUImg);
	cudaFree(GPUCopyImg);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
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
}



