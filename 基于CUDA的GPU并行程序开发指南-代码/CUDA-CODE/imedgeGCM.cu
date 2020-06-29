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
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))
#define MAXTHGAUSSKN4	128
#define MAXTHGAUSSKN5	128
#define MAXTHGAUSSKN67	1024
#define MAXTHGAUSSKN8	256

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
	ImgBW[MYpixIndex] = (R + G + B) / 3.0;
}



// Improved BWKernel. Uses pre-computed values and 2D blocks.
__global__
void BWKernel2(double *ImgBW, uch *ImgGPU, ui Hpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	// ui MYgtid = ThrPerBlk * MYbid + MYtid;
	double R, G, B;

	//ui NumBlocks = gridDim.x;
	// ui BlkPerRow = CEIL(Hpixels, ThrPerBlk);
	// ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYsrcIndex = MYrow * RowBytes + 3 * MYcol;
	ui MYpixIndex = MYrow * Hpixels + MYcol;

	B = (double)ImgGPU[MYsrcIndex];
	G = (double)ImgGPU[MYsrcIndex + 1];
	R = (double)ImgGPU[MYsrcIndex + 2];
	ImgBW[MYpixIndex] = (R + G + B) / 3.0;
}



// Improved BWKernel2. Calculates 4 pixels (3 int's) at a time
__global__
void BWKernel3(double *ImgBW, ui *ImgGPU32, ui Hpixels, ui RowInts)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui A, B, C;
	ui Pix1, Pix2, Pix3, Pix4;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	ui MYcolIndex = MYcol*3;
	if (MYcolIndex >= RowInts) return;			// col out of range
	ui MYoffset = MYrow * RowInts;
	ui MYsrcIndex = MYoffset + MYcolIndex;
	ui MYpixAddr = MYrow * Hpixels + MYcol*4;

	A = ImgGPU32[MYsrcIndex];					// A=[B1,R0,G0,B0]
	B = ImgGPU32[MYsrcIndex+1];					// B=[G2,B2,R1,G1]
	C = ImgGPU32[MYsrcIndex+2];					// C=[R3,G3,B3,R2]
	// Pix1 = R0+G0+B0;
	Pix1 = (A & 0x000000FF) + ((A >> 8) & 0x000000FF) + ((A >> 16) & 0x000000FF);
	// Pix2 = R1+G1+B1;
	Pix2 = ((A >> 24) & 0x000000FF) + (B & 0x000000FF) + ((B >> 8) & 0x000000FF);
	// Pix3 = R2+G2+B2;
	Pix3 = (C & 0x000000FF) + ((B >> 16) & 0x000000FF) + ((B >> 24) & 0x000000FF);
	// Pix4 = R3+G3+B3;
	Pix4 = ((C >> 8) & 0x000000FF) + ((C >> 16) & 0x000000FF) + ((C >> 24) & 0x000000FF);
	ImgBW[MYpixAddr]     = (double)Pix1 * 0.33333333;
	ImgBW[MYpixAddr + 1] = (double)Pix2 * 0.33333333;
	ImgBW[MYpixAddr + 2] = (double)Pix3 * 0.33333333;
	ImgBW[MYpixAddr + 3] = (double)Pix4 * 0.33333333;
}


__device__
double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 } };
// Kernel that calculates a Gauss image from the B&W image (one pixel)
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
		ImgGauss[MYpixIndex] = G/159.00;
	}
}


// Improved GaussKernel. Uses 2D blocks. Each kernel processes a single pixel
__global__
void GaussKernel2(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int row, col, indx, i, j;
	double G = 0.00;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
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


__constant__
double GaussC[5][5] = { { 2, 4, 5, 4, 2 },
						{ 4, 9, 12, 9, 4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9, 12, 9, 4 },
						{ 2, 4, 5, 4, 2 } };
// Improved GaussKernel2. Uses constant memory to store filter coefficients
__global__
void GaussKernel3(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int row, col, indx, i, j;
	double G;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
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
				G += (ImgBW[indx] * GaussC[i + 2][j + 2]);  // use constant memory
			}
		}
		ImgGauss[MYpixIndex] = G / 159.00;
	}
}


// Improved GaussKernel3. Reads multiple (5) rows into shared memory.
// Each thread computes 1 pixel.
__global__
void GaussKernel4(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	// 5 horizontal, 5 vertical neighbors stored in Shared Memory
	__shared__ double Neighbors[MAXTHGAUSSKN4][5][5];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int row, col, indx, i, j;
	double G;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)) {
		ImgGauss[MYpixIndex] = 0.0;
		return;
	}

	// Read from GM to Shared Memory
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			row = MYrow + i - 2;
			col = MYcol + j - 2;
			indx = row * Hpixels + col;
			Neighbors[MYtid][i][j] = ImgBW[indx];
		}
		//__syncthreads();
	}
	__syncthreads();

	G = 0.0;
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			G += (Neighbors[MYtid][i][j] * GaussC[i][j]);
		}
	}
	//__syncthreads();

	ImgGauss[MYpixIndex] = G / 159.00;
}


// Improved GaussKernel3. Reads multiple (5) rows into shared memory.
// Each thread computes 4 pixels. Horizontal resolution must be a multiple of 4.
__global__
void GaussKernel5(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	// 8 horizontal, 5 vertical neighbors
	__shared__ double Neighbors[MAXTHGAUSSKN5][5][8];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int row, col, indx, i, j, k;
	double G;

	ui MYrow = blockIdx.y;
	ui MYcol = (MYbid*ThrPerBlk + MYtid) * 4;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow < 2) || (MYrow > Vpixels - 3)){ // Top and bottom two rows
		ImgGauss[MYpixIndex] = 0.0;
		ImgGauss[MYpixIndex+1] = 0.0;
		ImgGauss[MYpixIndex+2] = 0.0;
		ImgGauss[MYpixIndex+3] = 0.0;
		return;
	}
	if (MYcol > Hpixels - 3) {				// Rightmost two columns
		ImgGauss[MYpixIndex] = 0.0;
		ImgGauss[MYpixIndex + 1] = 0.0; 
		return;
	}
	if (MYcol < 2) {						// Leftmost two columns
		ImgGauss[MYpixIndex] = 0.0;
		ImgGauss[MYpixIndex + 1] = 0.0;
		return;
	}
	MYpixIndex += 2;						// Process 2 pix. shifted
	MYcol += 2;
	
	// Read from GM to Shared Memory
	for (i = 0; i < 5; i++){
		for (j = 0; j < 8; j++){
			row = MYrow + i - 2;
			col = MYcol + j - 2;
			indx = row * Hpixels + col;
			Neighbors[MYtid][i][j] = ImgBW[indx];
		}
	}
	__syncthreads();

	for (k = 0; k < 4; k++){
		G = 0.000;
		for (i = 0; i < 5; i++){
			for (j = 0; j < 5; j++){
				G += (Neighbors[MYtid][i][j+k] * GaussC[i][j]);
			}
		}
		//__syncthreads();
		ImgGauss[MYpixIndex+k] = G / 159.00;
	}
}


// Improved GaussKernel4. Each thread computes 1 pixel.
// Each thread reads 5 pixels into Shared Memory.
// Pixel at the same column, but 5 different rows (row-2 ... row+2)
__global__
void GaussKernel6(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	// 5 vertical neighbors for each pixel that is represented by a thread
	__shared__ double Neighbors[MAXTHGAUSSKN67+4][5];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int indx, i, j;
	double G;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)) {
		ImgGauss[MYpixIndex] = 0.0;
		return;
	}
	ui IsEdgeThread=(MYtid==(ThrPerBlk-1));

	// Read from GM to Shared Memory
	// Each thread will read a single pixel
	indx = MYpixIndex-2*Hpixels-2; // start 2 rows above & 2 columns left
	if (!IsEdgeThread) {
		for (j = 0; j < 5; j++) {
			Neighbors[MYtid][j] = ImgBW[indx];
			indx += Hpixels; // Next iteration will read next row, same column 
		}
	}else{
		for (j = 0; j < 5; j++) {
			Neighbors[MYtid][j] = ImgBW[indx];
			Neighbors[MYtid + 1][j] = ImgBW[indx + 1];
			Neighbors[MYtid + 2][j] = ImgBW[indx + 2];
			Neighbors[MYtid + 3][j] = ImgBW[indx + 3];
			Neighbors[MYtid + 4][j] = ImgBW[indx + 4];
			indx += Hpixels; // Next iteration will read next row, same column 
		}
	}
	__syncthreads();


	G = 0.0;
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			G += (Neighbors[MYtid+i][j] * GaussC[i][j]); 
		}
	}
	//__syncthreads();

	ImgGauss[MYpixIndex] = G / 159.00;
}


// Improved GaussKernel6. Each block computes ThePerBlk-4 pixels.
// This eliminates the need to make exceptions for the "Edge" thread
__global__
void GaussKernel7(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	// 5 vertical neighbors for each pixel (read by each thread)
	__shared__ double Neighbors[MAXTHGAUSSKN67][5];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int indx, i, j;
	double G;

	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*(ThrPerBlk-4) + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYrow<2) || (MYrow>Vpixels - 3) || (MYcol<2) || (MYcol>Hpixels - 3)) {
		ImgGauss[MYpixIndex] = 0.0;
		return;
	}

	// Read from GM to Shared Memory.
	// Each thread will read a single pixel, for 5 neighboring rows
	// Each block reads ThrPerBlk pixels starting at (2 left) location
	indx = MYpixIndex - 2 * Hpixels - 2; // start 2 rows above & 2 columns left
	for (j = 0; j < 5; j++) {
		Neighbors[MYtid][j] = ImgBW[indx];
		indx += Hpixels; // Next iteration will read next row, same column 
	}
	__syncthreads();

	if (MYtid >= ThrPerBlk - 4) return; // Each block only computes only ThrPerBlk-4 pixels 

	G = 0.0;
	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			G += (Neighbors[MYtid + i][j] * GaussC[i][j]);
		}
	}
	//__syncthreads();

	ImgGauss[MYpixIndex] = G / 159.00;
}


// Improved GaussKernel7. Each block reads 12 rows.
// Each thread computes 8 vertical pixels.
__global__
void GaussKernel8(double *ImgGauss, double *ImgBW, ui Hpixels, ui Vpixels)
{
	// 12 vertical neighbors are saved in the Shared Memory
	// These are used to compute 8 vertical pixels by each thread
	// Reads from 2 top and 2 bottom pixels are wasted.
	__shared__ double Neighbors[MAXTHGAUSSKN8][12];

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	int indx, i, j, row;
	double G[8] = { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 };

	ui MYrow = blockIdx.y*8;
	ui isLastBlockY = (blockIdx.y == (blockDim.y - 1));
	ui MYcol = MYbid*(ThrPerBlk - 4) + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	if (MYrow >= Vpixels) return;			// row out of range

	ui MYpixIndex = MYrow * Hpixels + MYcol;
	if ((MYcol<2) || (MYcol>Hpixels - 3)) {
		ImgGauss[MYpixIndex] = 0.0;			// first and last 2 columns
		return;
	}
	if (MYrow == 0) {
		ImgGauss[MYpixIndex] = 0.0;					// row0
		ImgGauss[MYpixIndex+Hpixels] = 0.0;			// row1
	}
	if (isLastBlockY) {
		indx = (Vpixels - 2)*Hpixels + MYcol;
		ImgGauss[indx] = 0.0;					// last row-1
		ImgGauss[indx + Hpixels] = 0.0;			// last row
	}

	// Read from GM to Shared Memory.
	// Each thread will read a single pixel, for 12 neighboring rows
	// Each thread reads 12 pixels, but will only compute 8
	indx = MYpixIndex;
	for (j = 0; j < 12; j++) {
		if ((MYrow+j) < Vpixels) {
			Neighbors[MYtid][j] = ImgBW[indx];
			indx += Hpixels; // Next iteration will read next row, same column 
		}else{
			Neighbors[MYtid][j] = 0.00;
		}
	}
	__syncthreads();
	if (MYtid >= ThrPerBlk - 4) return; // Each block only computes only ThrPerBlk-4 pixels 

	for (row = 0; row < 8; row++) {
		for (i = 0; i < 5; i++) {
			for (j = 0; j < 5; j++) {
				G[row] += (Neighbors[MYtid + i][row+j] * GaussC[i][j]);
			}
		}
	}

	// Write all computed pixels back to GM
	for (j = 0; j < 8; j++) {
		ImgGauss[MYpixIndex] = G[j] / 159.00;
		MYpixIndex += Hpixels;
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
	// GPU code run times
	float			totalTime, totalKernelTime, tfrCPUtoGPU, tfrGPUtoCPU;
	float			kernelExecTimeBW, kernelExecTimeGauss, kernelExecTimeSobel, kernelExecTimeThreshold;
	cudaError_t		cudaStatus;
	cudaEvent_t		time1, time2, time2BW, time2Gauss, time2Sobel, time3, time4;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow, BlkPerRowG, ThrPerBlk=256, NumBlocks, NumBlocksG, NumBlocksG8;
	ui				GPUDataTfrBW, GPUDataTfrGauss, GPUDataTfrSobel, GPUDataTfrThresh, GPUDataTfrKernel, GPUDataTfrTotal;
	ui				RowBytes, RowInts;
	ui				*GPUImg32;
	cudaDeviceProp	GPUprop;
	void			*GPUptr;			// Pointer to the bulk-allocated GPU memory
	ul				GPUtotalBufferSize;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;		char SupportedBlocks[100];
	int				BWKN=1, GaussKN=1, SobelKN=1, ThresholdKN=1;
	char			BWKernelName[255], GaussKernelName[255], SobelKernelName[255], ThresholdKernelName[255];

	strcpy(ProgName, "imedgeGCM");
	switch (argc){
		case 10: ThresholdKN = atoi(argv[9]);
		case 9:  SobelKN = atoi(argv[8]);
		case 8:  GaussKN = atoi(argv[7]);
		case 7:  BWKN = atoi(argv[6]);
		case 6:  ThreshHi = atoi(argv[5]);
		case 5:  ThreshLo  = atoi(argv[4]);
		case 4:  ThrPerBlk = atoi(argv[3]);
		case 3:  strcpy(InputFileName, argv[1]);
				 strcpy(OutputFileName, argv[2]);
				 break;
		default: printf("\n\nUsage:   %s InputFilename OutputFilename [ThrPerBlk] [ThreshLo] [ThreshHi] [BWKernel=1-9] [GaussKernel=1-9] [SobelKernel=1-9] [ThresholdKernel=1-9]", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp 256", ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp 256 50 100",ProgName);
				 printf("\n\nExample: %s Astronaut.bmp Output.bmp 256 50 100 1 3 4 5", ProgName);
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
	if ((BWKN < 1) || (BWKN > 9) || (GaussKN < 1) || (GaussKN > 9) || (SobelKN < 1) || (SobelKN > 9) || (ThresholdKN < 1) || (ThresholdKN > 9)) {
		printf("Invalid kernel number ... Kernel numbers must be between 1 and 9\n");
		if ((BWKN < 1) || (BWKN > 9)) printf("BW Kernel number %d is out of range",BWKN);
		if ((GaussKN < 1) || (GaussKN > 9)) printf(" Kernel number %d is out of range", GaussKN);
		if ((SobelKN < 1) || (SobelKN > 9)) printf(" Kernel number %d is out of range", SobelKN);
		if ((ThresholdKN < 1) || (ThresholdKN > 9)) printf(" Kernel number %d is out of range", ThresholdKN);
		printf("\n\nNothing executed ... Exiting ...\n\n");
		exit(EXIT_FAILURE);
	}
	// Handle special cases
	if ((GaussKN == 4) && (ThrPerBlk>MAXTHGAUSSKN4)){
		printf("ThrPerBlk cannot be higher than %d in Gauss Kernel 4 ... Set to %d.\n", MAXTHGAUSSKN4, MAXTHGAUSSKN4);
		ThrPerBlk = MAXTHGAUSSKN4;
	}
	if ((GaussKN == 5) && (ThrPerBlk>MAXTHGAUSSKN5)) {
		printf("ThrPerBlk cannot be higher than %d in Gauss Kernel 5 ... Set to %d.\n", MAXTHGAUSSKN5, MAXTHGAUSSKN5);
		ThrPerBlk = MAXTHGAUSSKN5;
	}
	if (( (GaussKN == 6) || (GaussKN == 7)) && (ThrPerBlk>MAXTHGAUSSKN67)) {
		printf("ThrPerBlk cannot be higher than %d in Gauss Kernel 6 or 7 ... Set to %d.\n", MAXTHGAUSSKN67, MAXTHGAUSSKN67);
		ThrPerBlk = MAXTHGAUSSKN67;
	}
	if ((GaussKN == 8) && (ThrPerBlk>MAXTHGAUSSKN8)) {
		printf("ThrPerBlk cannot be higher than %d in Gauss Kernel 8 ... Set to %d.\n", MAXTHGAUSSKN8, MAXTHGAUSSKN8);
		ThrPerBlk = MAXTHGAUSSKN8;
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

	RowBytes = (IPH * 3 + 3) & (~3);
	RowInts = RowBytes / 4;
	BlkPerRow   = CEIL(IPH, ThrPerBlk);
	BlkPerRowG  = CEIL(IPH, (ThrPerBlk-4));
	NumBlocks   = BlkPerRow  * IPV;
	NumBlocksG  = BlkPerRowG * IPV;
	NumBlocksG8 = BlkPerRowG * CEIL(IPV, 8);
	dim3 dimGrid2D(BlkPerRow, ip.Vpixels);
	dim3 dimGrid2D4(CEIL(BlkPerRow, 4), IPV);
	dim3 dimGrid2DG(BlkPerRowG, IPV);
	dim3 dimGrid2DG8(BlkPerRowG, CEIL(IPV, 8));
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		goto EXITERROR;
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
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
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory\n");
		goto EXITERROR;
	}
	GPUImg			= (uch *)GPUptr;
	GPUImg32		= (ui *)GPUImg;
	GPUResultImg	= GPUImg + IMAGESIZE;
	GPUBWImg		= (double *)(GPUResultImg + IMAGESIZE);
	GPUGaussImg		= GPUBWImg + IMAGEPIX;
	GPUGradient		= GPUGaussImg + IMAGEPIX;
	GPUTheta		= GPUGradient + IMAGEPIX;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!\n");
		goto EXITCUDAERROR;
	}
	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	switch (BWKN){
		case 1: BWKernel <<< NumBlocks, ThrPerBlk >>> (GPUBWImg, GPUImg, IPH);
				strcpy(BWKernelName, "BWKernel:        Everything is passed into the kernel");
				break;
		case 2: BWKernel2 <<< dimGrid2D, ThrPerBlk >>> (GPUBWImg, GPUImg, IPH, RowBytes);
				strcpy(BWKernelName, "BWKernel2:       Pre-computed values and 2D blocks");
				break;
		case 3: BWKernel3 <<< dimGrid2D4, ThrPerBlk >>> (GPUBWImg, GPUImg32, IPH, RowInts);
				strcpy(BWKernelName, "BWKernel3:       Calculates 4 pixels (3 int) at a time");
				break;
		default:printf("...... BW Kernel Number=%d ... NOT IMPLEMENTED .... \n", BWKN);
				strcpy(BWKernelName, "*** NOT IMPLEMENTED ***");
				break;
	}
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR;
	cudaEventRecord(time2BW, 0);		// Time stamp after BW image calculation
	GPUDataTfrBW = sizeof(double)*IMAGEPIX + sizeof(uch)*IMAGESIZE;

	switch (GaussKN){
		case 1: GaussKernel <<< NumBlocks, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel:     Everything is passed into the kernel");
				break;
		case 2: GaussKernel2 <<< dimGrid2D, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel2:    Uses 2D blocks");
				break;
		case 3: GaussKernel3 <<< dimGrid2D, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel3:    Stores filter coeff in constant memory");
				break;
		case 4: GaussKernel4 <<< dimGrid2D, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel4:    Computes 1 pix/thread using Shared Memory");
				break;
		case 5: GaussKernel5 <<< dimGrid2D4, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel5:    Computes 4 pix/thread using Shared Memory");
				break;
		case 6: GaussKernel6 <<< dimGrid2D, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel6:    Each thread reads 5 rows of pixels into ShMem");
				break;
		case 7: GaussKernel7 <<< dimGrid2DG, ThrPerBlk >>> (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel7:    Blocks read 5 rows, compute ThrPerBlk-4 pixels");
				break;
		case 8: GaussKernel8 << < dimGrid2DG8, ThrPerBlk >> > (GPUGaussImg, GPUBWImg, IPH, IPV);
				strcpy(GaussKernelName, "GaussKernel8:    Blocks read 12 vertical pixels, and compute 8");
				break;
		default:printf("...... Gauss Kernel Number=%d ... NOT IMPLEMENTED .... \n", GaussKN);
			strcpy(GaussKernelName, "*** NOT IMPLEMENTED ***");
			break;
	}
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR; 
	cudaEventRecord(time2Gauss, 0);		// Time stamp after Gauss image calculation
	GPUDataTfrGauss = 2 * sizeof(double)*IMAGEPIX;

	switch (SobelKN){
		case 1: SobelKernel <<< NumBlocks, ThrPerBlk >>> (GPUGradient, GPUTheta, GPUGaussImg, IPH, IPV);
				strcpy(SobelKernelName, "SobelKernel:     Everything is passed into the kernel");
				break;
		default:printf("...... Sobel Kernel Number=%d ... NOT IMPLEMENTED .... \n", SobelKN);
			strcpy(SobelKernelName, "*** NOT IMPLEMENTED ***");
			break;
	}
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) goto KERNELERROR; 
	cudaEventRecord(time2Sobel, 0);		// Time stamp after Gradient, Theta computation
	GPUDataTfrSobel = 3 * sizeof(double)*IMAGEPIX;

	switch (ThresholdKN){
		case 1: ThresholdKernel <<< NumBlocks, ThrPerBlk >>> (GPUResultImg, GPUGradient, GPUTheta, IPH, IPV, ThreshLo, ThreshHi);
				strcpy(ThresholdKernelName, "ThresholdKernel: Everything is passed into the kernel");
				break;	
		default:printf("...... Threshold Kernel Number=%d ... NOT IMPLEMENTED .... \n",ThresholdKN);
				strcpy(ThresholdKernelName, "*** NOT IMPLEMENTED ***");
				break;
	}
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
	printf("\n\n--------------------------------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n",
		GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------------------------------\n");
	printf("%s %s %s %u %d %d %d %d %d %d   [Launched %u BLOCKS, %u BLOCKS/ROW]\n", 
		ProgName, InputFileName, OutputFileName, ThrPerBlk, ThreshLo, ThreshHi, BWKN, GaussKN, SobelKN, ThresholdKN, NumBlocks, BlkPerRow);
	if (GaussKN == 7) {
		printf("                                   Gauss Kernel 7: [Launched %u BLOCKS, %u BLOCKS/ROW]\n", NumBlocksG, BlkPerRowG);
	}
	if (GaussKN == 8) {
		printf("                                   Gauss Kernel 8: [Launched %u BLOCKS, %u BLOCKS/ROW]\n", NumBlocksG8, BlkPerRowG);
	}
	printf("--------------------------------------------------------------------------------------------------\n");
	printf("              CPU->GPU Transfer =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
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



