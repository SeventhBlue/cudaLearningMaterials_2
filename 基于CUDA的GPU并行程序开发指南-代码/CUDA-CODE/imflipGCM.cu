#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

#define	CEIL(a,b)		((a+b-1)/b)
#define SWAP(a,b,t)		t=b; b=a; a=t;
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



// Kernel that flips the given image horizontally
// each thread only flips a single pixel (R,G,B)
__global__
void Hflip(uch *ImgDst, uch *ImgSrc, ui Hpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
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


// Improved Hflip() kernel that flips the given image horizontally
// BlkPerRow, RowBytes variables are passed, rather than calculated
__global__
void Hflip2(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui BlkPerRow, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
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


// Improved Hflip2() kernel that flips the given image horizontally
// Grid is launched using 2D block numbers
__global__
void Hflip3(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
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


// Improved Hflip3() kernel that flips the given image horizontally
// Each kernel takes care of 2 consecutive pixels; half as many blocks are launched
__global__
void Hflip4(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol2 = (MYbid*ThrPerBlk + MYtid)*2;
	if (MYcol2 >= Hpixels) return;			// col (and col+1) are out of range
	ui MYmirrorcol = Hpixels - 1 - MYcol2;
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol2;
	ui MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
	if ((MYcol2 + 1) >= Hpixels) return;			// only col+1 is out of range
	ImgDst[MYdstIndex - 3] = ImgSrc[MYsrcIndex + 3];
	ImgDst[MYdstIndex - 2] = ImgSrc[MYsrcIndex + 4];
	ImgDst[MYdstIndex - 1] = ImgSrc[MYsrcIndex + 5];
}


// Improved Hflip3() kernel that flips the given image horizontally
// Each kernel takes care of 4 consecutive pixels; 1/4 as many blocks are launched
__global__
void Hflip5(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol4 = (MYbid*ThrPerBlk + MYtid) * 4;
	if (MYcol4 >= Hpixels) return;			// col (and col+1) are out of range
	ui MYmirrorcol = Hpixels - 1 - MYcol4;
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol4;
	ui MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	for (ui a = 0; a<4; a++){
		ImgDst[MYdstIndex - a * 3] = ImgSrc[MYsrcIndex + a * 3];
		ImgDst[MYdstIndex - a * 3 + 1] = ImgSrc[MYsrcIndex + a * 3 + 1];
		ImgDst[MYdstIndex - a * 3 + 2] = ImgSrc[MYsrcIndex + a * 3 + 2];
		if ((MYcol4 + a + 1) >= Hpixels) return;			// next pixel is out of range
	}
}


// Improved Hflip3() kernel that flips the given image horizontally
// Each kernel: copies a pixel from GlobalMem into shared memory (PixBuffer[])
// and writes back into the flipped Global Memory location
__global__
void Hflip6(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes)
{
	__shared__ uch PixBuffer[3072]; // holds 3*1024 Bytes (1024 pixels).

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYtid3 = MYtid * 3;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorcol = Hpixels - 1 - MYcol;
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	ui MYdstIndex = MYoffset + 3 * MYmirrorcol;

	// swap pixels RGB   @MYcol , @MYmirrorcol
	PixBuffer[MYtid3] = ImgSrc[MYsrcIndex];
	PixBuffer[MYtid3 + 1] = ImgSrc[MYsrcIndex + 1];
	PixBuffer[MYtid3 + 2] = ImgSrc[MYsrcIndex + 2];
	__syncthreads();
	ImgDst[MYdstIndex] = PixBuffer[MYtid3];
	ImgDst[MYdstIndex + 1] = PixBuffer[MYtid3 + 1];
	ImgDst[MYdstIndex + 2] = PixBuffer[MYtid3 + 2];
}


// Improved Hflip6() kernel that flips the given image horizontally
// Each kernel: uses Shared Memory (PixBuffer[]) to read in 12 Bytes
// (4 pixels). 12Bytes are flipped inside Shared Memory 
// After that, they are written into Global Mem as 3 int's
// Horizontal resolution MUST BE A POWER OF 4.
__global__
void Hflip7(ui *ImgDst32, ui *ImgSrc32, ui RowInts)
{
	__shared__ ui PixBuffer[3072]; // holds 3*1024*4 Bytes (1024*4 pixels).

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYtid3 = MYtid * 3;
	ui MYrow = blockIdx.y;
	ui MYcolIndex = (MYbid*ThrPerBlk + MYtid)*3;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYmirrorcol = RowInts - 1 - MYcolIndex;
	ui MYoffset = MYrow * RowInts;
	ui MYsrcIndex = MYoffset + MYcolIndex;
	ui MYdstIndex = MYoffset + MYmirrorcol - 2; // -2 is to copy 3 Bytes at a time
	uch SwapB;
	uch *SwapPtr;

	// read 4 pixel blocks (12B = 3 int's) into Shared Memory
	// PixBuffer:  [B0 G0 R0 B1] [G1 R1 B2 G2] [R2 B3 G3 R3]
	// Our Target: [B3 G3 R3 B2] [G2 R2 B1 G1] [R1 B0 G0 R0]
	PixBuffer[MYtid3] = ImgSrc32[MYsrcIndex];
	PixBuffer[MYtid3+1] = ImgSrc32[MYsrcIndex+1];
	PixBuffer[MYtid3+2] = ImgSrc32[MYsrcIndex+2];
	__syncthreads();
	
	// swap these 4 pixels inside Shared Memory
	SwapPtr = (uch *)(&PixBuffer[MYtid3]);      // [B0 G0 R0 B1] [G1 R1 B2 G2] [R2 B3 G3 R3]
	SWAP(SwapPtr[0], SwapPtr[9], SwapB)			// [B3 G0 R0 B1] [G1 R1 B2 G2] [R2 B0 G3 R3]
	SWAP(SwapPtr[1], SwapPtr[10], SwapB)		// [B3 G3 R0 B1] [G1 R1 B2 G2] [R2 B0 G0 R3]
	SWAP(SwapPtr[2], SwapPtr[11], SwapB)		// [B3 G3 R3 B1] [G1 R1 B2 G2] [R2 B0 G0 R0]
	SWAP(SwapPtr[3], SwapPtr[6], SwapB)			// [B3 G3 R3 B2] [G1 R1 B1 G2] [R2 B0 G0 R0]
	SWAP(SwapPtr[4], SwapPtr[7], SwapB)			// [B3 G3 R3 B2] [G2 R1 B1 G1] [R2 B0 G0 R0]
	SWAP(SwapPtr[5], SwapPtr[8], SwapB)			// [B3 G3 R3 B2] [G2 R2 B1 G1] [R1 B0 G0 R0]

	__syncthreads();
	//write the 4 pixels (3 int's) from Shared Memory into Global Memory
	ImgDst32[MYdstIndex] = PixBuffer[MYtid3];
	ImgDst32[MYdstIndex+1] = PixBuffer[MYtid3+1];
	ImgDst32[MYdstIndex+2] = PixBuffer[MYtid3+2];
}


// Improved Hflip7() that swaps 12Bytes (4 pixels) using registers
__global__
void Hflip8(ui *ImgDst32, ui *ImgSrc32, ui RowInts)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcolIndex = (MYbid*ThrPerBlk + MYtid) * 3;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYmirrorcol = RowInts - 1 - MYcolIndex;
	ui MYoffset = MYrow * RowInts;
	ui MYsrcIndex = MYoffset + MYcolIndex;
	ui MYdstIndex = MYoffset + MYmirrorcol - 2; // -2 is to copy 3 Bytes at a time
	ui  A, B, C, D, E, F;

	// read 4 pixel blocks (12B = 3 int's) into 3 long registers
	A = ImgSrc32[MYsrcIndex];
	B = ImgSrc32[MYsrcIndex + 1];
	C = ImgSrc32[MYsrcIndex + 2];
	
	// Do the shuffling using these registers
	//NOW:		  A=[B1,R0,G0,B0]   B=[G2,B2,R1,G1]    C=[R3,G3,B3,R2]
	//OUR TARGET: D=[B2,R3,G3,B3]   E=[G1,B1,R2,G2]    F=[R0,G0,B1,R1]
	D = (C >> 8) | ((B << 8) & 0xFF000000);     // D=[B2,R3,G3,B3]
	E = (B << 24) | (B >> 24) | ((A >> 8) & 0x00FF0000) | ((C << 8) & 0x0000FF00);     // E=[G1,B1,R2,G2]
	F = ((A << 8) & 0xFFFF0000) | ((A >> 16) & 0x0000FF00) | ((B >> 8) & 0x000000FF);		// F=[R0,G0,B1,R1]

	//write the 4 pixels (3 int's) from Shared Memory into Global Memory
	ImgDst32[MYdstIndex] = D;
	ImgDst32[MYdstIndex + 1] = E;
	ImgDst32[MYdstIndex + 2] = F;
}


// Kernel that flips the given image vertically
// each thread only flips a single pixel (R,G,B)
__global__
void Vflip(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	ui MYdstIndex = MYdstOffset + 3 * MYcol;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


// Improved Vflip() kernel that flips the given image vertically
// BlkPerRow, RowBytes variables are passed, rather than calculated
__global__
void Vflip2(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui BlkPerRow, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	ui MYrow = MYbid / BlkPerRow;
	ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	ui MYdstIndex = MYdstOffset + 3 * MYcol;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


// Improved Vflip2() kernel that flips the given image vertically
// Grid is launched using 2D block numbers
__global__
void Vflip3(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	ui MYdstIndex = MYdstOffset + 3 * MYcol;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
}


// Improved Vflip3() kernel that flips the given image vertically
// Each kernel takes care of 2 consecutive pixels; half as many blocks are launched
__global__
void Vflip4(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol2 = (MYbid*ThrPerBlk + MYtid)*2;
	if (MYcol2 >= Hpixels) return;				// col is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol2;
	ui MYdstIndex = MYdstOffset + 3 * MYcol2;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	ImgDst[MYdstIndex] = ImgSrc[MYsrcIndex];
	ImgDst[MYdstIndex + 1] = ImgSrc[MYsrcIndex + 1];
	ImgDst[MYdstIndex + 2] = ImgSrc[MYsrcIndex + 2];
	if ((MYcol2+1) >= Hpixels) return;			// only col+1 is out of range
	ImgDst[MYdstIndex + 3] = ImgSrc[MYsrcIndex + 3];
	ImgDst[MYdstIndex + 4] = ImgSrc[MYsrcIndex + 4];
	ImgDst[MYdstIndex + 5] = ImgSrc[MYsrcIndex + 5];
}


// Improved Vflip3() kernel that flips the given image vertically
// Each kernel takes care of 4 consecutive pixels; 1/4 as many blocks are launched
__global__
void Vflip5(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui RowBytes)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol4 = (MYbid*ThrPerBlk + MYtid)*4;
	if (MYcol4 >= Hpixels) return;			// col is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol4;
	ui MYdstIndex = MYdstOffset + 3 * MYcol4;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	for (ui a=0;  a<4;  a++){
		ImgDst[MYdstIndex + a * 3] = ImgSrc[MYsrcIndex + a * 3];
		ImgDst[MYdstIndex + a * 3 + 1] = ImgSrc[MYsrcIndex + a * 3 + 1];
		ImgDst[MYdstIndex + a * 3 + 2] = ImgSrc[MYsrcIndex + a * 3 + 2];
		if ((MYcol4 + a + 1) >= Hpixels) return;			// next pixel is out of range
	}
}


// Improved Vflip3() kernel that flips the given image vertically
// Each kernel: copies a pixel from GlobalMem into shared memory (PixBuffer[])
// and writes back into the flipped Global Memory location
__global__
void Vflip6(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui Vpixels, ui RowBytes)
{
	__shared__ uch PixBuffer[3072]; // holds 3*1024 Bytes (1024 pixels).

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYtid3 = MYtid*3;
	//ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui NumBlocks = gridDim.x;
	//ui BlkPerRow = CEIL(Hpixels,ThrPerBlk);
	//ui RowBytes = (Hpixels * 3 + 3) & (~3);
	//ui MYrow = MYbid / BlkPerRow;
	//ui MYcol = MYgtid - MYrow*BlkPerRow*ThrPerBlk;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowBytes;
	ui MYdstOffset = MYmirrorrow * RowBytes;
	ui MYsrcIndex = MYsrcOffset + 3 * MYcol;
	ui MYdstIndex = MYdstOffset + 3 * MYcol;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	PixBuffer[MYtid3] = ImgSrc[MYsrcIndex];
	PixBuffer[MYtid3 + 1] = ImgSrc[MYsrcIndex + 1];
	PixBuffer[MYtid3 + 2] = ImgSrc[MYsrcIndex + 2];
	__syncthreads();
	ImgDst[MYdstIndex] = PixBuffer[MYtid3];
	ImgDst[MYdstIndex + 1] = PixBuffer[MYtid3 + 1];
	ImgDst[MYdstIndex + 2] = PixBuffer[MYtid3 + 2];
}


// Improved Vflip6() kernel that uses shared memory to copy 4 Bytes at a time (int).
// It no longer worries about the pixel RGB boundaries
__global__
void Vflip7(ui *ImgDst32, ui *ImgSrc32, ui Vpixels, ui RowInts)
{
	__shared__ ui PixBuffer[1024]; // holds 1024 int = 4096B

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcolIndex = MYbid*ThrPerBlk + MYtid;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowInts;
	ui MYdstOffset = MYmirrorrow * RowInts;
	ui MYsrcIndex = MYsrcOffset + MYcolIndex;
	ui MYdstIndex = MYdstOffset + MYcolIndex;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	PixBuffer[MYtid] = ImgSrc32[MYsrcIndex];
	__syncthreads();
	ImgDst32[MYdstIndex] = PixBuffer[MYtid];
}


// Improved Vflip7() kernel that uses shared memory to copy 8 Bytes at a time (2 int).
__global__
void Vflip8(ui *ImgDst32, ui *ImgSrc32, ui Vpixels, ui RowInts)
{
	__shared__ ui PixBuffer[2048]; // holds 2048 int = 8192B

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYtid2 = MYtid * 2;
	ui MYrow = blockIdx.y;
	ui MYcolIndex = (MYbid*ThrPerBlk + MYtid) * 2;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowInts;
	ui MYdstOffset = MYmirrorrow * RowInts;
	ui MYsrcIndex = MYsrcOffset + MYcolIndex;
	ui MYdstIndex = MYdstOffset + MYcolIndex;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	PixBuffer[MYtid2] = ImgSrc32[MYsrcIndex];
	if ((MYcolIndex+1) < RowInts) PixBuffer[MYtid2+1] = ImgSrc32[MYsrcIndex+1];
	__syncthreads();
	ImgDst32[MYdstIndex] = PixBuffer[MYtid2];
	if ((MYcolIndex + 1) < RowInts) ImgDst32[MYdstIndex+1] = PixBuffer[MYtid2+1];
}


// Modified Vflip8() kernel that uses Global Memory only 
// to copy 8 Bytes at a time (2 int). It does NOT use shared memory
__global__
void Vflip9(ui *ImgDst32, ui *ImgSrc32, ui Vpixels, ui RowInts)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcolIndex = (MYbid*ThrPerBlk + MYtid) * 2;
	if (MYcolIndex >= RowInts) return;			// index is out of range
	ui MYmirrorrow = Vpixels - 1 - MYrow;
	ui MYsrcOffset = MYrow       * RowInts;
	ui MYdstOffset = MYmirrorrow * RowInts;
	ui MYsrcIndex = MYsrcOffset + MYcolIndex;
	ui MYdstIndex = MYdstOffset + MYcolIndex;

	// swap pixels RGB   @MYrow , @MYmirrorrow
	ImgDst32[MYdstIndex] = ImgSrc32[MYsrcIndex];
	if ((MYcolIndex + 1) < RowInts) ImgDst32[MYdstIndex + 1] = ImgSrc32[MYsrcIndex+1];
}


// Kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst) (one Byte at a time)
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


// Improved PixCopy() that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
// Each thread copies 2 consecutive Bytes
__global__
void PixCopy2(uch *ImgDst, uch *ImgSrc, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	ui MYaddr = MYgtid * 2;

	if (MYaddr > FS) return;			// outside the allocated memory
	ImgDst[MYaddr]   = ImgSrc[MYaddr];		// copy pixel
	if ((MYaddr + 1) > FS) return;			// outside the allocated memory
	ImgDst[MYaddr + 1] = ImgSrc[MYaddr + 1];	// copy consecutive pixel
}


// Improved PixCopy() that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
// Each thread copies 4 consecutive Bytes
__global__
void PixCopy3(uch *ImgDst, uch *ImgSrc, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;
	ui MYaddr = MYgtid * 4;

	for (ui a=0;  a<4;  a++){
		if ((MYaddr+a) > FS) return;
		ImgDst[MYaddr+a] = ImgSrc[MYaddr+a];
	}
}


// Improved kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst)
// Uses shared memory as a temporary local buffer.
// copies one byte at a time
__global__
void PixCopy4(uch *ImgDst, uch *ImgSrc, ui FS)
{
	__shared__ uch PixBuffer[1024]; // Shared Memory: holds 1024 Bytes.

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui MYaddr = MYgtid * 4;
	if (MYgtid > FS) return;				// outside the allocated memory
	PixBuffer[MYtid] = ImgSrc[MYgtid];
	__syncthreads();
	ImgDst[MYgtid] = PixBuffer[MYtid];
}


// Improved kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst) {which must both be passed as integer pointers}
// Uses shared memory as a temporary local buffer.
// copies 4 bytes (32 bits) at a time
__global__
void PixCopy5(ui *ImgDst32, ui *ImgSrc32, ui FS)
{
	__shared__ ui PixBuffer[1024];	// Shared Mem: holds 1024 int (4096 Bytes)

	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui MYaddr = MYgtid * 4;
	if ((MYgtid*4) > FS) return;		// outside the allocated memory

	PixBuffer[MYtid] = ImgSrc32[MYgtid];
	__syncthreads();
	ImgDst32[MYgtid] = PixBuffer[MYtid];
}


// Improved kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst) {which must both be passed as integer pointers}
// This kernel does not use shared memory, only 32-bit pointers ...
// copies 4 bytes (32 bits) at a time
__global__
void PixCopy6(ui *ImgDst32, ui *ImgSrc32, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	//ui MYaddr = MYgtid * 4;
	if ((MYgtid * 4) > FS) return;		// outside the allocated memory

	ImgDst32[MYgtid] = ImgSrc32[MYgtid];
}


// Improved kernel that copies an image from one part of the
// GPU memory (ImgSrc) to another (ImgDst) {which must both be passed as integer pointers}
// This kernel does not use shared memory, only 32-bit pointers ...
// copies 8 bytes (2x32 bits) at a time
__global__
void PixCopy7(ui *ImgDst32, ui *ImgSrc32, ui FS)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYgtid = ThrPerBlk * MYbid + MYtid;

	if ((MYgtid * 4) > FS) return;		// outside the allocated memory
	ImgDst32[MYgtid] = ImgSrc32[MYgtid];
	MYgtid++;
	if ((MYgtid * 4) > FS) return;		// next 32 bits
	ImgDst32[MYgtid] = ImgSrc32[MYgtid];
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
	char			Flip = 'H';
	float			totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t		cudaStatus, cudaStatus2;
	cudaEvent_t		time1, time2, time3, time4;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow, BlkPerRowInt, BlkPerRowInt2;
	ui				ThrPerBlk = 256, NumBlocks, NB2, NB4, NB8, GPUDataTransfer;
	ui				RowBytes, RowInts;
	cudaDeviceProp	GPUprop;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	ui				*GPUCopyImg32, *GPUImg32;
	char			SupportedBlocks[100];
	int				KernelNum=1;
	char			KernelName[255];


	strcpy(ProgName, "imflipG");
	switch (argc){
	case 6:	 KernelNum = atoi(argv[5]);
	case 5:  ThrPerBlk=atoi(argv[4]);
	case 4:  Flip = toupper(argv[3][0]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename [V/H/C/T] [ThrPerBlk] [Kernel=1-9]", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp H", ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp V  128",ProgName);
			 printf("\n\nExample: %s Astronaut.bmp Output.bmp V  128 2", ProgName);
			 printf("\n\nH=horizontal flip, V=vertical flip, T=Transpose, C=copy image\n\n");
			 exit(EXIT_FAILURE);
	}
	if ((Flip != 'V') && (Flip != 'H') && (Flip != 'C') && (Flip != 'T')) {
		printf("Invalid flip option '%c'. Must be 'V','H', 'T', or 'C' ... \n", Flip);
		exit(EXIT_FAILURE);
	}
	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
		exit(EXIT_FAILURE);
	}
	if ((KernelNum < 1) || (KernelNum > 9)) {
		printf("Invalid kernel number %d ... \n", KernelNum);
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
	// These are the same pointers as GPUCopyImg and GPUImg, however, casted to an integer pointer
	GPUCopyImg32 = (ui *)GPUCopyImg;
	GPUImg32 = (ui *)GPUImg;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	RowBytes = (IPH * 3 + 3) & (~3);
	RowInts = RowBytes / 4;
	BlkPerRow = CEIL(IPH,ThrPerBlk);
	BlkPerRowInt = CEIL(RowInts, ThrPerBlk);
	BlkPerRowInt2 = CEIL(CEIL(RowInts,2), ThrPerBlk);
	NumBlocks = IPV*BlkPerRow; 
	dim3 dimGrid2D(BlkPerRow,		   ip.Vpixels);
	dim3 dimGrid2D2(CEIL(BlkPerRow,2), ip.Vpixels);
	dim3 dimGrid2D4(CEIL(BlkPerRow,4), ip.Vpixels);
	dim3 dimGrid2Dint(BlkPerRowInt,    ip.Vpixels);
	dim3 dimGrid2Dint2(BlkPerRowInt2,  ip.Vpixels);

	switch (Flip){
		case 'H': switch (KernelNum){
					case 1: Hflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
							strcpy(KernelName, "Hflip : Each thread copies 1 pixel. Computes everything.");
							break;
					case 2: Hflip2 <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, BlkPerRow, RowBytes);
							strcpy(KernelName, "Hflip2 : Each thread copies 1 pixel. Uses pre-computed values.");
							break;
					case 3: Hflip3 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							strcpy(KernelName, "Hflip3 : Each therad copies 1 pixel (using a 2D grid)");
							break;
					case 4: Hflip4 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							strcpy(KernelName, "Hflip4 : Each therad copies 2 consecutive pixels");
							break;
					case 5: Hflip5 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							strcpy(KernelName, "Hflip5 : Each therad copies 4 consecutive pixels");
							break;
					case 6: Hflip6 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							strcpy(KernelName, "Hflip6 : Uses Shared Memory to copy one pixel at a time");
							break;
					case 7: Hflip7 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, RowInts);
							strcpy(KernelName, "Hflip7 : Flips 4 pixels (12B) at a time inside Shared Mem");
							break;
					case 8: Hflip8 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, RowInts);
							strcpy(KernelName, "Hflip8 : Flips 4 pixels (12B) using only registers");
							break;
					default:printf("...... Kernel Number=%d ... NOT IMPLEMENTED .... \n", KernelNum);
							strcpy(KernelName, "*** NOT IMPLEMENTED ***");
							break;
				  }
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2*IMAGESIZE;
				  break;
		case 'V': switch (KernelNum){
					case 1: Vflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV);
							strcpy(KernelName, "Vflip : Each thread copies 1 pixel. Computes everything.");
							break;
					case 2: Vflip2 <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, BlkPerRow, RowBytes);
							strcpy(KernelName, "Vflip2 : Each thread copies 1 pixel. Uses pre-computed values.");
							break;
					case 3: Vflip3 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Vflip3 : Each therad copies 1 pixel (using a 2D grid)");
							break;
					case 4: Vflip4 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Vflip4 : Each therad copies 2 consecutive pixels");
							break;
					case 5: Vflip5 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Vflip5 : Each therad copies 4 consecutive pixels");
							break;
					case 6: Vflip6 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Vflip6 : Uses Shared Memory to copy one pixel at a time");
							break;
					case 7: Vflip7 <<< dimGrid2Dint, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPV, RowInts);
							strcpy(KernelName, "Vflip7 : Uses Shared Memory to copy 1 int at a time");
							break;
					case 8: Vflip8 <<< dimGrid2Dint2, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPV, RowInts);
							strcpy(KernelName, "Vflip8 : Uses Shared Memory to copy 2 int at a time");
							break;
					case 9: Vflip9 <<< dimGrid2Dint2, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPV, RowInts);
							strcpy(KernelName, "Vflip9 : Uses only Global Memory to copy 2 int at a time");
							break;
					default:printf("...... Kernel Number=%d ... NOT IMPLEMENTED .... \n", KernelNum);
							strcpy(KernelName, "*** NOT IMPLEMENTED ***");
							break;
				  }
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2 * IMAGESIZE;
				  break;
		case 'T': switch (KernelNum){
					case 1: Hflip <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH);
							Vflip <<< NumBlocks, ThrPerBlk >>> (GPUImg, GPUCopyImg, IPH, IPV);
							strcpy(KernelName, "Hflip<<<  >>>()  ,  Vflip<<<  >>>()");
							break;
					case 2: Hflip2 <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, BlkPerRow, RowBytes);
							Vflip2 <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, BlkPerRow, RowBytes);
							strcpy(KernelName, "Hflip2<<<  >>>()  ,  Vflip2<<<  >>>()");
							break;
					case 3: Hflip3 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							Vflip3 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Hflip3<<<  >>>()  ,  Vflip3<<<  >>>()");
							break;
					case 4: Hflip4 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							Vflip4 <<< dimGrid2D2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Hflip4<<<  >>>()  ,  Vflip4<<<  >>>()");
							break;
					case 5: Hflip5 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							Vflip5 <<< dimGrid2D4, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Hflip5<<<  >>>()  ,  Vflip5<<<  >>>()");
							break;
					case 6: Hflip6 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes);
							Vflip6 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, IPV, RowBytes);
							strcpy(KernelName, "Hflip6<<<  >>>()  ,  Vflip6<<<  >>>()");
							break;
					default:printf("...... Kernel Number=%d ... NOT IMPLEMENTED .... \n", KernelNum);
							strcpy(KernelName, "*** NOT IMPLEMENTED ***");
							break;
				  }
				  GPUResult = GPUImg;
				  GPUDataTransfer = 4 * IMAGESIZE;
				  break;
		case 'C': NumBlocks = CEIL(IMAGESIZE,ThrPerBlk);
				  NB2 = CEIL(NumBlocks,2);
				  NB4 = CEIL(NumBlocks,4);
				  NB8 = CEIL(NumBlocks,8);
				  switch (KernelNum){
					case 1: PixCopy <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IMAGESIZE);
							strcpy(KernelName, "PixCopy : Each kernel copies one Byte at a time");
							break;
					case 2: PixCopy2 <<< NB2, ThrPerBlk >>> (GPUCopyImg, GPUImg, IMAGESIZE);
							strcpy(KernelName, "PixCopy2 : Each kernel copies 2 consecutive Bytes at a time");
							break;
					case 3: PixCopy3 <<< NB4, ThrPerBlk >>> (GPUCopyImg, GPUImg, IMAGESIZE);
							strcpy(KernelName, "PixCopy3 : Each kernel copies 4 consecutive Bytes at a time");
							break;
					case 4: PixCopy4 <<< NumBlocks, ThrPerBlk >>> (GPUCopyImg, GPUImg, IMAGESIZE);
							strcpy(KernelName, "PixCopy4 : Uses Shared Memory to copy one Byte at a time");
							break;
					case 5: PixCopy5 <<< NB4, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IMAGESIZE);
							strcpy(KernelName, "PixCopy5 : Uses Shared Memory to copy one int (32b) at a time");
							break;
					case 6: PixCopy6 <<< NB4, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IMAGESIZE);
							strcpy(KernelName, "PixCopy6 : Uses only Global Memory to copy one int (32b) at a time");
							break;
					case 7: PixCopy7 <<< NB8, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IMAGESIZE);
							strcpy(KernelName, "PixCopy7 : Uses only Global Memory to copy two int (8 Bytes) at a time");
							break;	
					default:printf("...... Kernel Number=%d ... NOT IMPLEMENTED .... \n",KernelNum);
							strcpy(KernelName, "*** NOT IMPLEMENTED ***");
						break;
				  }
				  GPUResult = GPUCopyImg;
				  GPUDataTransfer = 2 * IMAGESIZE;
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
	printf("\n--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n",
		GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");
	printf("%s %s %s %c %u %u  [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, Flip, ThrPerBlk, KernelNum, NumBlocks, BlkPerRow);
	printf("--------------------------------------------------------------------------\n");
	printf("%s\n",KernelName);
	printf("--------------------------------------------------------------------------\n");
	printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
	printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
	printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU)); 
	printf("--------------------------------------------------------------------------\n");
	printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2*IMAGESIZE+GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
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



