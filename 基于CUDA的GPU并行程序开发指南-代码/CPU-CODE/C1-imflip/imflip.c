#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ImageStuff.h"

#define REPS 	129

struct ImgProp ip;

unsigned char** FlipImageV(unsigned char** img)
{
	struct Pixel pix; //temp swap pixel
	int row, col;

	//vertical flip
	for(col=0; col<ip.Hbytes; col+=3)
	{
		row = 0;
		while(row<ip.Vpixels/2)
		{
			pix.B = img[row][col];
			pix.G = img[row][col+1];
			pix.R = img[row][col+2];

			img[row][col]   = img[ip.Vpixels-(row+1)][col];
			img[row][col+1] = img[ip.Vpixels-(row+1)][col+1];
			img[row][col+2] = img[ip.Vpixels-(row+1)][col+2];

			img[ip.Vpixels-(row+1)][col]   = pix.B;
			img[ip.Vpixels-(row+1)][col+1] = pix.G;
			img[ip.Vpixels-(row+1)][col+2] = pix.R;

			row++;
		}
	}
	return img;
}


unsigned char** FlipImageH(unsigned char** img)
{
	struct Pixel pix; //temp swap pixel
	int row, col;

	//horizontal flip
	for(row=0; row<ip.Vpixels; row++)
	{
		col = 0;
		while(col<(ip.Hpixels*3)/2)
		{
			pix.B = img[row][col];
			pix.G = img[row][col+1];
			pix.R = img[row][col+2];

			img[row][col]   = img[row][ip.Hpixels*3-(col+3)];
			img[row][col+1] = img[row][ip.Hpixels*3-(col+2)];
			img[row][col+2] = img[row][ip.Hpixels*3-(col+1)];

			img[row][ip.Hpixels*3-(col+3)] = pix.B;
			img[row][ip.Hpixels*3-(col+2)] = pix.G;
			img[row][ip.Hpixels*3-(col+1)] = pix.R;

			col+=3;
		}
	}
	return img;
}


int main(int argc, char** argv)
{
	if(argc != 4)
	{
		printf("\n\nUsage: imflip [input] [output] [V | H]");
		printf("\n\nExample: imflip square.bmp square_h.bmp h\n\n");
		return 0;
	}

	unsigned char** data = ReadBMP(argv[1]);
	double timer;
	unsigned int a;
	clock_t start,stop;

	start = clock();
	switch (argv[3][0]){
		case 'v' :
		case 'V' : for(a=0; a<REPS; a++) data = FlipImageV(data); break;
		case 'h' : 
		case 'H' : for(a=0; a<REPS; a++) data = FlipImageH(data); break;
		default  : printf("\nINVALID OPTION\n"); return 0;
	}
	stop = clock();
	timer = 1000*((double)(stop-start))/(double)CLOCKS_PER_SEC/(double)REPS;

	//merge with header and write to file
	WriteBMP(data, argv[2]);

	// free() the allocated memory for the image
	for(int i = 0; i < ip.Vpixels; i++) { free(data[i]); }
	free(data);

	printf("\n\nTotal execution time: %9.4f ms",timer);
	printf(" (%7.3f ns per pixel)\n", 1000000*timer/(double)(ip.Hpixels*ip.Vpixels));

	return 0;
}
