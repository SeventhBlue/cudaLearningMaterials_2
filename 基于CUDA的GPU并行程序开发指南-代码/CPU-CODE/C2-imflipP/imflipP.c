#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include "ImageStuff.h"

#define REPS 	     1
#define MAXTHREADS   128

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes
void (*FlipFunc)(unsigned char** img);		// Function pointer to flip the image
void* (*MTFlipFunc)(void *arg);				// Function pointer to flip the image, multi-threaded version

unsigned char**	TheImage;					// This is the main image
struct ImgProp 	ip;


void FlipImageV(unsigned char** img)
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
}


void FlipImageH(unsigned char** img)
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
}


void *MTFlipV(void* tid)
{
	struct Pixel pix; //temp swap pixel
	int row, col;

	long ts = *((int *) tid);       	// My thread ID is stored here
	ts *= ip.Hbytes/NumThreads;			// start index
	long te = ts+ip.Hbytes/NumThreads-1; 	// end index

	for(col=ts; col<=te; col+=3)
	{
		row=0;
		while(row<ip.Vpixels/2)
		{
			pix.B = TheImage[row][col];
			pix.G = TheImage[row][col+1];
			pix.R = TheImage[row][col+2];

			TheImage[row][col]   = TheImage[ip.Vpixels-(row+1)][col];
			TheImage[row][col+1] = TheImage[ip.Vpixels-(row+1)][col+1];
			TheImage[row][col+2] = TheImage[ip.Vpixels-(row+1)][col+2];

			TheImage[ip.Vpixels-(row+1)][col]   = pix.B;
			TheImage[ip.Vpixels-(row+1)][col+1] = pix.G;
			TheImage[ip.Vpixels-(row+1)][col+2] = pix.R;

			row++;
		}
	}
	pthread_exit(0);
}


void *MTFlipH(void* tid)
{
	struct Pixel pix; //temp swap pixel
	int row, col;

	long ts = *((int *) tid);       	// My thread ID is stored here
	ts *= ip.Vpixels/NumThreads;			// start index
	long te = ts+ip.Vpixels/NumThreads-1; 	// end index

	for(row=ts; row<=te; row++)
	{
		col=0;
		while(col<ip.Hpixels*3/2)
		{
			pix.B = TheImage[row][col];
			pix.G = TheImage[row][col+1];
			pix.R = TheImage[row][col+2];

			TheImage[row][col]   = TheImage[row][ip.Hpixels*3-(col+3)];
			TheImage[row][col+1] = TheImage[row][ip.Hpixels*3-(col+2)];
			TheImage[row][col+2] = TheImage[row][ip.Hpixels*3-(col+1)];

			TheImage[row][ip.Hpixels*3-(col+3)] = pix.B;
			TheImage[row][ip.Hpixels*3-(col+2)] = pix.G;
			TheImage[row][ip.Hpixels*3-(col+1)] = pix.R;

			col+=3;
		}
	}
	pthread_exit(NULL);
}


int main(int argc, char** argv)
{
	char 				Flip;
	int 				a,i,ThErr;
	struct timeval 		t;
	double         		StartTime, EndTime;
	double         		TimeElapsed;

	switch (argc){
		case 3 : NumThreads=1; 				Flip = 'V';						break;
		case 4 : NumThreads=1;  			Flip = toupper(argv[3][0]);		break;
		case 5 : NumThreads=atoi(argv[4]);  Flip = toupper(argv[3][0]);		break;
		default: printf("\n\nUsage: imflipP input output [v/h] [thread count]");
		printf("\n\nExample: imflipP infilename.bmp outname.bmp h 8\n\n");
		return 0;
	}
	if((Flip != 'V') && (Flip != 'H')) {
		printf("Flip option '%c' is invalid. Can only be 'V' or 'H' ... Exiting abruptly ...\n",Flip);
		exit(EXIT_FAILURE);
	}

	if((NumThreads<1) || (NumThreads>MAXTHREADS)){
		printf("\nNumber of threads must be between 1 and %u... Exiting abruptly\n",MAXTHREADS);
		exit(EXIT_FAILURE);
	}
	else{
		if(NumThreads != 1){
			printf("\nExecuting the multi-threaded version with %li threads ...\n",NumThreads);
			MTFlipFunc = (Flip=='V') ? MTFlipV:MTFlipH;
		}
		else{
			printf("\nExecuting the serial version ...\n");
			FlipFunc = (Flip=='V') ? FlipImageV:FlipImageH;
		}
	}

	TheImage = ReadBMP(argv[1]);

	gettimeofday(&t, NULL);
	StartTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);

	if(NumThreads >1){
		pthread_attr_init(&ThAttr);
		pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);
		for(a=0; a<REPS; a++){
			for(i=0; i<NumThreads; i++){
				ThParam[i] = i;
				ThErr = pthread_create(&ThHandle[i], &ThAttr, MTFlipFunc, (void *)&ThParam[i]);
				if(ThErr != 0){
					printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
					exit(EXIT_FAILURE);
				}
			}
			pthread_attr_destroy(&ThAttr);
			for(i=0; i<NumThreads; i++){
				pthread_join(ThHandle[i], NULL);
			}
		}
	}else{
		for(a=0; a<REPS; a++){
			(*FlipFunc)(TheImage);
		}
	}

	gettimeofday(&t, NULL);
	EndTime = (double)t.tv_sec*1000000.0 + ((double)t.tv_usec);
	TimeElapsed=(EndTime-StartTime)/1000.00;
	TimeElapsed/=(double)REPS;

	//merge with header and write to file
	WriteBMP(TheImage, argv[2]);

	// free() the allocated memory for the image
	for(i = 0; i < ip.Vpixels; i++) { free(TheImage[i]); }
	free(TheImage);

	printf("\n\nTotal execution time: %9.4f ms (%s flip)",TimeElapsed, Flip=='V'?"Vertical":"Horizontal");
	printf(" (%6.3f ns/pixel)\n", 1000000*TimeElapsed/(double)(ip.Hpixels*ip.Vpixels));

	return (EXIT_SUCCESS);
}
