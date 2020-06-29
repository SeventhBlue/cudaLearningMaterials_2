#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "ImageStuff.h"


double** CreateBlankDouble()
{
    int i;

	double** img = (double **)malloc(ip.Vpixels * sizeof(double*)); 
    for(i=0; i<ip.Vpixels; i++){
		img[i] = (double *)malloc(ip.Hpixels*sizeof(double));
		memset((void *)img[i],0,(size_t)ip.Hpixels*sizeof(double));
    }
    return img;
}


double** CreateBWCopy(unsigned char** img)
{
    int i,j,k;

	double** imgBW = (double **)malloc(ip.Vpixels * sizeof(double*)); 
    for(i=0; i<ip.Vpixels; i++){
		imgBW[i] = (double *)malloc(ip.Hpixels*sizeof(double));
		for(j=0; j<ip.Hpixels; j++){
			// convert each pixel to B&W = (R+G+B)/3
			k=3*j;
			imgBW[i][j]=((double)img[i][k]+(double)img[i][k+1]+(double)img[i][k+2])/3.0;  
		}
    }	
    return imgBW;
}


unsigned char** CreateBlankBMP(unsigned char FILL)
{
    int i,j;

	unsigned char** img = (unsigned char **)malloc(ip.Vpixels * sizeof(unsigned char*));
    for(i=0; i<ip.Vpixels; i++){
        img[i] = (unsigned char *)malloc(ip.Hbytes * sizeof(unsigned char));
		memset((void *)img[i],FILL,(size_t)ip.Hbytes); // zero out every pixel
    }
    return img;
}


// This thread function asynchronously pre-calculates a row of pixels.
// It uses the CounterMutex to updated the shared counter-based variables
pthread_mutex_t		CounterMutex;
struct PrPixel 		**PrIm;
int					NextRowToProcess, LastRowRead;
int					ThreadCtr[MAXTHREADS];  // Counts # rows processed by each thread


void *AMTPreCalcRow(void* ThCtr)
{
	unsigned char r, g, b;
	int i,j,Last;
	float R, G, B, BW, BW2, BW3, BW4, BW5, BW9, BW12, Z=0.0;

	do{
		// get the next row number safely
		pthread_mutex_lock(&CounterMutex);
			Last=LastRowRead;  
			i=NextRowToProcess;  
			if(Last>=i){
				NextRowToProcess++;
				j = *((int *)ThCtr);
				*((int *)ThCtr) = j+1; // One more row processed by this thread
			}
		pthread_mutex_unlock(&CounterMutex);
		if(Last<i) continue;
		if(i>=ip.Vpixels) break;
		for(j=0; j<ip.Hpixels; j++){
			b=PrIm[i][j].B;			B=(float)b;
			g=PrIm[i][j].G; 		G=(float)g;
			r=PrIm[i][j].R;			R=(float)r;
			BW3=R+G+B;
			PrIm[i][j].BW   = BW   = BW3*0.3333333;
			PrIm[i][j].BW2  = BW2  = BW+BW;
			PrIm[i][j].BW4  = BW4  = BW2+BW2;
			PrIm[i][j].BW5  = BW5  = BW4+BW;
			PrIm[i][j].BW9  = BW9  = BW5+BW4;
			PrIm[i][j].BW12 = BW12 = BW9+BW3;
			PrIm[i][j].BW15 = BW12+BW3;
			PrIm[i][j].Gauss = PrIm[i][j].Gauss2 = Z;
			PrIm[i][j].Theta = PrIm[i][j].Gradient = Z;
		}
	}while(i<ip.Vpixels);
	pthread_exit(NULL);
}


// This function calculates the pre-processed pixels while reading from disk
// It does it in an "asynhcorous Multi-threaded (AMT)" fashion using a Mutex
struct PrPixel** PrAMTReadBMP(char* filename)
{
	int i,j,k,ThErr;
	unsigned char Buffer[24576];
	pthread_t ThHan[MAXTHREADS];
	pthread_attr_t attr;
	
	FILE* f = fopen(filename, "rb");
	if(f == NULL){
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}
	unsigned char HeaderInfo[54];
	fread(HeaderInfo, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width*3 + 3) & (~3);		ip.Hbytes  = RowBytes;
	//copy header for re-use
	for(i=0; i<54; i++) {	ip.HeaderInfo[i] = HeaderInfo[i];	}
	printf("\n   Input BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);
	// allocate memory to store the main image
	PrIm = (struct PrPixel **)malloc(height * sizeof(struct PrPixel *));
	for(i=0; i<height; i++) {
		PrIm[i] = (struct PrPixel *)malloc(width * sizeof(struct PrPixel));
	}
	pthread_attr_init(&attr);		// Initialize threads and MUTEX
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_mutex_init(&CounterMutex, NULL);
	
	pthread_mutex_lock(&CounterMutex);
		NextRowToProcess=0; // Set the asynchronous row counter to 0
		LastRowRead=-1;
		for(i=0; i<PrThreads; i++) ThreadCtr[i]=0; // zero every thread counter
	pthread_mutex_unlock(&CounterMutex);
	// read the image from disk and pre-calculate the PRImage pixels
	for(i = 0; i<height; i++) {
		if(i==20){   // when sufficient # of rows are read, launch threads
			// PrThreads is the number of pre-processing threads
			for(j=0; j<PrThreads; j++){
				ThErr=pthread_create(&ThHan[j], &attr, AMTPreCalcRow, (void *)&ThreadCtr[j]);
				if(ThErr != 0){
					printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
					exit(EXIT_FAILURE);
				}
			}
		}
		fread(Buffer, sizeof(unsigned char), RowBytes, f);
		for(j=0,k=0; j<width; j++, k+=3){
			PrIm[i][j].B=Buffer[k];
			PrIm[i][j].G=Buffer[k+1];
			PrIm[i][j].R=Buffer[k+2];
		}
		pthread_mutex_lock(&CounterMutex);
			LastRowRead=i;
		pthread_mutex_unlock(&CounterMutex);
	}		
	// wait for all threads to be done	
	for(i=0; i<PrThreads; i++){	pthread_join(ThHan[i], NULL);	}
	pthread_attr_destroy(&attr);
	pthread_mutex_destroy(&CounterMutex);
	fclose(f);
	return PrIm;  // return the pointer to the main image
}


// This function calculates the pre-processed pixels while reading from disk
struct PrPixel** PrReadBMP(char* filename)
{
	int i,j,k;
	unsigned char r, g, b;
	float R, G, B, BW, BW2, BW3, BW4, BW5, BW9, BW12, Z=0.0;
	unsigned char Buffer[24576];
	FILE* f = fopen(filename, "rb");
	if(f == NULL){
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}
	unsigned char HeaderInfo[54];
	fread(HeaderInfo, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width*3 + 3) & (~3);		ip.Hbytes  = RowBytes;
	//copy header for re-use
	for(i=0; i<54; i++) {	ip.HeaderInfo[i] = HeaderInfo[i];	}
	printf("\n   Input BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);
	// allocate memory to store the main image
	struct PrPixel **PrIm = (struct PrPixel **)malloc(height * sizeof(struct PrPixel *));
	for(i=0; i<height; i++) {
		PrIm[i] = (struct PrPixel *)malloc(width * sizeof(struct PrPixel));
	}
	// read the image from disk and pre-calculate the PRImage pixels
	for(i = 0; i < height; i++) {
		fread(Buffer, sizeof(unsigned char), RowBytes, f);
		for(j=0,k=0; j<width; j++, k+=3){
			b=PrIm[i][j].B=Buffer[k];		B=(float)b;
			g=PrIm[i][j].G=Buffer[k+1];		G=(float)g;
			r=PrIm[i][j].R=Buffer[k+2];		R=(float)r;
			BW3=R+G+B;
			PrIm[i][j].BW   = BW   = BW3*0.3333333;
			PrIm[i][j].BW2  = BW2  = BW+BW;
			PrIm[i][j].BW4  = BW4  = BW2+BW2;
			PrIm[i][j].BW5  = BW5  = BW4+BW;
			PrIm[i][j].BW9  = BW9  = BW5+BW4;
			PrIm[i][j].BW12 = BW12 = BW9+BW3;
			PrIm[i][j].BW15 = BW12+BW3;
			PrIm[i][j].Gauss = PrIm[i][j].Gauss2 = Z;
			PrIm[i][j].Theta = PrIm[i][j].Gradient = Z;
		}
	}
	fclose(f);
	return PrIm;  // return the pointer to the main image
}


unsigned char** ReadBMP(char* filename)
{
	int i;
	FILE* f = fopen(filename, "rb");
	if(f == NULL){
		printf("\n\n%s NOT FOUND\n\n",filename);
		exit(1);
	}

	unsigned char HeaderInfo[54];
	fread(HeaderInfo, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width*3 + 3) & (~3);		ip.Hbytes  = RowBytes;
	//copy header for re-use
	for(i=0; i<54; i++) {	ip.HeaderInfo[i] = HeaderInfo[i];	}
	printf("\n   Input BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);
	// allocate memory to store the main image
	unsigned char **Img = (unsigned char **)malloc(height * sizeof(unsigned char*));
	for(i=0; i<height; i++) {
		Img[i] = (unsigned char *)malloc(RowBytes * sizeof(unsigned char));
	}
	// read the image from disk
	for(i = 0; i < height; i++) {
		fread(Img[i], sizeof(unsigned char), RowBytes, f);
	}
	fclose(f);
	return Img;  // return the pointer to the main image
}


void WriteBMP(unsigned char** img, char* filename)
{
	int i;
	FILE* f = fopen(filename, "wb");
	if(f == NULL){
		printf("\n\nFILE CREATION ERROR: %s\n\n",filename);
		exit(1);
	}
	//write header
	for(i=0; i<54; i++) {	fputc(ip.HeaderInfo[i],f);	}

	//write data
	for(i=0; i<ip.Vpixels; i++) {
		fwrite(img[i], sizeof(unsigned char), ip.Hbytes, f);
	}
	printf("\n  Output BMP File name: %20s  (%u x %u)",filename,ip.Hpixels,ip.Vpixels);
	fclose(f);
}
