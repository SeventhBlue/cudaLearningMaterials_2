#include <pthread.h>
#include <stdint.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "ImageStuff.h"

#define MAXTHREADS   128
#define PI			 3.1415926
#define EDGE		 0
#define NOEDGE       255

long  			NumThreads;         		// Total number of threads working in parallel
int 	     	ThParam[MAXTHREADS];		// Thread parameters ...
int				ThreshLo,ThreshHi;			// "Edge" vs. "No Edge" thresholds
pthread_t      	ThHandle[MAXTHREADS];		// Thread handles
pthread_attr_t 	ThAttr;						// Pthread attrributes

unsigned char**	TheImage;					// This is the main image
unsigned char**	CopyImage;					// This is the copy image (to store edges)
double			**BWImage;					// B&W of TheImage (each pixel=double)
double			**GaussImage;				// Gauss filtered version of the B&W image
double			**Gradient, **Theta;		// gradient and theta for each pixel
struct ImgProp 	ip;

//Sobel kernels
double Gx[3][3] = {		{ -1, 0, 1 },
						{ -2, 0, 2 },
						{ -1, 0, 1 }	};

double Gy[3][3] = {		{ -1, -2, -1 },
						{  0,  0,  0 },
						{  1,  2,  1 }	};

double Gauss[5][5] = {	{ 2, 4,  5,  4,  2 },
						{ 4, 9,  12, 9,  4 },
						{ 5, 12, 15, 12, 5 },
						{ 4, 9,  12, 9,  4 },
						{ 2, 4,  5,  4,  2 }	};


						// Function that takes BWImage and calculates the Gaussian filtered version
// Saves the result in the GaussFilter[][] array
void *GaussianFilter(void* tid)
{
    long tn;            // My thread number (ID) is stored here
    int row,col,i,j;
	double G;  			// temp to calculate the Gaussian filtered version

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;

    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<2) || (row>(ip.Vpixels-3))) continue;
        col=2;
        while(col<=(ip.Hpixels-3)){
			G=0.0;
			for(i=-2; i<=2; i++){
				for(j=-2; j<=2; j++){
					G+=BWImage[row+i][col+j]*Gauss[i+2][j+2];
				}
			}
			GaussImage[row][col]=G/159.00D;
            col++;
        }
    }
    pthread_exit(NULL);
}


// Function that calculates the Gradient and Theta for each pixel
// Takes the Gauss[][] array and creates the Gradient[][] and Theta[][] arrays
void *Sobel(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col,i,j;
	double GX,GY;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;

    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<1) || (row>(ip.Vpixels-2))) continue;
        col=1;
        while(col<=(ip.Hpixels-2)){
			// calculate Gx and Gy
			GX=0.0; GY=0.0;
			for(i=-1; i<=1; i++){
				for(j=-1; j<=1; j++){
					GX+=GaussImage[row+i][col+j]*Gx[i+1][j+1];
					GY+=GaussImage[row+i][col+j]*Gy[i+1][j+1];
				}
			}
			Gradient[row][col]=sqrt(GX*GX+GY*GY);
			Theta[row][col]=atan(GX/GY)*180.0/PI;
            col++;
        }
    }
    pthread_exit(NULL);
}

void *Threshold(void* tid)
{
    long tn;            		     // My thread number (ID) is stored here
    int row,col;
	unsigned char PIXVAL;
	double L,H,G,T;

    tn = *((int *) tid);           // Calculate my Thread ID
    tn *= ip.Vpixels/NumThreads;

    for(row=tn; row<tn+ip.Vpixels/NumThreads; row++)
    {
		if((row<1) || (row>(ip.Vpixels-2))) continue;
        col=1;
        while(col<=(ip.Hpixels-2)){
			L=(double)ThreshLo;		H=(double)ThreshHi;
			G=Gradient[row][col];
			PIXVAL=NOEDGE;
			if(G<=L){						// no edge
				PIXVAL=NOEDGE;
			}else if(G>=H){					// edge
				PIXVAL=EDGE;
			}else{
				T=Theta[row][col];
				if((T<-67.5) || (T>67.5)){   
					// Look at left and right
					PIXVAL=((Gradient[row][col-1]>H) || (Gradient[row][col+1]>H)) ? EDGE:NOEDGE;
				}else if((T>=-22.5) && (T<=22.5)){  
					// Look at top and bottom
					PIXVAL=((Gradient[row-1][col]>H) || (Gradient[row+1][col]>H)) ? EDGE:NOEDGE;
				}else if((T>22.5) && (T<=67.5)){   
					// Look at upper right, lower left
					PIXVAL=((Gradient[row-1][col+1]>H) || (Gradient[row+1][col-1]>H)) ? EDGE:NOEDGE;
				}else if((T>=-67.5) && (T<-22.5)){   
					// Look at upper left, lower right
					PIXVAL=((Gradient[row-1][col-1]>H) || (Gradient[row+1][col+1]>H)) ? EDGE:NOEDGE;
				}
			}
			CopyImage[row][col*3]=PIXVAL;
			CopyImage[row][col*3+1]=PIXVAL;
			CopyImage[row][col*3+2]=PIXVAL;
            col++;
        }
    }
    pthread_exit(NULL);
}


// returns the time stamps in ms
double GetDoubleTime()
{
    struct timeval 		tnow;

    gettimeofday(&tnow, NULL);
    return ((double)tnow.tv_sec*1000000.0 + ((double)tnow.tv_usec))/1000.00;
}


double ReportTimeDelta(double PreviousTime, char *Message)
{
	double	Tnow,TimeDelta;
	
	Tnow=GetDoubleTime();
	TimeDelta=Tnow-PreviousTime;
	printf("\n.....%-30s ... %7.0f ms\n",Message,TimeDelta);
	return Tnow;
}


int main(int argc, char** argv)
{
    int 		a,i,ThErr;
    double		t1,t2,t3,t4,t5,t6,t7,t8;
	
    switch (argc){
		case 3 : NumThreads=1;             ThreshLo=50; 	       ThreshHi=100;	break;
		case 4 : NumThreads=atoi(argv[3]); ThreshLo=50; 	       ThreshHi=100;	break;
		case 5 : NumThreads=atoi(argv[3]); ThreshLo=atoi(argv[4]); ThreshHi=100;	break;
		case 6 : NumThreads=atoi(argv[3]); ThreshLo=atoi(argv[4]); ThreshHi=atoi(argv[5]); break;
		default: printf("\n\nUsage: imedge infile outfile [Threads] [ThreshLo] [ThreshHi]");
				 printf("\n\nExample: imedge in.bmp out.bmp 8\n\n");
				 printf("\n\nExample: imedge in.bmp out.bmp 4 50 150\n\n");
				 printf("\n\nNothing executed ... Exiting ...\n\n");
				exit(EXIT_FAILURE);
    }
	if((NumThreads<1) || (NumThreads>MAXTHREADS)){
            printf("\nNumber of threads must be between 1 and %u... \n",MAXTHREADS);
            printf("\n'1' means Pthreads version with a single thread\n");
			 printf("\n\nNothing executed ... Exiting ...\n\n");
            exit(EXIT_FAILURE);
	}
	if((ThreshLo<0) || (ThreshHi>255) || (ThreshLo>ThreshHi)){
        printf("\nInvalid Thresholds: Threshold must be between [0...255] ...\n");
		printf("\n\nNothing executed ... Exiting ...\n\n");
        exit(EXIT_FAILURE);
	}else{
		printf("ThresLo=%d ... ThreadHi=%d\n",ThreshLo,ThreshHi);	
	}
	printf("\nExecuting the Pthreads version with %li threads ...\n",NumThreads);
	t1 = GetDoubleTime();
	TheImage=ReadBMP(argv[1]);  	  printf("\n");
	
    t2 = ReportTimeDelta(t1,"ReadBMP complete");	// Start time without IO

	CopyImage = CreateBlankBMP(NOEDGE);		// This will store the edges in RGB  
	BWImage    = CreateBWCopy(TheImage);
	GaussImage = CreateBlankDouble();
	Gradient = CreateBlankDouble();
	Theta    = CreateBlankDouble();
	t3=ReportTimeDelta(t2, "Auxiliary images created");
	
	pthread_attr_init(&ThAttr);
	pthread_attr_setdetachstate(&ThAttr, PTHREAD_CREATE_JOINABLE);

	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, GaussianFilter, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	t4=ReportTimeDelta(t3, "Gauss Image created");
	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, Sobel, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	t5=ReportTimeDelta(t4, "Gradient, Theta calculated");
	for(i=0; i<NumThreads; i++){
		ThParam[i] = i;
		ThErr = pthread_create(&ThHandle[i], &ThAttr, Threshold, (void *)&ThParam[i]);
		if(ThErr != 0){
			printf("\nThread Creation Error %d. Exiting abruptly... \n",ThErr);
			exit(EXIT_FAILURE);
		}
	}
	pthread_attr_destroy(&ThAttr);
	for(i=0; i<NumThreads; i++){
		pthread_join(ThHandle[i], NULL);
	}
	t6=ReportTimeDelta(t5, "Thresholding completed");
	
    //merge with header and write to file
    WriteBMP(CopyImage, argv[2]);  printf("\n");
	t7=ReportTimeDelta(t6, "WriteBMP completed");

	// free() the allocated area for image and pointers
	for(i = 0; i < ip.Vpixels; i++) { 
		free(TheImage[i]);   free(CopyImage[i]); free(BWImage[i]); 
		free(GaussImage[i]); free(Gradient[i]);  free(Theta[i]); 
	}
	free(TheImage);  	free(CopyImage);  	free(BWImage);  
	free(GaussImage);  free(Gradient); 	 	free(Theta);
    
    t8=ReportTimeDelta(t2, "Program Runtime without IO");
	
    return (EXIT_SUCCESS);
}
