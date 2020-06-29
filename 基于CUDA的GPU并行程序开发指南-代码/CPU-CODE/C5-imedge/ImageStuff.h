#define EDGE		 0
#define NOEDGE       255
#define MAXTHREADS   128

struct ImgProp{
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
};

struct Pixel{
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

struct PrPixel{
	unsigned char 	R;
	unsigned char 	G;
	unsigned char 	B;
	unsigned char 	x;    // unused. to make it an even 4B
	float 			BW;
	float 			BW2,BW4,BW5,BW9,BW12,BW15;
	float			Gauss, Gauss2;
	float			Theta,Gradient;
};

double** CreateBWCopy(unsigned char** img);
double** CreateBlankDouble();
unsigned char** CreateBlankBMP(unsigned char FILL);
struct PrPixel** PrAMTReadBMP(char*);
struct PrPixel** PrReadBMP(char*);
unsigned char** ReadBMP(char*);
void WriteBMP(unsigned char** , char*);

extern struct	ImgProp 	ip;
extern long 	NumThreads, PrThreads;
extern int		ThreadCtr[];
