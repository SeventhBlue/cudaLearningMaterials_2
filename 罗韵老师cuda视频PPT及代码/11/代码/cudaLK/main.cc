#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "cudaLK.h"
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <sys/time.h>

using namespace std;
using namespace cv;

void openCVLK(IplImage *img1, IplImage *img2);

int main(int argc, char **argv)
{
    if(argc != 3) {
        printf("./cudaLK img1.png img2.png\n");
        return 0;
    }

    IplImage *img1 = cvLoadImage(argv[1]);
    IplImage *img2 = cvLoadImage(argv[2]);

    if(!img1 || !img2) {
        printf("Error loading images\n");
        return 1;
    }

    cudaLK LK;
  
    LK.run((unsigned char*)img1->imageData, (unsigned char*)img2->imageData, img1->width, img1->height);

    IplImage *clone = cvCloneImage(img1);
    IplImage *grey1 = cvCreateImage(cvGetSize(img1), 8, 1);
    cvCvtColor(img1, grey1, CV_BGR2GRAY);
    cvCvtColor(grey1, clone, CV_GRAY2BGR);

    for(int y=0; y < img1->height; y+=16) {
        for(int x=0; x < img1->width; x+=16) {
            int idx = y*img1->width + x;

            if(LK.status[idx] == 0) 
                continue;

           cvLine(clone, cvPoint(x,y), cvPoint(x+LK.dx[idx], y+LK.dy[idx]), CV_RGB(255,0,0));
        }
    }

    cvSaveImage("cudaLK.png", clone);

    openCVLK(img1, img2);

    return 0;
}

void openCVLK(IplImage *img1, IplImage *img2)
{
    timeval start, end;

    IplImage *grey1 = cvCreateImage(cvGetSize(img1), 8, 1);
    IplImage *grey2 = cvCreateImage(cvGetSize(img2), 8, 1);

    vector<Point2f> prevPts, curPts; // OpenCV 2.x
    vector<CvPoint2D32f> pts1, pts2;
    vector<unsigned char> status;
    vector<float> err;

    for(int y=0; y < img1->height; y++) {
        for(int x=0; x < img1->width; x++) {
            prevPts.push_back(Point2f(x,y)); // OpenCV 2.x
            pts1.push_back(cvPoint2D32f(x,y));
        }
    }

    pts2.resize(pts1.size());
    status.resize(pts1.size());

    gettimeofday(&start, NULL);

    cvCvtColor(img1, grey1, CV_BGR2GRAY);
    cvCvtColor(img2, grey2, CV_BGR2GRAY);

    cvCalcOpticalFlowPyrLK(grey1, grey2, 0, 0, &pts1[0], &pts2[0], pts1.size(),
                           cvSize(13,13),3, (char*)&status[0], 0, 
                            cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,10,0.01),0);
           
    // OpenCV 2.x version, results are not as good as 1.x version                 
    //calcOpticalFlowPyrLK(grey1, grey2, prevPts, curPts, status, err, Size(17, 17), 3, TermCriteria(  TermCriteria::COUNT+TermCriteria::EPS, 10, 0.01), 0);

    gettimeofday(&end, NULL);

    long int total =  (end.tv_sec*1000 + end.tv_usec/1000) - (start.tv_sec*1000 + start.tv_usec/1000);

    printf("Opencv: %ld ms\n", total);

    IplImage *clone = cvCloneImage(img1);
    cvCvtColor(grey1, clone, CV_GRAY2BGR);

    for(int y=0; y < img1->height; y+=16) {
        for(int x=0; x < img1->width; x+=16) {
            int idx = y*img1->width + x;

            if(status[idx] == 0) 
                continue;       

           cvLine(clone, cvPoint(x,y), cvPoint(pts2[idx].x,pts2[idx].y), CV_RGB(255,0,0)); // OpenCV 1.x
           //cvLine(img1, cvPoint(x,y), cvPoint(curPts[idx].x,curPts[idx].y), CV_RGB(255,0,0)); // OpenCV 2.x
        }
    }

    cvSaveImage("opencv.png", clone);
}
