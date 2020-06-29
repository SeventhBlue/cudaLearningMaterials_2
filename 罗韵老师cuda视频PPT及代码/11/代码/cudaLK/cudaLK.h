#ifndef CUDALK_H
#define CUDALK_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define LEVELS 3
#define PATCH_R 6 // patch radius, patch size is (2*PATCH_R+1)*(2*PATCH_R+1)
#define NTHREAD_X 16
#define NTHREAD_Y 16

class cudaLK
{
public:
    cudaLK();
    ~cudaLK();
    void run(unsigned char *prev, unsigned char *cur, int w, int h);

    float *dx, *dy;
    char *status;
    int pyr_w[LEVELS], pyr_h[LEVELS];

private:
    void initMem();
    void checkCUDAError(const char *msg);

    int w, h;

    unsigned char *gpu_img_prev_RGB;
    unsigned char *gpu_img_cur_RGB;

    float *gpu_img_pyramid_prev[LEVELS];
    float *gpu_img_pyramid_cur[LEVELS];
    float *gpu_smoothed_prev_x;
    float *gpu_smoothed_cur_x;
    float *gpu_smoothed_prev;
    float *gpu_smoothed_cur;

    cudaArray *gpu_array_pyramid_prev;
    cudaArray *gpu_array_pyramid_prev_Ix;
    cudaArray *gpu_array_pyramid_prev_Iy;
    cudaArray *gpu_array_pyramid_cur;
    float *gpu_dx, *gpu_dy;
    char *gpu_status;

    inline int getTimeNow() {
        timeval t;
        gettimeofday(&t, NULL);

        return (t.tv_sec*1000 + t.tv_usec/1000);
    }
};

#endif
