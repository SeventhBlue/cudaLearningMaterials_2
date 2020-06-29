#include "cudaLK.h"
#include <stdio.h>

const float scaling[] = {1, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f};

// Can't use an array of texture<> !! so we'll just re-use the one texture buffer for each image
texture<float, 2, cudaReadModeElementType> texRef_pyramid_prev;
texture<float, 2, cudaReadModeElementType> texRef_pyramid_cur;

__global__ void convertToGrey(unsigned char *d_in, float *d_out, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < N) 
        d_out[idx] = d_in[idx*3]*0.1144f + d_in[idx*3+1]*0.5867f + d_in[idx*3+2]*0.2989f;
}

__global__ void pyrDownsample(float *in, int w1, int h1, float *out, int w2, int h2)
{
    // Input has to be greyscale
    int x2 = blockIdx.x*blockDim.x + threadIdx.x;
    int y2 = blockIdx.y*blockDim.y + threadIdx.y;

    if( (x2 < w2) && (y2 < h2) ) {    
        int x = x2*2;
        int y = y2*2;
        int x_1 = x-1;
        int y_1 = y-1;
        int x_2 = x+1;
        int y_2 = y+1;

        if(x_1 < 0) x_1 = 0;
        if(y_1 < 0) y_1 = 0;
        if(x_2 >= w1) x_2 = w1 - 1;
        if(y_2 >= h1) y_2 = h1 - 1;

        out[y2*w2 + x2] = 0.25f*in[y*w1+x] + 0.125f*(in[y*w1+x_1] + in[y*w1+x_2] + in[y_1*w1+x] + in[y_2*w1+x]) +
                          0.0625f*(in[y_1*w1+x_1] + in[y_2*w1+x_1] + in[y_1*w1+x_2] + in[y_2*w1+x_2]);
    }
}
    
    
__global__ void smoothX(float *in, int w, int h, float *out) 
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= w || y >= h)
        return;

    int idx = y*w;

    int a = x-2;
    int b = x-1;
    int c = x;
    int d = x+1;
    int e = x+2;

    if(a < 0) a = 0;
    if(b < 0) b = 0;
    if(c >= w) c = w-1;
    if(d >= w) d = w-1;

    out[y*w+x] = 0.0625f*in[idx+a] + 0.25f*in[idx+b] + 0.375f*in[idx+c] + 0.25f*in[idx+d] + 0.0625f*in[idx+e];
}
    
__global__ void smoothY(float *in, int w, int h, float *out) 
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= w || y >= h)
        return;

    int a = y-2;
    int b = y-1;
    int c = y;
    int d = y+1;
    int e = y+2;

    if(a < 0) a = 0;
    if(b < 0) b = 0;
    if(c >= h) c = h-1;
    if(d >= h) d = h-1;

    out[y*w+x] = 0.0625f*in[a*w+x] + 0.25f*in[b*w+x] + 0.375f*in[c*w+x] + 0.25f*in[d*w+x] + 0.0625f*in[e*w+x];
}

// Call recursively
// w/h - original dimension of image

__global__ void track(const int w, const int h, 
                      const int pyr_w, const int pyr_h, 
                      float scaling, int level, char initGuess, 
                      float *dx, float *dy, char *status)
{        
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = y*w + x;
    
    if(x > w-1 || y > h-1) 
        return;

    if(status[idx] == 0)
        return;

    float prev_x = x*scaling;
    float prev_y = y*scaling;

    float Vx, Vy;
    float cur_x, cur_y;
    float sum_Ixx = 0;
    float sum_Ixy = 0;
    float sum_Iyy = 0;
    float sum_Ixt;
    float sum_Iyt;
    float Ix, Iy, It;
    int xx, yy;
    float det, D;
    float I, J;
    float vx, vy;
    int j;

    if(initGuess) {
        Vx = 0;
        Vy = 0;
        cur_x = prev_x;
        cur_y = prev_y;
    }
    else {
        Vx = dx[idx];
        Vy = dy[idx];
        cur_x = prev_x + Vx;
        cur_y = prev_y + Vy;
    }
  
    // Calculate spatial gradient 
    for(yy=-PATCH_R; yy <= PATCH_R; yy++) {
        for(xx=-PATCH_R; xx <= PATCH_R; xx++) {  
            Ix = (tex2D(texRef_pyramid_prev, prev_x + xx+1, prev_y + yy) - tex2D(texRef_pyramid_prev, prev_x + xx-1, prev_y + yy))*0.5f;
            Iy = (tex2D(texRef_pyramid_prev, prev_x + xx, prev_y + yy+1) - tex2D(texRef_pyramid_prev, prev_x + xx, prev_y + yy-1))*0.5f;

            sum_Ixx += Ix*Ix;
            sum_Ixy += Ix*Iy;
            sum_Iyy += Iy*Iy;
        }
    }

    det = sum_Ixx*sum_Iyy - sum_Ixy*sum_Ixy;

    if(det < 0.00001f) {
        status[idx] = 0;
        return;
    }

    D = 1/det;

    // Iteration part
    for(j=0; j < 10; j++) {
        if(cur_x < 0 || cur_x > pyr_w || cur_y < 0 || cur_y > pyr_h) {
            status[idx] = 0;
            return;
        }

        sum_Ixt = 0;
        sum_Iyt = 0;

        // No explicit handling of pixels outside the image ... maybe we don't have to because the hardware interpolation scheme
        // will always give a result for pixels outside the image. How greatly the duplicated pixel values affect the result is unknown at the moment.
        for(yy=-PATCH_R; yy <= PATCH_R; yy++) {
            for(xx=-PATCH_R; xx <= PATCH_R; xx++) {            
                I = tex2D(texRef_pyramid_prev, prev_x + xx, prev_y + yy);   
                J = tex2D(texRef_pyramid_cur, cur_x + xx, cur_y + yy);

                Ix = (tex2D(texRef_pyramid_prev, prev_x + xx+1, prev_y + yy) - tex2D(texRef_pyramid_prev, prev_x + xx-1, prev_y + yy))*0.5f;
                Iy = (tex2D(texRef_pyramid_prev, prev_x + xx, prev_y + yy+1) - tex2D(texRef_pyramid_prev, prev_x + xx, prev_y + yy-1))*0.5f;

                It = J - I;

                sum_Ixt += Ix*It;
                sum_Iyt += Iy*It;
            }            
        }

        // Find the inverse of the 2x2 matrix using a mix of determinant and adjugate matrix
        // http://cnx.org/content/m19446/latest/
        vx = D*(-sum_Iyy*sum_Ixt + sum_Ixy*sum_Iyt);
        vy = D*( sum_Ixy*sum_Ixt - sum_Ixx*sum_Iyt);

        Vx += vx;
        Vy += vy;
        cur_x += vx;
        cur_y += vy;
 
        // Movement very small
        if(fabsf(vx) < 0.01f && fabsf(vy) < 0.01f)
            break;
    }

    if(level != 0) {
        cur_x += cur_x;
        cur_y += cur_y;

        Vx += Vx;
        Vy += Vy;
    }

    dx[idx] = Vx;
    dy[idx] = Vy;
}

cudaLK::cudaLK()
{

}

cudaLK::~cudaLK()
{
    for(int i=0; i < LEVELS; i++) {
        cudaFree(gpu_img_pyramid_prev[i]);
        cudaFree(gpu_img_pyramid_cur[i]);
    }

    cudaFree(gpu_smoothed_prev_x);
    cudaFree(gpu_smoothed_cur_x);
    cudaFree(gpu_smoothed_prev);
    cudaFree(gpu_smoothed_cur);

    cudaFreeArray(gpu_array_pyramid_prev);
    cudaFreeArray(gpu_array_pyramid_prev_Ix);
    cudaFreeArray(gpu_array_pyramid_prev_Iy);
    cudaFreeArray(gpu_array_pyramid_cur);

    cudaFree(gpu_dx);
    cudaFree(gpu_dy);
    cudaFree(gpu_status);

    delete [] dx;
    delete [] dy;
    delete [] status;
}

void cudaLK::checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

void cudaLK::initMem()
{
    cudaMalloc((void**)&gpu_img_prev_RGB, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpu_img_cur_RGB, sizeof(char)*w*h*3);
    cudaMalloc((void**)&gpu_img_pyramid_prev[0], sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_img_pyramid_cur[0], sizeof(float)*w*h);

    cudaMalloc((void**)&gpu_smoothed_prev_x, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_smoothed_cur_x, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_smoothed_prev, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_smoothed_cur, sizeof(float)*w*h);

    // Texture
    cudaMallocArray(&gpu_array_pyramid_prev, &texRef_pyramid_prev.channelDesc, w, h);
    cudaMallocArray(&gpu_array_pyramid_cur, &texRef_pyramid_cur.channelDesc, w, h);
    cudaBindTextureToArray(texRef_pyramid_prev, gpu_array_pyramid_prev);
    cudaBindTextureToArray(texRef_pyramid_cur, gpu_array_pyramid_cur);

    texRef_pyramid_prev.normalized = 0;
    texRef_pyramid_prev.filterMode = cudaFilterModeLinear;
    texRef_pyramid_prev.addressMode[0] = cudaAddressModeClamp;
    texRef_pyramid_prev.addressMode[1] = cudaAddressModeClamp;

    texRef_pyramid_cur.normalized = 0;
    texRef_pyramid_cur.filterMode = cudaFilterModeLinear;
    texRef_pyramid_cur.addressMode[0] = cudaAddressModeClamp;
    texRef_pyramid_cur.addressMode[1] = cudaAddressModeClamp;

    cudaMalloc((void**)&gpu_dx, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_dy, sizeof(float)*w*h);
    cudaMalloc((void**)&gpu_status, sizeof(char)*w*h);

    int _w = w;
    int _h = h;

    dx = new float[w*h];
    dy = new float[w*h];
    status = new char[w*h];

    pyr_w[0] = w;
    pyr_h[0] = h;

    for(int i=1; i < LEVELS; i++) {
        _w /= 2;
        _h /= 2;
        pyr_w[i] = _w;
        pyr_h[i] = _h;

        cudaMalloc((void**)&gpu_img_pyramid_prev[i], sizeof(float)*_w*_h);
        cudaMalloc((void**)&gpu_img_pyramid_cur[i], sizeof(float)*_w*_h);
    }
}

void cudaLK::run(unsigned char *prev, unsigned char *cur, int _w, int _h)
{;
    w = _w;
    h = _h;
    initMem();

    int nThreadsX = NTHREAD_X;
    int nThreadsY = NTHREAD_Y;

    int blocksW = w/nThreadsX + ((w % nThreadsX)?1:0);
    int blocksH = h/nThreadsY + ((h % nThreadsY )?1:0);
    dim3 blocks(blocksW, blocksH);
    dim3 threads(nThreadsX, nThreadsY);
    int blocks1D = (w*h)/256 + (w*h % 256?1:0); // for greyscale
  
    int start = getTimeNow();
    int s;

    // Copy image to GPU 
    s = getTimeNow();
    cudaMemcpy(gpu_img_prev_RGB, prev, w*h*3, cudaMemcpyHostToDevice);  
    cudaMemcpy(gpu_img_cur_RGB, cur, w*h*3, cudaMemcpyHostToDevice);  
    checkCUDAError("start");

    printf("Copying 2 images from CPU to GPU: %d ms\n", getTimeNow() - s);

    // RGB -> grey
    s = getTimeNow();
    convertToGrey<<<blocks1D, 256>>>(gpu_img_prev_RGB, gpu_img_pyramid_prev[0], w*h);
    convertToGrey<<<blocks1D, 256>>>(gpu_img_cur_RGB, gpu_img_pyramid_cur[0], w*h);
    cudaThreadSynchronize();
    checkCUDAError("convertToGrey");
    printf("Converting from RGB to greyscale: %d ms\n", getTimeNow() - s);

  
    s = getTimeNow();

    for(int i=0; i < LEVELS-1; i++) {
        smoothX<<<blocks, threads>>>(gpu_img_pyramid_prev[i], pyr_w[i], pyr_h[i], gpu_smoothed_prev_x);
        smoothX<<<blocks, threads>>>(gpu_img_pyramid_cur[i], pyr_w[i], pyr_h[i], gpu_smoothed_cur_x);
        cudaThreadSynchronize();
        smoothY<<<blocks, threads>>>(gpu_smoothed_prev_x, pyr_w[i], pyr_h[i], gpu_smoothed_prev);
        smoothY<<<blocks, threads>>>(gpu_smoothed_cur_x, pyr_w[i], pyr_h[i], gpu_smoothed_cur);
        cudaThreadSynchronize();

        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_prev, pyr_w[i], pyr_h[i], gpu_img_pyramid_prev[i+1], pyr_w[i+1], pyr_h[i+1]);
        pyrDownsample<<<blocks, threads>>>(gpu_smoothed_cur,  pyr_w[i], pyr_h[i], gpu_img_pyramid_cur[i+1],  pyr_w[i+1], pyr_h[i+1]);
        cudaThreadSynchronize();

        checkCUDAError("pyrDownsample here");  
    }

    printf("Generating the pyramids: %d ms\n", getTimeNow() - s);

    s = getTimeNow();
    cudaMemset(gpu_status, 1, sizeof(char)*w*h);

    // Do the actual tracking
    for(int l=LEVELS-1; l >= 0; l--) {

        cudaMemcpy2DToArray(gpu_array_pyramid_prev, 0, 0, gpu_img_pyramid_prev[l], 
                            sizeof(float)*pyr_w[l], sizeof(float)*pyr_w[l], pyr_h[l], cudaMemcpyDeviceToDevice);

        cudaMemcpy2DToArray(gpu_array_pyramid_cur, 0, 0, gpu_img_pyramid_cur[l], 
                            sizeof(float)*pyr_w[l], sizeof(float)*pyr_w[l], pyr_h[l], cudaMemcpyDeviceToDevice);

        track<<<blocks, threads>>>(w, h, pyr_w[l], pyr_w[l], scaling[l], l, (l == LEVELS-1), gpu_dx, gpu_dy, gpu_status);

        cudaThreadSynchronize();
    }

    printf("Tracking: %d ms\n", getTimeNow() - s);

    // Copy back results 
    s = getTimeNow();
    cudaMemcpy(dx, gpu_dx, sizeof(float)*w*h, cudaMemcpyDeviceToHost);  
    cudaMemcpy(dy, gpu_dy, sizeof(float)*w*h, cudaMemcpyDeviceToHost);  
    cudaMemcpy(status, gpu_status, sizeof(char)*w*h, cudaMemcpyDeviceToHost);  
    printf("Copying results from GPU to CPU: %d ms\n", getTimeNow() - s);

    printf("Total time for cudaLK: %d ms\n", getTimeNow() - start);
}

