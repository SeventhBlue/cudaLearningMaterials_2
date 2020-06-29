
#include "GL/glut.h"
//#include "cutil.h"

#define W	640
#define H	400
float h_a[W], *d_a;

__global__ void do_cuda(float *a) {
    int inx=blockIdx.x*blockDim.x+threadIdx.x;
    a[inx]=sinf(inx*0.1);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for(int i=0; i<W; i++) glVertex2f((float)(i*2.0/W-1), 0.2f*h_a[i]);
    glEnd();
    glFinish();
}

int main(int argc, char **argv) {
    cudaMalloc((void**)&d_a, sizeof(h_a));
    do_cuda<<<20,32>>>(d_a);
    cudaMemcpy(h_a, d_a, sizeof(h_a), cudaMemcpyDeviceToHost);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(W, H);
    glutCreateWindow("f=sin(x)");
    glutDisplayFunc(display);
    glutMainLoop();
}

