#include "mpi.h"
#include "stdio.h"
//#include <iostream>

int main(int argc,char* argv[]){
	int rank,size,tag=1;
	int senddata,recvdata;
	MPI_Status status;
	/* Initialize the MPI library */
	MPI_Init(&argc,&argv);
	/* Determine the calling process rank and total number of ranks */
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank==0){
		senddata=9999;
		MPI_Send( &senddata, 1, MPI_INT, 1, tag, MPI_COMM_WORLD ); /*发送数据到进程1*/
		printf("rank:0 senddata:%d\n",senddata);
	}

	if(rank==1){
		MPI_Recv(&recvdata, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);/*从进程0接收数据*/
		printf("rank:1 recvdata:%d\n",recvdata);
	}

	/* Shutdown MPI library */
	MPI_Finalize();
	return 0;
}
