#include <omp.h>
#define CHUNKSIZE 100
#define N 1000

int main(void)
{
	//Initialize the arrays and other variables.
  int a[N], b[N], c[N];
  int i, chunk;
  for(int i=0; i<N; i++)
  {
	a[i]=rand() % 100;
	b[i]=rand() % 100;
  }
  chunk = CHUNKSIZE;
  
  #pragma omp parallel shared(a,b,c) private(tid)
  {
	#pragma omp for schedule(dynamic, chunk) nowait
	for (i=0; i < N; i++)
		c[i] = a[i] + b[i];
  }
}