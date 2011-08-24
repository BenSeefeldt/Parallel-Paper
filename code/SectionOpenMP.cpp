#pragma omp sections shared(a,b,c,d) private(i)
{
  #pragma omp section nowait
  {
    #pragma omp section
    for (i=0; i < N; i++)
      c[i] = a[i] + b[i];

    #pragma omp section
    for (i=0; i < N; i++)
      d[i] = a[i] * b[i];
  }
}
