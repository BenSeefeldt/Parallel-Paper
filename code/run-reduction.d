float SerialSumFoo( float a[], size_t n ) {
  float sum = 0;
  for( size_t i=0; i!=n; ++i)
    sum += Foo(a[i]);
  return sum;
}

float ParallelSumFoo( const float a[], size_t n) {
  SumFoo sf(a);
  parallel_reduce( blocked_range<size_t>(0,n), sf);;
}
