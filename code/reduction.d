class SumFoo {
  float* my_a;
public:
  float my_sum;
  void operator()( const blocked_range<size_t>& r ) {
    float * a = my_a;
    float sum = my_sum;
    size_t end = r.end();
    for( size_t i=r.begin(); i!=end; ++i )
      sum += Foo(a[i]);
    my_sum = sum;
  }

  SumFoo( SumFoo& x, split) : my_a(x.my_a), my_sum(0) {}

  void join( const SumFoo& y ) {my__sum+=y.sum;}

  SumFoo(float a[] ) :
    my_a(a), my_sum(0)
  {}
};
