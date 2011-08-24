#include "compact.h"
#include "scatter.h"
#include "scan.h"
#include "mp3-util.h"
#include <algorithm>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>



//A kernel that will creates the bit vector of evens.
__global__ void even_bit_vector_kernel(const unsigned int *d_input,
                                       const int n,
                                       unsigned int *d_output)
{
  unsigned int i = blockIdx.x * blockDim.x +threadIdx.x;

  if(i<n)
  {
    if(d_input[i] & 1 ? false : true)
      d_output[i]=1;
    else
      d_output[i]=0;
  }
}

// compact_even_elements copies the even-valued elements of d_input
// to a compacted output in d_result, and returns the number of 
// elements in the compacted output
// for example, given
// d_input           = [ 0 1 2 4 10 2 3 5 0 3 ] and
// n = 10,
// this function yields
// d_result          = [ 0 2 4 10 2 0 <undefined...> ]
// and returns the value 6
size_t compact_even_elements(const unsigned int *d_input,
                             const size_t n,
                             unsigned int *d_result)
{
  //Launch conditions.
  int blockSize = 512;
  int nBlocks= n/blockSize + (n%blockSize == 0?0:1);
  unsigned int comp_ele[1];
  comp_ele[0]=5;

  //A temporary array for the bit vector.
  unsigned int *d_temp = 0;
  cudaMalloc((void **) &d_temp, n*sizeof(unsigned int));
  if(d_temp == 0)
    printf("CUDA allocation Error: BEN THREW.");
  even_bit_vector_kernel<<< nBlocks, blockSize >>>(d_input, n, d_temp);
  inplace_exclusive_scan(d_temp, n);
  scatter_even_elements(d_input, d_temp, n, d_result);  
  //Deallocate bit vector array.
  cudaMemcpy(&comp_ele[0],&d_temp[n-1], 1*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_temp);

  printf("COMPELE is: %u\n",comp_ele);
  size_t num_compacted_elements = comp_ele[0];

  return num_compacted_elements;
}

//A kernel that will creates the bit vector of evens.
__global__ void min_bit_vector_kernel(const real *d_call,
                                      const real *d_put,
                                      const int n,
                                      real min_call_threshold,
                                      real min_put_threshold,
                                      unsigned int *d_output)
{
  unsigned int i = blockIdx.x * blockDim.x +threadIdx.x;

  if(i<n)
  {
    if(d_call[i] >= min_call_threshold && d_put[i]>=min_put_threshold)
      d_output[i]=1;
    else
      d_output[i]=0;
  }
}

// compact_options copies the input options whose call and put
// results from the first round meet or exceed the given call & put
// thresholds to a compacted output in three result arrays.
size_t compact_options(const real min_call_threshold,
                       const real min_put_threshold,
                       const real *d_call,
                       const real *d_put,
                       const real *d_stock_price_input,
                       const real *d_option_strike_input,
                       const real *d_option_years_input,
                       const size_t n,
                       real *d_stock_price_result,
                       real *d_option_strike_result,
                       real *d_option_years_result)
{
  //Launch conditions.
  int blockSize = 512;
  int nBlocks= n/blockSize + (n%blockSize == 0?0:1);
  unsigned int comp_ele[1];
  comp_ele[0]=5;

  //A temporary array for the bit vector.
  unsigned int *d_temp = 0;
  cudaMalloc((void **) &d_temp, n*sizeof(real));
  if(d_temp == 0)
    printf("CUDA allocation Error: BEN THREW. Compacting options");
  min_bit_vector_kernel<<< nBlocks, blockSize >>>(d_call, d_put, n, min_call_threshold, min_put_threshold, d_temp);
  inplace_exclusive_scan(d_temp, n);
  scatter_options(min_call_threshold, min_put_threshold, d_call, d_put, d_stock_price_input, d_option_strike_input,
                  d_option_years_input, d_temp, n, d_stock_price_result, d_option_strike_result, d_option_years_result);

  //Deallocate bit vector array.
  cudaMemcpy(&comp_ele[0], &d_temp[n-1], 1*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(d_temp);

  size_t num_compacted_elements = comp_ele[0];

  return num_compacted_elements;
}

//A serial implementation to
void compact_host(unsigned int *in,
                  int num,
                  unsigned int *out)
{
  int num_placed=0;
  for(int i = 0; i < num ; i++)
  {
    if(in[i] & 1 ? false : true)
    {
      out[num_placed]=in[i];
      num_placed++;
    }
  }
}
