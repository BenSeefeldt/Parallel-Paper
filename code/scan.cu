#include "scan.h"
#include "mp3-util.h"
#include <algorithm>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

void inplace_exclusive_scan(unsigned int *d_data,
                            const size_t n)
{
  //We first create a Thrust pointer to the data that is on our device.
  thrust::device_ptr<unsigned int> dev_ptr(d_data);
  //Run the scan on the data.
  thrust::exclusive_scan(dev_ptr, dev_ptr+n, dev_ptr);
  //Return the pointer.
  d_data = thrust::raw_pointer_cast(&dev_ptr[0]);
}

//A serial implementation of the scan.
void exclusive_scan_host(unsigned int *data,
                         unsigned int *out_data,
                         const size_t n)
{
  out_data[0] = 0;
  for(size_t i=1; i<n; i++)
  {
    out_data[i]=out_data[i-1]+data[i-1];
  }
}

