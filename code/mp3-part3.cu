#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <limits>
#include <numeric>

#include "black_scholes.h"
#include "compact.h"
#include "mp3-util.h"

//Method for allocating device memory.
bool allocate_device_storage(real *&d_first_round_call_result, real *&d_first_round_put_result,
                             real *&d_subsequent_round_call_result, real *&d_subsequent_round_put_result,
                             real *&d_stock_price, real *&d_option_strike,
                             real *&d_option_years,
                             real *&d_compacted_stock_price,
                             real *&d_compacted_option_strike,
                             real *&d_compacted_option_years,
                             const size_t n)
{
  cudaMalloc((void **) &d_first_round_call_result, n*sizeof(float3));
  cudaMalloc((void **) &d_first_round_put_result, n*sizeof(float3));
  cudaMalloc((void **) &d_subsequent_round_call_result, n*sizeof(float3));
  cudaMalloc((void **) &d_subsequent_round_put_result, n*sizeof(float3));
  cudaMalloc((void **) &d_stock_price, n*sizeof(float3));
  cudaMalloc((void **) &d_option_strike, n*sizeof(float3));
  cudaMalloc((void **) &d_option_years, n*sizeof(float3));

  cudaMalloc((void **) &d_compacted_stock_price, n*sizeof(float3));
  cudaMalloc((void **) &d_compacted_option_strike, n*sizeof(float3));
  cudaMalloc((void **) &d_compacted_option_years, n*sizeof(float3));



  if(d_first_round_call_result == 0 || d_first_round_put_result == 0 || d_subsequent_round_call_result == 0 ||
     d_subsequent_round_put_result == 0 || d_stock_price == 0 || d_option_strike == 0 || d_option_years == 0 || 
     d_compacted_stock_price == 0 || d_compacted_option_strike == 0 || d_compacted_option_years ==0)
    return false;
  else
    return true;
}

//Frees  device memory.
void deallocate_device_storage(real *d_first_round_call_result, real *d_first_round_put_result,
                               real *d_subsequent_round_call_result, real *d_subsequent_round_put_result,
                               real *d_stock_price, real *d_option_strike,
                               real *d_option_years,
                               real *d_compacted_stock_price,
                               real *d_compacted_option_strike,
                               real *d_compacted_option_years)
{
 
  cudaFree(d_first_round_call_result);
  cudaFree(d_first_round_put_result);
  cudaFree(d_subsequent_round_call_result);
  cudaFree(d_subsequent_round_put_result);
  cudaFree(d_stock_price);
  cudaFree(d_option_strike);
  cudaFree(d_option_years);
  cudaFree(d_compacted_stock_price);
  cudaFree(d_compacted_option_strike);
  cudaFree(d_option_years);
}


int main(void)
{
  event_pair timer;

  const size_t num_subsequent_rounds = 5;
  float compaction_time = 0;
  std::vector<float> gpu_time(1 + num_subsequent_rounds);
  std::vector<float> cpu_time(1 + num_subsequent_rounds);

  // create arrays for 4M options
  size_t num_options = 1<<22;

  // allocate host storage
  std::vector<real> h_first_round_call_result(num_options,0);
  std::vector<real> h_first_round_put_result(num_options, 0);
  std::vector<real> h_subsequent_round_call_result(num_options,0);
  std::vector<real> h_subsequent_round_put_result(num_options, 0);
  std::vector<real> h_stock_price(num_options);
  std::vector<real> h_option_strike(num_options);
  std::vector<real> h_option_years(num_options);

  // generate options set
  srand(5347);
  for(int i = 0; i < num_options; ++i)
  {
    h_stock_price[i]   = random_real(5.0,  30.0);
    h_option_strike[i] = random_real(1.0, 100.0);
    h_option_years[i]  = random_real(0.25, 10.0);
  }

  // some pointers to the data set which will live in device memory
  real *d_first_round_call_result      = 0;
  real *d_first_round_put_result       = 0;
  real *d_subsequent_round_call_result = 0;
  real *d_subsequent_round_put_result  = 0;
  real *d_stock_price                  = 0;
  real *d_option_strike                = 0;
  real *d_option_years                 = 0;
  real *d_compacted_stock_price        = 0;
  real *d_compacted_option_strike      = 0;
  real *d_compacted_option_years       = 0;

  // allocate device storage
  if(!allocate_device_storage(d_first_round_call_result, d_first_round_put_result,
                              d_subsequent_round_call_result, d_subsequent_round_put_result,
                              d_stock_price, d_option_strike, d_option_years,
                              d_compacted_stock_price,
                              d_compacted_option_strike,
                              d_compacted_option_years,
                              num_options))
  {
    std::cerr << "Error allocating device memory!" << std::endl;
    exit(-1);
  }

  // fill the result arrays with 0
  cudaMemset(d_first_round_call_result,      0, sizeof(real) * num_options);
  cudaMemset(d_first_round_put_result,       0, sizeof(real) * num_options);
  cudaMemset(d_subsequent_round_call_result, 0, sizeof(real) * num_options);
  cudaMemset(d_subsequent_round_put_result,  0, sizeof(real) * num_options);

  // copy input to GPU
  start_timer(&timer);
  cudaMemcpy(d_stock_price, &h_stock_price[0], sizeof(real) * num_options, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_strike, &h_option_strike[0], sizeof(real) * num_options, cudaMemcpyHostToDevice);
  cudaMemcpy(d_option_years, &h_option_years[0], sizeof(real) * num_options, cudaMemcpyHostToDevice);
  stop_timer(&timer, "host to device copy of input"); 


  // BEGIN ROUND 0

  // we will use the two following parameters
  // to first round of the Black-Scholes algorithm
  const real first_round_riskless_rate = 0.02;
  const real first_round_volatility    = 0.30;

  //Calculates kernel launch parameters.
  int blockSize = 512;
  int nBlocks = num_options/blockSize + (num_options%blockSize == 0?0:1);

  // do one round of Black-Scholes using our parameters
  start_timer(&timer);
  black_scholes_kernel<<< nBlocks, blockSize >>>(d_stock_price, d_option_strike, d_option_years, d_first_round_call_result,
                                                 d_first_round_put_result, first_round_riskless_rate,
                                                 first_round_volatility, num_options);
 
  gpu_time[0] = stop_timer(&timer, "GPU Black-Scholes round 0");
  check_cuda_error("GPU Black-Scholes round 0", __FILE__, __LINE__);

  // do round 0 of Black-Scholes on the host
  start_timer(&timer);
  black_scholes_host(&h_stock_price[0],
                     &h_option_strike[0],
                     &h_option_years[0],
                     &h_first_round_call_result[0],
                     &h_first_round_put_result[0],
                     first_round_riskless_rate,
                     first_round_volatility,
                     num_options);
  cpu_time[0] = stop_timer(&timer, "CPU Black-Scholes round 0");

  // validate gpu results from round 0
  std::vector<real> h_validate_me(num_options);
  cudaMemcpy(&h_validate_me[0], d_first_round_call_result, sizeof(real) * num_options, cudaMemcpyDeviceToHost);
  // pass true as a final optional argument to fuzzy_validate for verbose output
  if(!fuzzy_validate(&h_validate_me[0], &h_first_round_call_result[0], num_options))
  {
    std::cerr << "Error: round 0 of call results don't match!" << std::endl;
    exit(-1);
  }

  cudaMemcpy(&h_validate_me[0],  d_first_round_put_result,  sizeof(real) * num_options, cudaMemcpyDeviceToHost);
  if(!fuzzy_validate(&h_validate_me[0], &h_first_round_put_result[0], num_options))
  {
    std::cerr << "Error: round 0 of put results don't match!" << std::endl;
    exit(-1);
  }


  // BEGIN COMPACTION


  // in subsequent rounds, select the stocks whose call & put prices from the first round
  // meet or exceed these thresholds
  const real min_call_threshold = 2.0;
  const real min_put_threshold  = 4.0;

  // compact the options, copying those that meet our call & put thresholds
  // to the arrays for round 2
  start_timer(&timer);
  size_t num_compacted_options = 0;
  //Compacts the options
  num_compacted_options = compact_options(min_call_threshold, min_put_threshold, d_first_round_call_result, d_first_round_put_result, d_stock_price, d_option_strike, d_option_years, num_options, d_compacted_stock_price, d_compacted_option_strike, d_compacted_option_years);
  compaction_time = stop_timer(&timer, "GPU Compaction");


  // BEGIN SUBSEQUENT ROUNDS

  size_t num_compacted_options_reference = 0;

  for(int round = 1; round < num_subsequent_rounds + 1; ++round)
  {
    // change the parameters of the model in each subsequent round
    const real riskless_rate = random_real(0.03, 0.04);
    const real volatility    = random_real(0.50, 0.60);

    //Redefines kernel launch parameters for the new data.
    //I'm using a smaller blockSize because there are presumably less elements this time.
    blockSize = 256;
    nBlocks = num_compacted_options/blockSize + (num_compacted_options%blockSize == 0?0:1);

    // do round of Black-Scholes using new parameters on the device
    start_timer(&timer);
    black_scholes_kernel<<< nBlocks, blockSize >>>(d_compacted_stock_price,
                                                   d_compacted_option_strike,
                                                   d_compacted_option_years,
                                                   d_subsequent_round_call_result,
                                                   d_subsequent_round_put_result,
                                                   riskless_rate,
                                                   volatility,
                                                   num_compacted_options);
 
    char message[256];
    sprintf(message, "GPU Black-Scholes round %d", round);
    gpu_time[round] = stop_timer(&timer, message);
    check_cuda_error(message, __FILE__, __LINE__);


    // do a round of Black-Scholes on the host using new parameters
    // filter the set of options to compute given the results of the last round,
    // but compact the output
    start_timer(&timer);
    num_compacted_options_reference =
      compacted_black_scholes_host(&h_stock_price[0],
                                   &h_option_strike[0],
                                   &h_option_years[0],
                                   &h_first_round_call_result[0],
                                   &h_first_round_put_result[0],
                                   &h_subsequent_round_call_result[0],
                                   &h_subsequent_round_put_result[0],
                                   min_call_threshold,
                                   min_put_threshold,
                                   riskless_rate,
                                   volatility,
                                   num_options);
    sprintf(message, "CPU Black-Scholes round %d", round);
    cpu_time[round] = stop_timer(&timer, message);

    if(num_compacted_options_reference != num_compacted_options)
    {
      std::cerr << "Error: round " << round << " num_compacted_options (" << num_compacted_options << ") doesn't match num_compacted_options_reference (" << num_compacted_options_reference << ")" << std::endl;
      exit(-1);
    }

    // validate gpu results from this round
    cudaMemcpy(&h_validate_me[0], d_subsequent_round_call_result, sizeof(real) * num_compacted_options_reference, cudaMemcpyDeviceToHost);
    if(!fuzzy_validate(&h_validate_me[0], &h_subsequent_round_call_result[0], num_compacted_options_reference))
    {
      std::cerr << "Error: round " << round << " of call results don't match!" << std::endl;
      exit(-1);
    }

    cudaMemcpy(&h_validate_me[0],  d_subsequent_round_put_result,  sizeof(real) * num_compacted_options_reference, cudaMemcpyDeviceToHost);
    if(!fuzzy_validate(&h_validate_me[0], &h_subsequent_round_put_result[0], num_compacted_options_reference))
    {
      std::cerr << "Error: round " << round << " of put results don't match!" << std::endl;
      exit(-1);
    }

  } // end for subsequent round

  deallocate_device_storage(d_first_round_call_result, d_first_round_put_result,
                            d_subsequent_round_call_result, d_subsequent_round_put_result,
                            d_stock_price, d_option_strike,
                            d_option_years,
                            d_compacted_stock_price,
                            d_compacted_option_strike,
                            d_compacted_option_years);

  // output a report
  std::cout << std::endl;

  real first_round_gpu_throughput = static_cast<real>(num_options) / (gpu_time[0] / 1000.0f);
  real first_round_cpu_throughput = static_cast<real>(num_options) / (cpu_time[0] / 1000.0f);

  std::cout << "Round 0: " << num_options << " options" << std::endl;
  std::cout << "Throughput of GPU Black-Scholes Round 0: " << (first_round_gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Throughput of CPU Black-Scholes Round 0: " << (first_round_cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Speedup of Round 0: " << first_round_gpu_throughput / first_round_cpu_throughput << "x" << std::endl << std::endl;

  for(int i = 1; i < gpu_time.size(); ++i)
  {
    real gpu_throughput = static_cast<real>(num_compacted_options_reference) / (gpu_time[i] / 1000.0f);
    real cpu_throughput = static_cast<real>(num_compacted_options_reference) / (cpu_time[i] / 1000.0f);

    std::cout << "Round " << i << ": " << num_compacted_options_reference << " options" << std::endl;
    std::cout << "Throughput of GPU Black-Scholes Round " << i << ": " << (gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
    std::cout << "Throughput of CPU Black-Scholes Round " << i << ": " << (cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
    std::cout << "Speedup of Round " << i << ": " << gpu_throughput / cpu_throughput << "x" << std::endl << std::endl;
  }

  // report overall performance
  real total_gpu_time = compaction_time + std::accumulate(gpu_time.begin(), gpu_time.end(), 0.0);
  real total_cpu_time = std::accumulate(cpu_time.begin(), cpu_time.end(), 0.0);
  real gpu_throughput = static_cast<real>(num_options + num_subsequent_rounds*num_compacted_options_reference) / ((total_gpu_time) / 1000.0f);
  real cpu_throughput = static_cast<real>(num_options + num_subsequent_rounds*num_compacted_options_reference) / ((total_cpu_time) / 1000.0f);

  std::cout << "Overall GPU throughput: " << (gpu_throughput / 1e6) << " Megaoptions/sec" << std::endl;
  std::cout << "Overall CPU throughput: " << (cpu_throughput / 1e6) << " Megaoptions/sec" << std::endl << std::endl;

  std::cout << "Overall speedup: " << gpu_throughput / cpu_throughput << "x" << std::endl;

  return 0;
}

