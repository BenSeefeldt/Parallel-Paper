//We will be implementing a the Pagerank algorithm. 
//This an iterative function run on
//a graph. We will be doing 20 iterations. 
//Keep in mind this code contains a solution
//for both serial and parallel implementations. 
//This allows us to double check our answer
//is correct. 
//GOTO 141
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctime>
#include <limits>
#include "mp1-util.h"

event_pair timer;

// amount of floating point numbers between answer and computed value 
// for the answer to be taken correctly. 2's complement magick.
const int maxUlps = 1000;
  
//This method is called from host_graph_iterate and does the
//same as d_host_propagate but in serial.
void host_graph_propagate(unsigned int *graph_indices, unsigned int *graph_edges, float *graph_nodes_in, float *graph_nodes_out, float * inv_edges_per_node, int array_length)
{
  //Loops through every element.
  for(int i=0; i < array_length; i++)
  {
    //The computation.
    float sum = 0.f; 
    for(int j = graph_indices[i]; j < graph_indices[i+1]; j++)
    {
      sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
    }
    graph_nodes_out[i] = 0.5f/(float)array_length + 0.5f*sum;
  }
  //GOTO 212
}

//Our method that is called from main to solve the 
//problem in serial.
void host_graph_iterate(unsigned int *graph_indices, unsigned int *graph_edges, float *graph_nodes_A, float *graph_nodes_B, float * inv_edges_per_node, int nr_iterations, int array_length)
{
  assert((nr_iterations % 2) == 0);
  for(int iter = 0; iter < nr_iterations; iter+=2)
  {
    host_graph_propagate(graph_indices, graph_edges, graph_nodes_A, graph_nodes_B, inv_edges_per_node, array_length);
    host_graph_propagate(graph_indices, graph_edges, graph_nodes_B, graph_nodes_A, inv_edges_per_node, array_length);
  }
  //GOTO 19
}

//The kernel that we will use to solve the problem on the GPU.
//This is called from within the device_graph_iterate method.
__global__ void d_host_propagate(unsigned int *graph_indices, unsigned int *graph_edges, float *graph_nodes_in, float *graph_nodes_out, float *inv_edges_per_node, int array_length)
{
  //We are going to do this by having each thread compute on one
  //piece of data. Note that we are using the formula that was 
  //mentioned earlier for thread ID.
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  //This is necessary to make sure that if for some reason we
  //launched more threads then there are elements the extra
  //threads don't attempt to modify data that isn't present.
  if(idx<array_length)
  {
    //The computation.
    float sum = 0.f;
    for(int j = graph_indices[idx]; j < graph_indices[idx+1]; j++)
    {
      sum += graph_nodes_in[graph_edges[j]]*inv_edges_per_node[graph_edges[j]];
    }
    graph_nodes_out[idx] = 0.5f/(float)array_length + 0.5f*sum;
  }
  //GOTO 207
}

//This method is called from main and passed the data in order to run the code
//on the GPU.
void device_graph_iterate(unsigned int *h_graph_indices,
                          unsigned int *h_graph_edges,
                          float *h_graph_nodes_input,
                          float *h_graph_nodes_result,
                          float *h_inv_edges_per_node,
                          int nr_iterations,
                          int num_elements,
                          int avg_edges)
{
  unsigned int *d_graph_indices, *d_graph_edges;
  float *d_graph_nodes_input, *d_graph_nodes_result, *d_inv_edges_per_node;

  //The cudaMalloc command is the same as the malloc command in C++, except
  //it creates space in the global memory of the GPU.
  //The general format is cudaMalloc(pointer, sizeToAllocate).
  cudaMalloc((void **) &d_graph_indices, (num_elements+1)*sizeof(unsigned int));
  cudaMalloc((void **) &d_graph_edges, num_elements*avg_edges*sizeof(unsigned int));
  cudaMalloc((void **) &d_graph_nodes_input, num_elements*sizeof(float));
  cudaMalloc((void **) &d_graph_nodes_result, num_elements*sizeof(float));
  cudaMalloc((void **) &d_inv_edges_per_node, num_elements*sizeof(float));

  //Now that we have pointers on the device, we have to get the data from the computer
  //onto the GPU. We use cudaMemcpy to do this. General format is:
  //cudaMemcpy(destination, origin, amountofdata, directionOfCopy)
  cudaMemcpy(d_graph_indices, h_graph_indices, (num_elements+1)*sizeof(unsigned int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_edges, h_graph_edges, num_elements*avg_edges*sizeof(unsigned int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_nodes_input, h_graph_nodes_input, num_elements*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph_nodes_result, h_graph_nodes_result, num_elements*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node, num_elements*sizeof(float),cudaMemcpyHostToDevice);

  //These will be our launch parameters. Although 256 is not our maximum, it is a
  //good place to start and can be adjusted to improve performance.
  int blockSize=256;
  //Note the end code ensures that we will have extra block space is all of
  //threads don't fit into block evenly.
  int nBlocks = num_elements/blockSize + (num_elements%blockSize == 0?0:1);

  start_timer(&timer);
  //The for-loop that will be doing our kernel launching for us.
  for(int iter = 0; iter < nr_iterations; iter+=2)
  {
    //A kernel launch looks exactly like a standard method call with the
    //edition of <<< ... >>>. This section holds the launch parameters.
    //We are using the parameters we defined above.
    //GOTO 52
    d_host_propagate <<< nBlocks, blockSize >>> (d_graph_indices, d_graph_edges, d_graph_nodes_input, d_graph_nodes_result, d_inv_edges_per_node, num_elements);
    d_host_propagate <<< nBlocks, blockSize >>> (d_graph_indices, d_graph_edges, d_graph_nodes_result, d_graph_nodes_input, d_inv_edges_per_node, num_elements);
  }
  
  check_launch("gpu graph propagate");
  stop_timer(&timer,"gpu graph propagate");

  //This code copies only the solution back to global memory.
  //The answer should end up in h_graph_nodes_result.
  cudaMemcpy(h_graph_nodes_result, d_graph_nodes_result, (num_elements)*sizeof(float),cudaMemcpyDeviceToHost);

  //Everything is freed before continuing.
  cudaFree(d_graph_indices);
  cudaFree(d_graph_edges);
  cudaFree(d_graph_nodes_result);
  cudaFree(d_graph_nodes_input);
  cudaFree(d_inv_edges_per_node);
}


int main(void)
{
  //This defines the size of our graph.
  int num_elements = 1 << 21;
  int avg_edges = 8;
  int iterations = 20;
  
  //We're going to store everything in arrays, so we first create pointers.
  unsigned int *h_graph_indices = 0;
  float *h_inv_edges_per_node = 0;
  unsigned int *h_graph_edges = 0;
  float *h_graph_nodes_input = 0;
  float *h_graph_nodes_result = 0;
  float *h_graph_nodes_checker_A = 0;
  float *h_graph_nodes_checker_B = 0;
  
  
  // malloc host array
  // index array has to be n+1 so that the last thread can 
  // still look at its neighbor for a stopping point
  h_graph_indices = (unsigned int*)malloc((num_elements+1) * sizeof(unsigned int));
  h_inv_edges_per_node = (float*)malloc((num_elements) * sizeof(float));
  h_graph_edges = (unsigned int*)malloc(num_elements * avg_edges * sizeof(unsigned int));
  h_graph_nodes_input = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_result = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_A = (float*)malloc(num_elements * sizeof(float));
  h_graph_nodes_checker_B = (float*)malloc(num_elements * sizeof(float));
  
  // if any memory allocation failed, report an error message
  if(h_graph_indices == 0 || h_graph_edges == 0 || h_graph_nodes_input == 0 || h_graph_nodes_result == 0 || 
	 h_inv_edges_per_node == 0 || h_graph_nodes_checker_A == 0 || h_graph_nodes_checker_B == 0)
  {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  // generate random input
  // initialize
  srand(time(NULL));
   
  //This whole loop just created our data.
  h_graph_indices[0] = 0;
  for(int i=0;i< num_elements;i++)
  {
    int nr_edges = (i % 15) + 1;
    h_inv_edges_per_node[i] = 1.f/(float)nr_edges;
    h_graph_indices[i+1] = h_graph_indices[i] + nr_edges;
    if(h_graph_indices[i+1] >= (num_elements * avg_edges))
    {
      printf("more edges than we have space for\n");
      exit(1);
    }
    for(int j=h_graph_indices[i];j<h_graph_indices[i+1];j++)
    {
      h_graph_edges[j] = rand() % num_elements;
    }
    
    h_graph_nodes_input[i] =  1.f/(float)num_elements;
    h_graph_nodes_checker_A[i] =  h_graph_nodes_input[i];
    h_graph_nodes_result[i] = std::numeric_limits<float>::infinity();
  }
  
  //This single line does all of the execution for the code on the GPU
  //by calling a method that we've defined above.
  //GOTO 74
  device_graph_iterate(h_graph_indices, h_graph_edges, h_graph_nodes_input, h_graph_nodes_result, h_inv_edges_per_node, iterations, num_elements, avg_edges);
  
  start_timer(&timer);
  // generate reference output on a serial program.
  //GOTO 37
  host_graph_iterate(h_graph_indices, h_graph_edges, h_graph_nodes_checker_A, h_graph_nodes_checker_B, h_inv_edges_per_node, iterations, num_elements);
  
  check_launch("host graph propagate");
  stop_timer(&timer,"host graph propagate");
  
  // check CUDA output versus reference output
  int error = 0;
  int num_errors = 0;
  for(int i=0;i<num_elements;i++)
  {
    float n = h_graph_nodes_result[i];
    float c = h_graph_nodes_checker_A[i];
    //An extra method we've defined elsewhere to help us find if numbers
    //are equal within a certain margin of error.
    if(!AlmostEqual2sComplement(n,c,maxUlps)) 
    {
      num_errors++;
      if (num_errors < 10)
      {
            printf("%d:%.3f::",i, n-c);
      }
      error = 1;
    }
  }
  
  if(error)
  {
    printf("Output of CUDA version and normal version didn't match! \n");
  }
  else
  {
    printf("Worked! CUDA and reference output match. \n");
  }

  // deallocate memory
  free(h_graph_indices);
  free(h_inv_edges_per_node);
  free(h_graph_edges);
  free(h_graph_nodes_input);
  free(h_graph_nodes_result);
  free(h_graph_nodes_checker_A);
  free(h_graph_nodes_checker_B);
}

