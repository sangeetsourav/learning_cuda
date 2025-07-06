// This program estimates the value of pi through Monte-Carlo simulations. It will generate points in a 1x1 square
// with a corner at (0,0) and then count how many points fall within the quadrant centered at 0,0.
// 4 * (pts inside quadrant)/(total pts) = pi

#include <cuda_runtime.h> // Runtime API, For cudamalloc stuff
#include <device_launch_parameters.h>	// For threadIdx stuff
#include <curand_kernel.h> // device-side library
#include <chrono>
#include <cmath>
#include <iostream>

typedef unsigned long long int big_counter;

const int WARP_SIZE = 32;
const int WARPS_PER_BLOCK = 2;
const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

const int NUM_BLOCKS = 2560; // Number of CUDA cores on my GPU (RTX 1000 Ada)

const big_counter ITERATIONS_PER_THREAD = 100000;

__global__ void runMonteCarlo(big_counter* total_count)
{	
	// Shared memory for all threads in a block
	__shared__ big_counter current_block_counts[THREADS_PER_BLOCK];

	// Get thread id
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	curandState_t rng;
	// clock64() acts as the seed	
	curand_init(clock64(), thread_id, 0, &rng);

	// Start Monte-Carlo for this thread
	double x, y;
	current_block_counts[threadIdx.x] = 0;
	for (size_t i = 0; i < ITERATIONS_PER_THREAD; i++)
	{
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);

		if (std::sqrt(x*x + y*y) <=1 )
		{
			current_block_counts[threadIdx.x] += 1;
		}
	}
	
	// Synchronization is needed because only the threads in a warp are guaranteed to be synchronised
	if (WARPS_PER_BLOCK > 1)
	{
		__syncthreads();
	}

	total_count[blockIdx.x] = 0;

	for (size_t i = 0; i < THREADS_PER_BLOCK; i++)
	{
		total_count[blockIdx.x] += current_block_counts[i];
	}

}

int main()
{
	std::cout << "Monte Carlo Simulations for estimating Pi.\n";
	std::cout << "Running on " << NUM_BLOCKS << " cores, with 1 thread-block per core and " << THREADS_PER_BLOCK << " threads per block\n";
	std::cout << "\tTotal threads = " << NUM_BLOCKS * THREADS_PER_BLOCK << "\n";
	std::cout << "\tIterations per thread = " << ITERATIONS_PER_THREAD << "\n";
	std::cout << "\tTotal random tests = " << ITERATIONS_PER_THREAD * THREADS_PER_BLOCK * NUM_BLOCKS/1000000 << " Million\n";

	auto start = std::chrono::system_clock::now();

	// We need to allocate space for 1 counter per block. Each block's counter will store how many pts were inside the quadrant
	size_t bytes = NUM_BLOCKS * sizeof(big_counter);

	big_counter* block_counters;

	cudaMallocManaged(&block_counters, bytes);

	runMonteCarlo <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (block_counters);
	cudaDeviceSynchronize();

	big_counter total = 0;
	for (int i = 0; i < NUM_BLOCKS; i++) {
		total += block_counters[i];
	}

	std::cout << "\tTotal random tests falling within quadrant = " << total/1000000 << " Million\n";
	std::cout << "Estimation of Pi = " << 4 * (double)total / (ITERATIONS_PER_THREAD * THREADS_PER_BLOCK * NUM_BLOCKS) << "\n";

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> duration_in_seconds = end - start;
	std::cout << "Computation time (s) = " << duration_in_seconds.count();
}
