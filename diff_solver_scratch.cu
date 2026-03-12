#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using Clock = std::chrono::steady_clock;

// usable from both host and device
__host__ __device__
double helper_func(double T) {
    return static_cast<double> (T < 0.0);
}

// CUDA kernel: one time step on device arrays
__global__ void step_once_kernel(const double* T, const double* phi, double* Tnext, double* phinext, 
		double dt, double param, double lambda_, size_t N){
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = thread_idx; i < N; i += stride) {
	// 1) term1 for ALL i
    	double term1 = -(1.0 - phi[i]) * T[i] * helper_func(T[i]);

    	// 2) update phi for ALL i
    	phinext[i] = phi[i] + dt * term1;

    	// 3) update 
    	if (i == 0) {
	    Tnext[i] = T[i]; // left boundary untouched
	}
    	else if (i == N-1) {
	    Tnext[i] = T[i]; // right boundary untouched for now
    	}
    	else {
            const double lap = (T[i+1] + T[i-1] - 2.0 * T[i]);
            const double term2 = param * lap + dt * lambda_ * term1;
            Tnext[i] = T[i] + term2;
    	}
	}
}

__global__ void apply_right_bc(double* T, size_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        T[N - 1] = T[N - 2];
    }
}

// grid size is 1.0

// auto start = Clock::now();

int main() {
    auto start = Clock::now();
    // parameters
    const double lambda_ = 5.0;
    const double param   = 0.01;
    const double dt      = 0.01;
    const size_t N       = 1000000; // no. of grids
    const int    steps   = 100000;   // time steps
    const int    output_interval = 50000; // copy back & write to csv

    
    // initialize/allocate
    vector<double> T(N, 0.0);
    vector<double> phi(N, 0.0);
    T[0] = -1.0;
    //vector<double> term1(N, 0.0);
    //vector<double> Tnext(N, 0.0);
    //vector<double> phinext(N, 0.0);
    
    // initiate null pointers to use later
    double *d_T = nullptr, *d_phi = nullptr;
    double *d_Tnext = nullptr, *d_phinext = nullptr;
    
    size_t d_N = N;
    // int threadsPerBlock = 256;
    //int blocks = 0; // no. of blocks required will be calculated later

    // GPU memory allocation and host to device memory copy
    cudaMalloc(&d_T,       N * sizeof(double));
    cudaMalloc(&d_phi,     N * sizeof(double));
    cudaMalloc(&d_Tnext,   N * sizeof(double));
    cudaMalloc(&d_phinext, N * sizeof(double));

    cudaMemcpy(d_T,   T.data(),   N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    //cudaMemset(d_Tnext, 0, N * sizeof(double));
    //cudaMemset(d_phinext, 0, N * sizeof(double));

    // kernel launch configuration
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;

    int block = 256;  // one d block dimension -- how many threads per block
    int grid = 8 * num_sms; // one d grid dimension -- how many thread blocks?

    // for Nvidia a100, num_sms = 108, and each sm can hold max of 2048 resident threads.
    // this is the hardware limit. If I have 256 threads per block, that means each sm
    // can hold 2048/256 = 8 blocks max. -- minimimum no. of blocks that should be launched is 8*108  
    // 64 warps per sm.-- tune grid: 4, 8, 16, 32 

    // output files
    ofstream tfile("T_gpu_profiling.csv");
    ofstream phifile("phi_gpu_profiling.csv");

    // optional: consistent numeric formatting
    tfile.setf(ios::fixed); tfile.precision(6);
    phifile.setf(ios::fixed); phifile.precision(6);

    // write step 0 (initial state) – already in host vectors
    tfile << 0;
    for (size_t i = 0; i < N; ++i) tfile << ',' << T[i];
    tfile << '\n';

    phifile << 0;
    for (size_t i = 0; i < N; ++i) phifile << ',' << phi[i];
    phifile << '\n';
    // --------------------------------------------

    for (int s = 0; s < steps; ++s) {
        // one step on GPU
	step_once_kernel<<<grid, block>>>(d_T, d_phi,
                                          d_Tnext, d_phinext,
                                          dt, param, lambda_, d_N);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    	    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
            return 1;
        }

	// apply right boundary condition
	apply_right_bc<<<1,1>>>(d_T, d_N);
	// swap
	swap(d_T, d_Tnext);
	swap(d_phi, d_phinext);
        // every 1M steps: copy back and write
        if ((s + 1) % output_interval == 0) {
            const int out_step = s + 1;
            
	    // copy device to host
            cudaMemcpy(T.data(),   d_T,   d_N * sizeof(double), cudaMemcpyDeviceToHost);
    	    cudaMemcpy(phi.data(), d_phi, d_N * sizeof(double), cudaMemcpyDeviceToHost);

            tfile << out_step;
            for (size_t i = 0; i < N; ++i) tfile << ',' << T[i];
            tfile << '\n';

            phifile << out_step;
            for (size_t i = 0; i < N; ++i) phifile << ',' << phi[i];
            phifile << '\n';
        }

        if (s % 10000 == 0) { // progress print every 10k steps
            cout << "step: " << s << '\n';
        }
    }

    cudaDeviceSynchronize();
    
    cudaFree(d_T);
    cudaFree(d_phi);
    cudaFree(d_Tnext);
    cudaFree(d_phinext);

    auto end = Clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Total time (s): " << seconds << "\n";

    return 0;
}

// next -- create an error check setup -- write a .cpp script to compare two csv files
// introduce a cuda event setup to properly time it -- tabulate performance in the grid-block space
// profile with nsight system and write cuda graphs
