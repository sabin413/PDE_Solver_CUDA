#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using Clock = std::chrono::steady_clock;
// initialize T and phi at t = 0 (host)
void initial_state(size_t N, vector<double>& T, vector<double>& phi){
    T.assign(N, 0.0);
    phi.assign(N, 0.0);
    if (N > 0) T[0] = -1.0;
}

// usable from both host and device
__host__ __device__
double helper_func(double T) {
    if (T < 0.0) return 1.0;
    else         return 0.0;
}
// can it lead to thread divergence? Yes! Think about it.

// CUDA kernel: one time step on device arrays
__global__
void step_once_kernel(const double* T, const double* phi,
                      double* Tnext, double* phinext,
                      double dt, double param, double lambda_, size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 1) term1 for ALL i
    double term1 = -(1.0 - phi[i]) * T[i] * helper_func(T[i]);

    // 2) update phi for ALL i
    phinext[i] = phi[i] + dt * term1;

    // 3) update 
    if (i == 0) {
	    Tnext[i] = T[i]; // left boundary untouched
    }
    else if (i == N-1) {
	    Tnext[i] = T[i-1]; // no flux
    }
    else {
        const double lap = (T[i+1] + T[i-1] - 2.0 * T[i]);
        const double term2 = param * lap + dt * lambda_ * term1;
        Tnext[i] = T[i] + term2;
    }
}

// --- device storage and simple wrappers ---

double *d_T = nullptr, *d_phi = nullptr;
double *d_Tnext = nullptr, *d_phinext = nullptr;
size_t d_N = 0;
int threadsPerBlock = 256;
int blocks = 0; // no. of blocks required will be calculated later

// initialize for device
void init_device(size_t N, const vector<double>& T, const vector<double>& phi) {
    d_N = N;
    cudaMalloc(&d_T,       N * sizeof(double));
    cudaMalloc(&d_phi,     N * sizeof(double));
    cudaMalloc(&d_Tnext,   N * sizeof(double));
    cudaMalloc(&d_phinext, N * sizeof(double));

    cudaMemcpy(d_T,   T.data(),   N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    blocks = (static_cast<int>(N) + threadsPerBlock - 1) / threadsPerBlock;
}

void step_once_device(double dt, double param, double lambda_) {
    step_once_kernel<<<blocks, threadsPerBlock>>>(d_T, d_phi,
                                                  d_Tnext, d_phinext,
                                                  dt, param, lambda_, d_N);
    // swap pointers on device
    std::swap(d_T, d_Tnext);
    std::swap(d_phi, d_phinext);
}

void copy_device_to_host(vector<double>& T, vector<double>& phi) {
    cudaMemcpy(T.data(),   d_T,   d_N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(phi.data(), d_phi, d_N * sizeof(double), cudaMemcpyDeviceToHost);
}

void cleanup_device() {
    cudaFree(d_T);
    cudaFree(d_phi);
    cudaFree(d_Tnext);
    cudaFree(d_phinext);
}

// grid size is 1.0

auto start = Clock::now();

int main() {
    // parameters
    const double lambda_ = 5.0;
    const double param   = 0.01;
    const double dt      = 0.01;
    const size_t N       = 100000; // 10 - 100k
    const int    steps   = 100000;   // 1 - 10M time steps
    const int    output_interval = 50000; // copy back & write every 0.1 - 1M steps

    vector<double> T, phi; // initialize T and phi before using them
    initial_state(N, T, phi); // simply assigns values to T and phi

    // initialize GPU with initial state
    init_device(N, T, phi);

    // --- minimal CSV output setup (two files) ---
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
        step_once_device(dt, param, lambda_);

        // every 1M steps: copy back and write
        if ((s + 1) % output_interval == 0) {
            const int out_step = s + 1;

            copy_device_to_host(T, phi);

            tfile << out_step;
            for (size_t i = 0; i < N; ++i) tfile << ',' << T[i];
            tfile << '\n';

            phifile << out_step;
            for (size_t i = 0; i < N; ++i) phifile << ',' << phi[i];
            phifile << '\n';
        }

        if (s % 10000 == 0) { // progress print every 100k steps
            cout << "step: " << s << '\n';
        }
    }

    cudaDeviceSynchronize();
    cleanup_device();

    auto end = Clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();
    std::cout << "Total time (s): " << seconds << "\n";

    return 0;
}

