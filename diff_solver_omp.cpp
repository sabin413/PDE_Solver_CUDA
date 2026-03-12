#include <vector>
#include <iostream>
#include <omp.h>
#include <fstream>   // <-- added
#include <chrono>    // <-- added
using namespace std;



double helper_func(double x) {
    if (x < 0.0) return 1.0;
    else         return 0.0;
}


void update_one_grid(int i, vector<double>& T, vector<double>& phi, vector<double>& term1, vector<double>& Tnext, vector<double>& phinext,double dt, double param, double lambda_){
    int N = T.size();
    term1[i] = -(1.0 - phi[i]) * T[i] *helper_func(T[i]);
    phinext[i] = phi[i] + dt * term1[i];
    const double lap = T[i+1] + T[i-1] - 2.0 * T[i];
    const double term2 = param * lap + dt * lambda_ * term1[i];
    Tnext[i] = T[i] + term2;    
}



// grid size is 1.0

int main() {
    // parameters
    const double lambda_ = 5.0;
    const double param   = 0.01;
    const double dt      = 0.01;
    const size_t N       = 1000000;
    const int    steps   = 100000; // no. of time steps - // for parallization
    
    // int no_of_threads = 10;

    // initialize/allocate
    vector<double> T(N, 0.0);
    vector<double> phi(N, 0.0);
    T[0] = -1.0;
    vector<double> term1(N, 0.0);
    vector<double> Tnext(N, 0.0);
    vector<double> phinext(N, 0.0);

    auto start = chrono::steady_clock::now(); 

    // CSV output setup (two files) ---
    ofstream tfile("temp_profiles_cpu.csv");
    ofstream phifile("phi_profiles_cpu.csv");

    // optional: consistent numeric formatting
    tfile.setf(ios::fixed); tfile.precision(6);
    phifile.setf(ios::fixed); phifile.precision(6);

    // write step 0 (initial state)
    tfile << 0;
    for (size_t i = 0; i < N; ++i) tfile << ',' << T[i];
    tfile << '\n';

    phifile << 0;
    for (size_t i = 0; i < N; ++i) phifile << ',' << phi[i];
    phifile << '\n';
    
    // Parallellization
    //int no_of_threads = 10;
    
    #pragma omp parallel
    {
        #pragma omp single
        cout << "Total threads: " << omp_get_num_threads() << std::endl; // omp single ends right here

        for (int t = 0; t < steps; ++t) {

            // Parallelize the spatial loop.
            // The implicit barrier at the end of omp for ensures all threads finished Tnext/phinext updates.
            #pragma omp for schedule(static)
            for (int s = 1; s < N - 1; ++s) {
                update_one_grid(s, T, phi, term1, Tnext, phinext, dt, param, lambda_);
            }
	    // implicit barrier here

            // One thread applies boundary conditions and swaps buffers.
            // 'single' has an implicit barrier at the end, so next timestep won't start early.
            #pragma omp single
            {
                Tnext[N - 1] = Tnext[N - 2];
                Tnext[0]     = -1.0;
                T.swap(Tnext);
                phi.swap(phinext);
            }
        }
    } 
    auto end = chrono::steady_clock::now(); 
    chrono::duration<double> elapsed_seconds = end - start; 
    cout << "Total time (s): " << elapsed_seconds.count() << '\n'; 

    // files close automatically on destruction, but explicit close is fine:
    // tfile.close(); phifile.close();

    return 0;
}

