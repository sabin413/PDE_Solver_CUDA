#include <vector>
#include <iostream>
#include <fstream>   // <-- added
#include <chrono>    // <-- added
using namespace std;

// initialize T and phi at t = 0
void initial_state(size_t N, vector<double>& T, vector<double>& phi){
    T.assign(N, 0.0);
    phi.assign(N, 0.0);
    T[0] = -1.0;
}
// size_t: standard unsigned integer data type -- practically it forces N to be// a non negative number, 0 is fine
// passing by references above to make sure that the the arrays are
// eddited inplace, as opposed to editing their local copies. & T is just
// an alias to T thought, it allows inplace edit while T does not
// A function that accepts the aliases to T and phi, allowing inplace edits

// if I were to define the above function using pointers instead
// A function that accepts the pointers to the first elements of T and phi
// as arguments
// T[i] is *(T+i) under the hood -- pointer arithmatics
// void initial_state(size_t N, double* T, double* phi) {
//    for (size_t i = 0; i < N; ++i) {
//        T[i] = 0.0; // *(T+i) -> move the address by i*size_of_double from the base address of T (first element) and dereference that address (read/edit the value there)
//        phi[i] = 0.0;
//    }
//    T[0] = -1.0;
//}


double helper_func(double T) {
    if (T < 0.0) return 1.0;
    else         return 0.0;
}

void step_once(vector<double>& T, vector<double>& phi,
               double dt, double param, double lambda_) {
    const size_t N = T.size();
    //if (phi.size() != N || N < 2) return;

    vector<double> term1(N, 0.0);
    vector<double> Tnext = T;      // start from current; keep boundaries unchanged
    vector<double> phinext = phi;

    // 1) term1 for ALL i
    for (size_t i = 0; i < N; ++i) {
        term1[i] = -(1.0 - phi[i]) * T[i] * helper_func(T[i]);
    }

    // 2) update phi for ALL i
    for (size_t i = 0; i < N; ++i) {
        phinext[i] = phi[i] + dt * term1[i];
    }

    // 3) term2 and 4) update T only for interior points (no mirroring)
    for (size_t i = 1; i + 1 < N; ++i) {
        const double lap = (T[i+1] + T[i-1] - 2.0 * T[i]);
        const double term2 = param * lap + dt * lambda_ * term1[i];
        Tnext[i] = T[i] + term2;
    }

    // commit
    Tnext[N-1] = Tnext[N-2];
    T.swap(Tnext);
    phi.swap(phinext);
}

// grid size is 1.0

int main() {
    // parameters
    const double lambda_ = 5.0;
    const double param   = 0.01;
    const double dt      = 0.01;
    const size_t N       = 100000;
    const int    steps   = 100000; // no. of time steps -- N = 100k, steps = 100k, this takes 1 min to finish: if I increase the no of time steps to 10^7 (10M), it takes 100 min to finish. GPU acceleration? The arithmatic intensity of this algorithm is quite low as the number if FLOPS scales linearly with N as does the memory. 

    vector<double> T, phi;
    initial_state(N, T, phi);

    auto start = std::chrono::steady_clock::now(); // <-- added

    // --- minimal CSV output setup (two files) ---
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
    // --------------------------------------------

    for (int s = 0; s < steps; ++s) {
        step_once(T, phi, dt, param, lambda_);

        // write every 1000 steps AFTER the update: steps 1000, 2000, ..., 10000
        if ((s + 1) % 500000 == 0) {
            const int out_step = s + 1;

            tfile << out_step; // write to (file) operator
            for (size_t i = 0; i < N; ++i) tfile << ',' << T[i];
            tfile << '\n';

            phifile << out_step;
            for (size_t i = 0; i < N; ++i) phifile << ',' << phi[i];
            phifile << '\n';
        }

        if (s % 100000 == 0) {
            cout << "step: " << s << '\n';
        }
    }

    auto end = std::chrono::steady_clock::now(); // <-- added
    std::chrono::duration<double> elapsed = end - start; // <-- added
    cout << "Total time (s): " << elapsed.count() << '\n'; // <-- added

    // files close automatically on destruction, but explicit close is fine:
    // tfile.close(); phifile.close();

    return 0;
}

