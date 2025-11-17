// This file is part of the ALP GraphBLAS repository and inherits the repository license.
// Eigen (header-only) is used without modification. New code here follows repository licensing.
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "mm_loader.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cstring>

using Clock = std::chrono::steady_clock;

struct RunStats {
    double mean_iter_ms = 0.0;
};

struct AggregateStats {
    double mean = 0.0;
    double stddev = 0.0;
    double stderr = 0.0; // standard error
};

static void usage(const char *prog) {
    std::cerr << "Usage: " << prog << " <matrix.mtx> [--niter N] [--repetitions R] [--max-iter M] [--tolerance T] [--fixed]" << std::endl;
}

// Manual CG for benchmarking. If fixed mode: always run exactly NITER iterations regardless of convergence.
RunStats run_cg(const Eigen::SparseMatrix<double> &A,
               const Eigen::VectorXd &b,
               int max_iter,
               double tol,
               int niter_fixed,
               bool fixed_mode) {
    const int n = A.rows();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b - A * x;
    Eigen::VectorXd p = r;
    double rsold = r.dot(r);

    std::vector<double> iter_times_ms;
    iter_times_ms.reserve(fixed_mode ? niter_fixed : max_iter);

    int iterations = fixed_mode ? niter_fixed : max_iter;
    for(int it = 0; it < iterations; ++it) {
        auto t0 = Clock::now();
        // Corresponds to GraphBLAS steps in `conjugate_gradient.hpp` main loop:
        //   temp = A * u;                         // grb::mxv<descr_dense>(temp, A, u, ring)
        //   beta = (A * u)' * u;                  // grb::dot<descr_dense>(beta, temp, u, ...)
        //   alpha = sigma / beta;                 // grb::apply<descr>(alpha, sigma, beta, divide)
        //   x = x + alpha * u;                    // grb::eWiseMul<descr_dense>(x, alpha, u, ring)
        //   r = r - alpha .* temp;                // grb::foldr<descr_dense>(alpha, temp, ...) + grb::foldl<descr_dense>(r, temp, minus)
        // In this benchmark we do not use preconditioning (z==r), so sigma == rsold.
        Eigen::VectorXd Ap = A * p; // Ap == temp (A * u)
        // p.dot(Ap) == (A*u)' * u  (GraphBLAS 'beta' in the loop above)
        double alpha = rsold / p.dot(Ap); // alpha = sigma / beta (since sigma == rsold)
        x += alpha * p; // x = x + alpha * u
        r -= alpha * Ap; // r = r - alpha .* temp
        // Compute new residual (r' * r) — corresponds to grb::dot<descr_dense>(alpha, r, r, ...)
        double rsnew = r.dot(r);
        if(!fixed_mode) {
            double rel = std::sqrt(rsnew) / b.norm();
            auto t1 = Clock::now();
            iter_times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if(rel < tol) {
                break; // convergence reached
            }
        } else {
            auto t1 = Clock::now();
            iter_times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        // Update search direction:
        // In GraphBLAS loop this is handled via:
        //   if (preconditioned) { ... } else { beta = alpha; }
        //   alpha = beta / sigma;    // (different reuse of names in GraphBLAS)
        //   u_next = z + beta * u_previous;  // here z==r, so u_next = r + beta * u
        //   sigma = beta; (sigma holds r' * r for next iter)
        // The compact Eigen form below implements the non-preconditioned update:
        double beta = rsnew / rsold; // beta = (r' * r) / (previous r' * r)
        p = r + beta * p; // p is the search direction (u_next = r + beta * u)
        rsold = rsnew; // sigma <- beta (for next iteration, expressed as rsold)
    }

    double sum = 0.0;
    for(double v : iter_times_ms) sum += v;
    RunStats stats;
    stats.mean_iter_ms = iter_times_ms.empty() ? 0.0 : sum / static_cast<double>(iter_times_ms.size());
    return stats;
}

AggregateStats aggregate(const std::vector<RunStats> &runs) {
    AggregateStats agg;
    if(runs.empty()) return agg;
    std::vector<double> means;
    means.reserve(runs.size());
    for(const auto &r : runs) means.push_back(r.mean_iter_ms);
    double sum = 0.0;
    for(double m : means) sum += m;
    agg.mean = sum / means.size();
    double var = 0.0;
    for(double m : means) var += (m - agg.mean) * (m - agg.mean);
    var /= means.size();
    agg.stddev = std::sqrt(var);
    agg.stderr = agg.stddev / std::sqrt(means.size());
    return agg;
}

int main(int argc, char **argv) {
    if(argc < 2) { usage(argv[0]); return 1; }
    std::string matrix_path;
    int niter_fixed = 256; // benchmark fixed iterations
    int repetitions = 32;
    int max_iter = 2000;
    double tol = 1e-8;
    bool fixed_mode = false; // when true: ignore convergence, always run niter_fixed iterations

    matrix_path = argv[1];
    for(int i=2; i<argc; ++i) {
        if(std::strcmp(argv[i], "--niter") == 0 && i+1 < argc) {
            niter_fixed = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--repetitions") == 0 && i+1 < argc) {
            repetitions = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--max-iter") == 0 && i+1 < argc) {
            max_iter = std::atoi(argv[++i]);
        } else if(std::strcmp(argv[i], "--tolerance") == 0 && i+1 < argc) {
            tol = std::atof(argv[++i]);
        } else if(std::strcmp(argv[i], "--fixed") == 0) {
            fixed_mode = true;
        } else if(std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    try {
        auto A = alp_bench::load_matrix_market(matrix_path);
        if(A.rows() != A.cols()) {
            std::cerr << "Warning: matrix is not square; CG may not be valid.\n";
        }
        int n = A.rows();
        // Random b with deterministic seed for repeatability
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        Eigen::VectorXd b(n);
    for(int i=0; i<n; ++i) b(i) = dist(gen);

        std::vector<RunStats> runs;
        runs.reserve(repetitions);
        for(int r=0; r<repetitions; ++r) {
            runs.emplace_back(run_cg(A, b, max_iter, tol, niter_fixed, fixed_mode));
        }
        auto agg = aggregate(runs);
        std::cout << "matrix=" << matrix_path
                  << ", mode=" << (fixed_mode?"fixed":"solve")
                  << ", niter_fixed=" << niter_fixed
                  << ", repetitions=" << repetitions
                  << ", mean_iter_ms=" << agg.mean
                  << ", stddev=" << agg.stddev
                  << ", stderr=" << agg.stderr
                  << std::endl;
    } catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
