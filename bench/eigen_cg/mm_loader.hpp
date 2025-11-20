// This file is part of the ALP GraphBLAS repository and inherits the repository license.
// It uses Eigen headers (header-only) without modification; Eigen retains its own license
// in extern/eigen. We do NOT duplicate Eigen's license here.
#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace alp_bench {

// Simple Matrix Market (.mtx) loader for real sparse matrices in coordinate format.
// Assumptions:
//  - 1-based indices in file -> converted to 0-based
//  - "coordinate" format; symmetric matrices may have 'symmetric' banner but we do not
//    auto-complete entries; we assume full explicit listing.
//  - Ignores comments and blank lines.
//  - Returns a RowMajor sparse matrix so Eigen's OpenMP SpMV kernels can engage.
// Throws std::runtime_error on parse errors.
using SparseMatrixRM = Eigen::SparseMatrix<double, Eigen::RowMajor>;
SparseMatrixRM load_matrix_market(const std::string &path);

} // namespace alp_bench
