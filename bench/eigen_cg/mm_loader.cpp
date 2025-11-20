// This file is part of the ALP GraphBLAS repository and inherits the repository license.
// Uses Eigen headers (unmodified) located in extern/eigen.
#include "mm_loader.hpp"
#include <iostream>

namespace alp_bench {

SparseMatrixRM load_matrix_market(const std::string &path) {
    std::ifstream in(path);
    if(!in) {
        throw std::runtime_error("Cannot open matrix file: " + path);
    }
    std::string line;
    // Header
    if(!std::getline(in, line)) {
        throw std::runtime_error("Empty matrix file: " + path);
    }
    if(line.find("MatrixMarket") == std::string::npos) {
        throw std::runtime_error("Invalid MatrixMarket header in: " + path);
    }
    // Parse header keywords to detect 'coordinate' vs 'array', 'pattern' vs 'real', and symmetry
    // Example header: %%MatrixMarket matrix coordinate pattern symmetric
    bool is_coordinate = false;
    bool is_pattern = false;
    bool is_symmetric = false;
    {
        std::istringstream hss(line);
        std::string banner, mtx, format, field, symmetry;
        hss >> banner >> mtx >> format >> field >> symmetry;
        if(format == "coordinate") is_coordinate = true;
        if(field == "pattern") is_pattern = true;
        if(symmetry == "symmetric") is_symmetric = true;
    }
    // Skip comments
    while(std::getline(in, line)) {
        if(line.empty()) continue;
        if(line[0] == '%') continue;
        break; // first non-comment line after header: size line
    }
    if(in.fail()) {
        throw std::runtime_error("Unexpected EOF before size line: " + path);
    }
    std::istringstream iss(line);
    size_t rows=0, cols=0, nnz=0;
    iss >> rows >> cols >> nnz;
    if(rows==0 || cols==0 || nnz==0) {
        throw std::runtime_error("Failed parsing size line: " + line);
    }
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    size_t count=0;
    while(std::getline(in, line)) {
        if(line.empty() || line[0]=='%') continue;
        std::istringstream iss_entry(line);
        size_t i,j; double v = 1.0;
        if(is_pattern) {
            // pattern format: only i j present
            if(!(iss_entry >> i >> j)) {
                throw std::runtime_error("Malformed entry line: " + line);
            }
            v = 1.0;
        } else {
            if(!(iss_entry >> i >> j >> v)) {
                throw std::runtime_error("Malformed entry line: " + line);
            }
        }
        // Convert 1-based to 0-based
        if(i==0 || j==0) {
            throw std::runtime_error("Encountered 0-based index in file (expected 1-based): " + line);
        }
        triplets.emplace_back(static_cast<int>(i-1), static_cast<int>(j-1), v);
        if(is_symmetric && i != j) {
            // add symmetric counterpart
            triplets.emplace_back(static_cast<int>(j-1), static_cast<int>(i-1), v);
        }
        ++count;
    }
    if(count != nnz) {
        std::cerr << "Warning: declared nnz=" << nnz << " but parsed " << count << " entries\n";
    }
    SparseMatrixRM A(static_cast<int>(rows), static_cast<int>(cols));
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

} // namespace alp_bench
