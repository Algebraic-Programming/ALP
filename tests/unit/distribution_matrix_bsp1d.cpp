/*
 *   Copyright 2024 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <iomanip>

#include <graphblas.hpp>

using namespace grb;

template<typename LpfData>
void print_local(const LpfData& lpf_data, std::stringstream& local_ss) {
	for (size_t p = 0; p < lpf_data.P; ++p) {
		spmd<BSP1D>::barrier();
		if (p == lpf_data.s) {
			if( local_ss.str().empty() ) {
				std::cerr << "Process " << lpf_data.s << ":  [nothing to print]" << std::endl;
			} else {
				std::cerr << "Process " << lpf_data.s << ":\n" << local_ss.str() << std::endl;
			}
		}
		usleep(1000);
	}
	local_ss.clear();
}

template<typename D>
void grb_program(const size_t& n, RC& rc) {
	rc = SUCCESS;
	Matrix<D, BSP1D> I_distributed(n, n, n); {
		// Build the identity matrix
		std::vector<D> values(n, 1);
		std::vector<size_t> iota_indices(n, 0);
		std::iota(iota_indices.begin(), iota_indices.end(), 0);
		rc = buildMatrixUnique(
			I_distributed, iota_indices.data(), iota_indices.data(), values.data(), n, SEQUENTIAL
		);
	}


	// Each process now have to check if the global
	// coordinates match, since ti should always be: i == j
	const Matrix<D, reference>& local_matrix = internal::getLocal(I_distributed);
	const auto& lpf_data = internal::grb_BSP1D.cload();
	//
	std::stringstream local_ss; {
		// Check for the CRS
		const auto& crs = internal::getCRS(local_matrix);
		for (size_t i = 0; i < nrows(local_matrix); ++i) {
			for (size_t k = crs.col_start[i]; k < crs.col_start[i + 1]; ++k) {
				const auto j = crs.row_index[k]; // Local AND global since the distribution is 1D

				const size_t col_pid = internal::Distribution<>::offset_to_pid(j, ncols(I_distributed), lpf_data.P );
				const size_t col_off = internal::Distribution<>::local_offset(ncols(I_distributed), col_pid, lpf_data.P );
				const auto global_i  = internal::Distribution<>::local_index_to_global(i, nrows(I_distributed), lpf_data.s, lpf_data.P );
				const auto global_j  = internal::Distribution<>::local_index_to_global(j - col_off, ncols(I_distributed), col_pid, lpf_data.P );

				if (global_i != global_j) {
					local_ss << "  Wrong coordinate in CRS found at: ( "
					    << std::setw(3) << i << ", " << std::setw(3) << global_j << " )  --(mapped to global)-->  ( "
					    << std::setw(3) << global_i << ", "<< std::setw(3)  << global_j << " )\n";
					rc = FAILED;
				}
			}
		}
	}

	print_local(lpf_data, local_ss);

     { // Check for the CCS
         const auto &ccs = internal::getCCS( local_matrix );
         for( size_t j = 0; j < ncols(local_matrix); ++j ) {
             for( size_t k = ccs.col_start[j]; k < ccs.col_start[j+1]; ++k ) {
                 const auto i = ccs.row_index[k];

             	const size_t col_pid = internal::Distribution<>::offset_to_pid(j, ncols(I_distributed), lpf_data.P );
             	const size_t col_off = internal::Distribution<>::local_offset(ncols(I_distributed), col_pid, lpf_data.P );
             	const auto global_i  = internal::Distribution<>::local_index_to_global(i, nrows(I_distributed), lpf_data.s, lpf_data.P );
             	const auto global_j  = internal::Distribution<>::local_index_to_global(j - col_off, ncols(I_distributed), col_pid, lpf_data.P );

                 if( global_i != global_j ) {
                     local_ss << "  Wrong coordinate in CCS found at: ( "
                         << std::setw(3) << i << ", " << std::setw(3) << global_j << " )  --(mapped to global)-->  ( "
                         << std::setw(3) << global_i << ", " << std::setw(3) << global_j << " )\n";
                     rc = RC::FAILED;
                 }
             }
         }
     }

	print_local( lpf_data, local_ss );

	if (collectives<>::allreduce(rc, operators::any_or<RC>()) != SUCCESS) {
		rc = PANIC;
		return;
	}
}

int main(int argc, char** argv) {
	// defaults

	if (argc != 2) {
		std::cerr << "Usage: " << argv[0] << " <n>\n";
		return 1;
	}
	std::cout << "This is functional test " << argv[0] << "\n";

	Launcher<AUTOMATIC> launcher;
	RC out;
	size_t n = std::strtoul(argv[1], nullptr, 10);

	const RC launch_rc = launcher.exec(&grb_program<int>, n, out, true);
	if (launch_rc != SUCCESS) {
		std::cerr << "Launch test failed\n";
		out = launch_rc;
	}

	if (out != SUCCESS) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << toString(out) << ")\n" << std::endl;
		return 1;
	}

	std::cout << "Test OK\n" << std::endl;
	return 0;
}
