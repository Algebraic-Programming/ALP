
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
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

#include <exception>
#include <iostream>
#include <vector>


#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/algorithms/ising_machine_sb.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

#include <utils/output_verification.hpp>


using namespace grb;
using namespace algorithms;



using IOType = double;

int main( int argc, char ** argv ) {
    constexpr std::size_t N = 100;
    constexpr std::size_t Nz = 400;

    /* --- Initialise ALP/GraphBLAS --- */
    if( grb::init() != grb::SUCCESS ) {
        std::cerr << "ALP init failed\n";
        return -1;
    }

    /* --- Problem setup (toy) --- */
    grb::Matrix<IOType> J( N, N, Nz );
    grb::Vector<IOType> h( N );
    grb::Vector<IOType> x0( N ), y0( N ); // initialy 
    // ... populate J with random values
    // generate random nonzero locations and values for J
    std::vector<std::size_t> row_indices(Nz);
    std::vector<std::size_t> col_indices(Nz);
    std::vector<IOType> nonzero_values(Nz);
    // TODO: make sure there are not duplicate entries in J and J is symmetric

    IOType sumJ2_test = 0;
    for( std::size_t i = 0; i < Nz; ++i ) {
        row_indices[i] = rand() % N;
        col_indices[i] = rand() % N;
        nonzero_values[i] = static_cast<IOType>(rand()) / RAND_MAX; // random value between 0 and 1
        sumJ2_test+=nonzero_values[i]*nonzero_values[i];
    }
    std::cout << "sumJ2_test: " << sumJ2_test << '\n';
    IOType xi_test = 0.5 / std::sqrt( sumJ2_test / static_cast<IOType>( N - 1 )  );
    std::cout << "xi_test: " << xi_test << '\n';

    grb::RC rc = grb::SUCCESS;
    rc = rc ? rc : buildMatrixUnique(J, row_indices.data(), col_indices.data(), nonzero_values.data(), Nz, grb::SEQUENTIAL);
    if(rc != grb::SUCCESS) {
        std::cerr << "matrix build failed\n";
        return grb::RC::PANIC;
    }

    // Fill h, x0, y0 with random values using buildVector
    std::vector<IOType> h_buffer(N), x0_buffer(N), y0_buffer(N);
    for( std::size_t i = 0; i < N; ++i ) {
        h_buffer[i]  = static_cast<IOType>(rand()) / RAND_MAX;
        x0_buffer[i] = static_cast<IOType>(rand()) / RAND_MAX / 0.2 - 0.1;
        y0_buffer[i] = static_cast<IOType>(rand()) / RAND_MAX / 0.2 - 0.1;
    }
    rc = rc ? rc : buildVector(h, h_buffer.begin(), h_buffer.end(), grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(x0, x0_buffer.begin(), x0_buffer.end(), grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(y0, y0_buffer.begin(), y0_buffer.end(), grb::SEQUENTIAL);
    if(rc != grb::SUCCESS) {
        std::cerr << "Vector build failed\n";
        return grb::RC::PANIC;
    }

    const IOType p0  = 0.1;
    const IOType p1  = 1.0;
    const std::size_t num_iters = 1000;
    const IOType dt  = 0.01;

    // energies is array of length num_iters, initialized to 0
    std::vector< IOType > energies( num_iters, 0 );

    rc = rc ? rc : bSB(
        energies, x0, y0, J, h, p0, p1, num_iters, dt
    );

    if( rc != grb::SUCCESS ) {
        std::cerr << "bSB returned error code " << rc << '\n';
    } else {
        std::cout << "Final energy = " << energies[num_iters-1] << '\n';
    }

    grb::finalize();
    return 0;
}
