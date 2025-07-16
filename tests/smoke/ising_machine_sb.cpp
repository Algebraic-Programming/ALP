
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


// test data from python implementation
constexpr std::size_t N = 10;
constexpr std::size_t Nz = 54;
static const size_t i_arr[ Nz ] = {
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 
    4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9
};
static const size_t j_arr[ Nz ] = { 
    0, 2, 3, 4, 5, 1, 4, 5, 6, 7, 9, 0, 2, 4, 6, 9, 0, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 
    3, 4, 0, 1, 3, 5, 6, 8, 1, 2, 3, 5, 6, 1, 3, 7, 8, 9, 3, 5, 7, 8, 1, 2, 3, 7, 9
};
static const double v_arr[ Nz ] = { 
    -1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1, -1, -1, -1,  
    1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1
};

static const double  h_arr[ N ] = { 1, -1,  1, -1,  1,  1, -1,  1,  1,  1 };
static const double x_arr[ N ] = { -0.0996, -0.0315,  0.0572,  0.0630,  0.0087, -0.0143, -0.0170, -0.0411, 0.0433, -0.0298 };
static const double y_arr[ N ] = {  0.0373,  0.0540,  0.0486, -0.0877, -0.0418, -0.0261,  0.0018, -0.0710, 0.0507, -0.0483 };

const std::size_t num_iters = 100;
static const double energies_ref[ num_iters ] = { 
    -3,  -3,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
    -1,  -9, -11, -11, -11, -11, -11, -11, -11, -11,  -9,  -9, -11,
    -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9, -11, -11, -11,  -9,  -9,  -9,  -9,  -9,  -9,  -9,  -9,
    -9,  -9,  -9,  -9,  -9,  -9,  -9, -11, -11, -11, -11, -11, -11,
    -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
    -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
    -11, -11, -11, -11, -11, -11, -11, -11, -11
};

using IOType = double;
using JType = double;


int main( int argc, char ** argv ) {

    /* --- Initialise ALP/GraphBLAS --- */
    if( grb::init() != grb::SUCCESS ) {
        std::cerr << "ALP init failed\n";
        return -1;
    }

    /* --- Problem setup (toy) --- */
    grb::Matrix<IOType> J( N, N, Nz );
    grb::Vector<IOType> h( N );
    grb::Vector<IOType> x0( N ), y0( N ); // initialy 
    // ... populate J with test (random) values
    grb::RC rc = grb::SUCCESS;
    rc = rc ? rc : buildMatrixUnique( J, &( i_arr[ 0 ] ), &( j_arr[ 0 ] ), &( v_arr[ 0 ] ), Nz, grb::SEQUENTIAL );

    if(rc != grb::SUCCESS) {
        std::cerr << "matrix build failed\n";
        return grb::RC::PANIC;
    }

    // Fill h, x0, y0 with random values using buildVector
    rc = rc ? rc : buildVector(h, h_arr, h_arr + N, grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(x0, x_arr, x_arr + N, grb::SEQUENTIAL);
    rc = rc ? rc : buildVector(y0, y_arr, y_arr + N, grb::SEQUENTIAL);
    if(rc != grb::SUCCESS) {
        std::cerr << "Vector build failed\n";
        return grb::RC::PANIC;
    }

    const IOType p0  = 0.;
    const IOType p1  = 1.1;
    const IOType dt  = 0.25;

    // energies is array of length num_iters, initialized to 0
    std::vector< IOType > energies( num_iters, 0 );

	grb::Vector< IOType > Jx( N );
    grb::Vector< IOType > temp( N );
	grb::Vector< bool > mask( N );
	grb::Vector< IOType > sol( N );

    rc = rc ? rc : bSB(
        energies, x0, y0, J, h, p0, p1, num_iters, dt,
        Jx, temp, mask, sol
    );


    if( rc != grb::SUCCESS ) {
        std::cerr << "bSB returned error code " << rc << '\n';
    } else {
        // print all energies
        for (std::size_t i = 0; i < num_iters; ++i) {
           std::cout << "Energy at iteration " << i << " = " << energies[i] << '\n';
           if( energies[i] != energies_ref[i]) {
                std::cerr << "Error: Energy at iteration " << i << " does not match reference value.\n";
                std::cerr << "Expected: " << energies_ref[i] << ", got: " << energies[i] << '\n';
                return -1;  
           }
        }
    }

    grb::finalize();
    return 0;
}
