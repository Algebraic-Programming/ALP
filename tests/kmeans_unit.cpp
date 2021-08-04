
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

#include <iostream>
#include <utility>

#include <graphblas/algorithms/kmeans.hpp>

#include "graphblas.hpp"

using namespace grb;

#define ERR( ret, fun )    \
	ret = ret ? ret : fun; \
	assert( ret == SUCCESS );

// sample data

static const size_t I_X[ 34 ] = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };
static const size_t J_X[ 34 ] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16 };
static const double V_X[ 34 ] = { -2, 8, -1, 8, 0, 8, -1, 9, 0, 9, 0, 10, 6, 5, 7, 5, 8, 5, 6, 4, 7, 4, 0, 3, -1, 3, 0, 2, -1, 2, 0, 0, -2, 0 };
// static const double     V_X[ 34 ] = {-2, 5, -1, 5, 0, 5, -1, 6, 0, 6, 0, 7, 3, 5, 4, 5, 5, 5, 3, 4, 4,  4,  0,  3, -1,  3,  0,  2, -1,  2,  0,  0, -2,  0 };

static const size_t I_K[ 6 ] = { 0, 0, 1, 1, 2, 2 };
static const size_t J_K[ 6 ] = { 0, 1, 0, 1, 0, 1 };
static const double V_K[ 6 ] = { -1, 4, 0, 4, 1, 5 };

// graphblas program
void grbProgram( const void *, const size_t in_size, grb::RC & ret ) {
	if( in_size != 0 ) {
		std::cerr << "Unit tests called with unexpected input\n";
		ret = FAILED;
		return;
	}

	const size_t n = 17;
	const size_t m = 2;
	const size_t k = 3;
	const size_t nelts_X = 34;
	const size_t nelts_K = 6;

	grb::Matrix< double > X( m, n );
	grb::Matrix< double > K( k, m );
	grb::Vector< std::pair< size_t, double > > classes_and_centroids( n );

	ERR( ret, grb::resize( X, nelts_X ) );
	ERR( ret, grb::resize( K, nelts_K ) );
	ERR( ret, grb::buildMatrixUnique( X, I_X, J_X, V_X, nelts_X, SEQUENTIAL ) );
	ERR( ret, grb::algorithms::kpp_initialisation( K, X ) );
	ERR( ret, grb::algorithms::kmeans_iteration( K, classes_and_centroids, X ) );

#ifdef _DEBUG
	for( const auto & pair : classes_and_centroids ) {
		std::cout << "\tpoint " << pair.first << "\tcluster " << pair.second.first << "\tsquared distance " << pair.second.second << "\n";
	}
#endif
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	grb::RC rc = SUCCESS;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, rc ) != SUCCESS ) {
		std::cerr << "Test failed to launch\n";
		rc = FAILED;
	}
	if( rc == SUCCESS ) {
		std::cout << "Test OK.\n";
		return 0;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED." << std::endl;
		return 255;
	}
}
