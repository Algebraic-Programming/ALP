
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

#include <utility>
#include <iostream>
#include <set>

#include "alp.hpp"

template< typename Container >
bool has_no_conflict( const size_t k, const size_t i, const size_t j, const Container &indices ) {

	if( indices.find( k ) != indices.end() ) {
		std::cerr << "Coordinate ( " << i << ", " << j << " ) maps to the same location as another coordinate\n";
		return false;
	} else {
		return true;
	}
}

bool is_within_bounds( const size_t k, const size_t i, const size_t j, const size_t storage_dimensions ) {

	if( k >= storage_dimensions ) {
		std::cerr << "Coordinate ( " << i << ", " << j << " ) maps outside of storage bounds [0, " << storage_dimensions << ").\n";
		return false;
	} else {
		return true;
	}
}

bool maps_to_more( const size_t stored, const size_t storage_dimensions ) {

	if( stored > storage_dimensions ) {
		std::cerr << "Polynomial maps to more elements than the claimed amount of " << storage_dimensions << "elements.\n";
		return true;
	} else {
		return false;
	}
}

bool maps_to_less( const size_t stored, const size_t storage_dimensions ) {

	if( stored < storage_dimensions ) {
		std::cerr << "Polynomial maps to less elements than the claimed amount of " << storage_dimensions << "elements.\n";
		return true;
	} else {
		return false;
	}
}

// alp program
void alpProgram( const size_t &n, alp::RC &rc ) {

	const size_t m = 2 * n;

	// Test full storage of size m * n
	{
		typedef alp::storage::polynomials::FullFactory<> Factory;
		const auto poly = Factory::Create( m, n );
		const size_t storage_dimensions = Factory::GetStorageDimensions( m, n );
		std::set< size_t > indices;
		for( size_t i = 0; i < m; ++i ) {
			for( size_t j = 0; j < n; ++j ) {
				const size_t k = poly.evaluate( i, j );
				if( !has_no_conflict( k, i, j, indices ) ) {
					return;
				}
				if( !is_within_bounds( k, i, j, storage_dimensions ) ) {
					return;
				}
				indices.insert( k );
			}
		}
		if( maps_to_less( indices.size(), storage_dimensions ) ) {
			return;
		}
		if( maps_to_more( indices.size(), storage_dimensions ) ) {
			return;
		}
	}

	// Test packed storage of size n * n storing upper triangular portion row-wise
	{
		typedef alp::storage::polynomials::PackedFactory< alp::storage::UPPER, alp::storage::ROW_WISE > Factory;
		const auto poly = Factory::Create( n, n );
		const size_t storage_dimensions = Factory::GetStorageDimensions( n, n );
		std::set< size_t > indices;
		for( size_t i = 0; i < n; ++i ) {
			for( size_t j = i; j < n; ++j ) {
				const size_t k = poly.evaluate( i, j );
				if( !has_no_conflict( k, i, j, indices ) ) {
					return;
				}
				if( !is_within_bounds( k, i, j, storage_dimensions ) ) {
					return;
				}
				indices.insert( k );
			}
		}
		if( maps_to_less( indices.size(), storage_dimensions ) ) {
			return;
		}
		if( maps_to_more( indices.size(), storage_dimensions ) ) {
			return;
		}
	}

	// Test array storage of size n * 1
	{
		typedef alp::storage::polynomials::ArrayFactory Factory;
		const auto poly = Factory::Create( n, 1 );
		const size_t storage_dimensions = Factory::GetStorageDimensions( n, 1 );
		std::set< size_t > indices;
		for( size_t i = 0; i < n; ++i ) {
			const size_t k = poly.evaluate( i, 0 );
			if( !has_no_conflict( k, i, 0, indices ) ) {
				return;
			}
			if( !is_within_bounds( k, i, 0, storage_dimensions ) ) {
				return;
			}
			indices.insert( k );
		}
		if( maps_to_less( indices.size(), storage_dimensions ) ) {
			return;
		}
		if( maps_to_more( indices.size(), storage_dimensions ) ) {
			return;
		}
	}

	rc = alp::SUCCESS;

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 100;

	// error checking
	if( argc > 2 ) {
		printUsage = true;
	}
	if( argc == 2 ) {
		size_t read;
		std::istringstream ss( argv[ 1 ] );
		if( ! ( ss >> read ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( ! ss.eof() ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if( read == 0 ) {
			std::cerr << "n must be a positive number\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an integer, the "
					 "test size.\n";
		return 1;
	}
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alpProgram, in, out, true ) != alp::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != alp::SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

