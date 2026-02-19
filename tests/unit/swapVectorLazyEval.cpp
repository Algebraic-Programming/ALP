
/*
 *   Copyright 2026 Huawei Technologies Co., Ltd.
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

/**
 * Unit test for Issue #408: std::swap for nonblocking vectors was broken due to
 * lazy evaluation system holding stale pointers after swap.
 *
 * This test specifically exercises the lazy evaluation pipeline to ensure that
 * std::swap properly flushes pending operations before swapping vectors.
 */

#include <iostream>
#include <sstream>
#include <cmath>

#include <graphblas.hpp>


using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	// Create vectors for testing
	grb::Vector< double > x( n ), y( n ), z( n );
	
	// Initialize x with value 1.5
	rc = grb::set( x, 1.5 );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation of x FAILED\n";
		return;
	}
	
	// Initialize y with value 2.5
	rc = grb::set( y, 2.5 );
	if( rc != SUCCESS ) {
		std::cerr << "\tinitialisation of y FAILED\n";
		return;
	}
	
	// Verify initial state
	if( grb::nnz( x ) != n || grb::nnz( y ) != n ) {
		std::cerr << "\tinitialisation FAILED: unexpected nnz\n";
		rc = FAILED;
		return;
	}
	
	// Perform operations that may get queued in lazy evaluation
	// These operations should be associated with the correct vectors
	rc = grb::eWiseApply( z, x, y, grb::operators::add< double >() );
	if( rc != SUCCESS ) {
		std::cerr << "\teWiseApply before swap FAILED\n";
		return;
	}
	
	// Verify z = x + y = 1.5 + 2.5 = 4.0
	double sum_before = 0.0;
	for( const auto & pair : z ) {
		sum_before = pair.second;
		break;
	}
	if( std::abs( sum_before - 4.0 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in z before swap: " << sum_before << ", expected 4.0\n";
		rc = FAILED;
		return;
	}
	
	// Now swap x and y
	// Without the fix, this could leave stale pointers in the lazy evaluation system
	std::swap( x, y );
	
	// Verify swap worked: x should now have 2.5, y should have 1.5
	double x_val = 0.0, y_val = 0.0;
	for( const auto & pair : x ) {
		x_val = pair.second;
		break;
	}
	for( const auto & pair : y ) {
		y_val = pair.second;
		break;
	}
	
	if( std::abs( x_val - 2.5 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in x after swap: " << x_val << ", expected 2.5\n";
		rc = FAILED;
		return;
	}
	if( std::abs( y_val - 1.5 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in y after swap: " << y_val << ", expected 1.5\n";
		rc = FAILED;
		return;
	}
	
	// Check nnz after swap
	if( grb::nnz( x ) != n ) {
		std::cerr << "\tunexpected nnz in x after swap: " << grb::nnz( x ) << ", expected " << n << "\n";
		rc = FAILED;
		return;
	}
	if( grb::nnz( y ) != n ) {
		std::cerr << "\tunexpected nnz in y after swap: " << grb::nnz( y ) << ", expected " << n << "\n";
		rc = FAILED;
		return;
	}
	
	// Perform more operations after swap
	// These should use the swapped vectors correctly
	rc = grb::eWiseApply( z, x, y, grb::operators::add< double >() );
	if( rc != SUCCESS ) {
		std::cerr << "\teWiseApply after swap FAILED\n";
		return;
	}
	
	// z should still be x + y = 2.5 + 1.5 = 4.0 (same sum, different order)
	double sum_after = 0.0;
	for( const auto & pair : z ) {
		sum_after = pair.second;
		break;
	}
	if( std::abs( sum_after - 4.0 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in z after swap and operation: " << sum_after << ", expected 4.0\n";
		rc = FAILED;
		return;
	}
	
	// Additional test: multiply operations
	rc = grb::eWiseApply( z, x, y, grb::operators::mul< double >() );
	if( rc != SUCCESS ) {
		std::cerr << "\teWiseApply multiply after swap FAILED\n";
		return;
	}
	
	// z should be x * y = 2.5 * 1.5 = 3.75
	double prod = 0.0;
	for( const auto & pair : z ) {
		prod = pair.second;
		break;
	}
	if( std::abs( prod - 3.75 ) > 1e-9 ) {
		std::cerr << "\tunexpected product in z: " << prod << ", expected 3.75\n";
		rc = FAILED;
		return;
	}
	
	// Test with clear and new values
	rc = grb::clear( x );
	if( rc != SUCCESS ) {
		std::cerr << "\tclear x FAILED\n";
		return;
	}
	rc = grb::clear( y );
	if( rc != SUCCESS ) {
		std::cerr << "\tclear y FAILED\n";
		return;
	}
	
	// Set new values
	rc = grb::set( x, 10.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\tset x to 10.0 FAILED\n";
		return;
	}
	rc = grb::set( y, 20.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\tset y to 20.0 FAILED\n";
		return;
	}
	
	// Swap again
	std::swap( x, y );
	
	// Verify: x should be 20.0, y should be 10.0
	x_val = y_val = 0.0;
	for( const auto & pair : x ) {
		x_val = pair.second;
		break;
	}
	for( const auto & pair : y ) {
		y_val = pair.second;
		break;
	}
	
	if( std::abs( x_val - 20.0 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in x after second swap: " << x_val << ", expected 20.0\n";
		rc = FAILED;
		return;
	}
	if( std::abs( y_val - 10.0 ) > 1e-9 ) {
		std::cerr << "\tunexpected value in y after second swap: " << y_val << ", expected 10.0\n";
		rc = FAILED;
		return;
	}
	
	return;
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
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	std::cout << "Testing Issue #408: std::swap for nonblocking vectors with lazy evaluation\n";
	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
