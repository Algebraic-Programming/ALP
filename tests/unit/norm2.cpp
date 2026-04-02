
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
#include <sstream>
#include <cmath>
#include <complex>

#include <graphblas.hpp>
#include <graphblas/algorithms/norm.hpp>


using namespace grb;

void grb_program( const size_t &n, grb::RC &rc ) {
	// create test vector
	grb::Vector< double > x( n );
	
	// test 1: zero vector
	rc = grb::set( x, 0.0 );
	if( rc != grb::SUCCESS ) {
		std::cerr << "\t test 1 (zero vector): initialisation FAILED\n";
		return;
	}
	
	double norm = 0.0;
	grb::Semiring<
		grb::operators::add< double >, grb::operators::mul< double >,
		grb::identities::zero, grb::identities::one
	> ring;
	
	rc = grb::algorithms::norm2( norm, x, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 1 (zero vector): norm2 FAILED\n";
		return;
	}
	
	if( !utils::equals( norm, 0.0, 1 ) ) {
		std::cerr << "\t test 1 (zero vector): unexpected output "
			<< "( " << norm << ", expected 0.0 )\n";
		rc = FAILED;
		return;
	}
	
	// test 2: unit vector (all ones)
	rc = grb::set( x, 1.0 );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 2 (unit vector): initialisation FAILED\n";
		return;
	}
	
	norm = 0.0;
	rc = grb::algorithms::norm2( norm, x, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 2 (unit vector): norm2 FAILED\n";
		return;
	}
	
	const double expected_norm2 = std::sqrt( static_cast< double >( n ) );
	if( !utils::equals( norm, expected_norm2, n ) ) {
		std::cerr << "\t test 2 (unit vector): unexpected output "
			<< "( " << norm << ", expected " << expected_norm2 << " )\n";
		rc = FAILED;
		return;
	}
	
	// test 3: 3-4-5 triangle (should give norm 5 for vector [3, 4])
	if( n >= 2 ) {
		rc = grb::clear( x );
		rc = rc ? rc : grb::setElement( x, 3.0, 0 );
		rc = rc ? rc : grb::setElement( x, 4.0, 1 );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 3 (3-4-5 triangle): initialisation FAILED\n";
			return;
		}
		
		norm = 0.0;
		rc = grb::algorithms::norm2( norm, x, ring );
		if( rc != SUCCESS ) {
			std::cerr << "\t test 3 (3-4-5 triangle): norm2 FAILED\n";
			return;
		}
		
		if( !utils::equals( norm, 5.0, 3 ) ) {
			std::cerr << "\t test 3 (3-4-5 triangle): unexpected output "
				<< "( " << norm << ", expected 5.0 )\n";
			rc = FAILED;
			return;
		}
	}
	
	// test 4: accumulation into existing value
	norm = 10.0; // start with 10
	rc = grb::set( x, 1.0 );
	rc = rc ? rc : grb::algorithms::norm2( norm, x, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 4 (accumulation): norm2 FAILED\n";
		return;
	}
	
	const double expected_accum = 10.0 + std::sqrt( static_cast< double >( n ) );
	if( !utils::equals( norm, expected_accum, n + 10 ) ) {
		std::cerr << "\t test 4 (accumulation): unexpected output "
			<< "( " << norm << ", expected " << expected_accum << " )\n";
		rc = FAILED;
		return;
	}
	
	// test 5: sparse vector
	rc = grb::clear( x );
	for( size_t i = 0; i < n; i += 2 ) {
		rc = rc ? rc : grb::setElement( x, 2.0, i );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t test 5 (sparse vector): initialisation FAILED\n";
		return;
	}
	
	norm = 0.0;
	rc = grb::algorithms::norm2( norm, x, ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 5 (sparse vector): norm2 FAILED\n";
		return;
	}
	
	const size_t nnz = (n + 1) / 2; // ceiling division
	const double expected_sparse = 2.0 * std::sqrt( static_cast< double >( nnz ) );
	if( !utils::equals( norm, expected_sparse, 2 * nnz ) ) {
		std::cerr << "\t test 5 (sparse vector): unexpected output "
			<< "( " << norm << ", expected " << expected_sparse << " )\n";
		rc = FAILED;
		return;
	}
	
	// test 6: complex vector (3+4i at each position)
	grb::Vector< std::complex< double > > cx( n );
	rc = grb::set( cx, std::complex< double >( 3.0, 4.0 ) );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 6 (complex vector): initialisation FAILED\n";
		return;
	}
	
	grb::Semiring<
		grb::operators::add< std::complex< double > >,
		grb::operators::mul< std::complex< double > >,
		grb::identities::zero, grb::identities::one
	> complex_ring;
	
	double cnorm = 0.0;
	rc = grb::algorithms::norm2( cnorm, cx, complex_ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 6 (complex vector): norm2 FAILED\n";
		return;
	}
	
	// For complex number z = 3+4i, |z|^2 = 3^2 + 4^2 = 25, so |z| = 5
	// For n such numbers, norm2 = sqrt(n * 25) = 5 * sqrt(n)
	const double expected_cnorm = 5.0 * std::sqrt( static_cast< double >( n ) );
	if( !utils::equals( cnorm, expected_cnorm, 5 * n ) ) {
		std::cerr << "\t test 6 (complex vector): unexpected output "
			<< "( " << cnorm << ", expected " << expected_cnorm << " )\n";
		rc = FAILED;
		return;
	}
	
	// test 7: sparse complex vector
	rc = grb::clear( cx );
	for( size_t i = 0; i < n; i += 3 ) {
		rc = rc ? rc : grb::setElement( cx, std::complex< double >( 1.0, 1.0 ), i );
	}
	if( rc != SUCCESS ) {
		std::cerr << "\t test 7 (sparse complex vector): initialisation FAILED\n";
		return;
	}
	
	cnorm = 0.0;
	rc = grb::algorithms::norm2( cnorm, cx, complex_ring );
	if( rc != SUCCESS ) {
		std::cerr << "\t test 7 (sparse complex vector): norm2 FAILED\n";
		return;
	}
	
	// For complex number z = 1+i, |z|^2 = 1^2 + 1^2 = 2, so |z| = sqrt(2)
	// For (n+2)/3 such numbers, norm2 = sqrt(nnz_complex * 2)
	const size_t nnz_complex = (n + 2) / 3; // ceiling division
	const double expected_cnorm_sparse = std::sqrt( 2.0 * static_cast< double >( nnz_complex ) );
	if( !utils::equals( cnorm, expected_cnorm_sparse, 2 * nnz_complex ) ) {
		std::cerr << "\t test 7 (sparse complex vector): unexpected output "
			<< "( " << cnorm << ", expected " << expected_cnorm_sparse << " )\n";
		rc = FAILED;
		return;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << "\n";

	int error = 0;
	size_t n = 100;
	if( argc > 1 ) {
		n = atoi( argv[ 1 ] );
	}

	grb::Launcher< AUTOMATIC > launcher;
	grb::RC out;

	if( launcher.exec( &grb_program, n, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		error = 255;
	}
	if( error == 0 && out != SUCCESS ) {
		std::cerr << "Test returned error: " << grb::toString( out ) << "\n";
		error = 1;
	}

	if( ! error ) {
		std::cout << "Test OK\n" << std::endl;
	} else {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n" << std::endl;
	}

	return error;
}
