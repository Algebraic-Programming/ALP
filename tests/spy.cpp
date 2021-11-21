
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

#include <graphblas.hpp>
#include <graphblas/algorithms/spy.hpp>
#include <graphblas/utils/parser/MatrixFileReader.hpp>


void grb_program( const void * const fn_p, const size_t fn_length, grb::RC & rc ) {
	const char * const fn = static_cast< const char * >(fn_p);
	if( strnlen( fn, fn_length ) >= fn_length ) {
		std::cerr << "Error: non-NULL terminated string passed as input file\n";
		rc = grb::ILLEGAL;
		return;
	}

	grb::utils::MatrixFileReader< double > reader( fn );
	const size_t m = reader.m();
	const size_t n = reader.n();
	grb::Matrix< double > original( m, n );
	rc = grb::buildMatrixUnique( original, reader.cbegin(), reader.cend(), grb::PARALLEL );
	if( rc != grb::SUCCESS ) {
		std::cerr << "Initialisation FAILED\n";
		return;
	}

	// check if call to spy succeeds
	const size_t p = m / 8 + ((m % 8) > 0 ? 1 : 0);
	const size_t q = n / 8 + ((n % 8) > 0 ? 1 : 0);
	grb::Matrix< double > spy( p, q );
	rc = rc ? rc : grb::algorithms::spy( spy, original );

	// print nonzero count of spy to screen to enable 3rd-party verification
	if( rc == grb::SUCCESS && grb::spmd<>::pid() == 0 ) {
		std::cout << "Spy matrix of " << p << " by " << q << " pixels has " << grb::nnz( spy ) << " nonzeroes, versus " << grb::nnz( original ) << " nonzeroes in the original " << m << " by " << n << " matrix\n";
	}
	for( const auto &nonzero : spy ) {
		if( nonzero.second <= 0 ) {
			std::cout << "Invalid entry at spy( " << nonzero.first.first << ", " << nonzero.first.second << " ): " << nonzero.second << ", expected something strictly larger than 0\n";
			rc = grb::FAILED;
		}
	}

	// check if call to spy with normalize=true succeeds and yields a consistent number of nonzeroes
	grb::Matrix< double > spy2( p, q );
	rc = rc ? rc : grb::algorithms::spy< true >( spy2, original );
	if( rc == grb::SUCCESS && grb::nnz( spy ) != grb::nnz( spy2 ) ) {
		std::cerr << "Unexpected number of nonzeroes for spy2: " << grb::nnz(spy2) << ", expected " << grb::nnz(spy) << "\n";
		rc = grb::FAILED;
	}
	for( const auto &nonzero : spy2 ) {
		if( nonzero.second <= 0 || nonzero.second > 1 ) {
			std::cout << "Invalid entry at spy2( " << nonzero.first.first << ", " << nonzero.first.second << " ): " << nonzero.second << ", expected a value x in the range 0 < x <= 1\n";
			rc = grb::FAILED;
		}
	}

	// check if normalisation was successful
	grb::Semiring<
		grb::operators::add< double >,
		grb::operators::mul< double >,
		grb::identities::zero,
		grb::identities::one
	> ring;
	if( rc == grb::SUCCESS ) {
		grb::Matrix< double > chk( p, q );
		rc = rc ? rc : grb::resize( chk, grb::nnz( spy ) );
		rc = rc ? rc : grb::eWiseApply( chk, spy, spy2, ring.getMultiplicativeOperator() );
		if( rc == grb::SUCCESS && grb::nnz( chk ) != grb::nnz( spy ) ) {
			std::cerr << "Unexpected number of nonzeroes for chk: " << grb::nnz(chk) << ", expected " << grb::nnz(spy) << "\n";
			rc = grb::FAILED;
		}
		for( const auto &triple : chk ) {
			if( !grb::utils::equals( triple.second, 1.0, 1 ) ) {
				std::cerr << "Verification of normalised spy failed at ( " << triple.first.first << ", " << triple.first.second << " ): " << triple.second << ", expected 1\n";
				rc = grb::FAILED;
			}
		}
	}

	// check if we can have pattern input to spy
	grb::Matrix< void > pattern( m, n );
	rc = rc ? rc : grb::resize( pattern, grb::nnz( original ) );
	rc = rc ? rc : grb::buildMatrixUnique( pattern, reader.cbegin(), reader.cend(), grb::PARALLEL );
	rc = rc ? rc : grb::algorithms::spy( spy2, pattern );
	if( rc == grb::SUCCESS && grb::nnz( spy2 ) != grb::nnz( spy ) ) {
		std::cerr << "Unexpected number of nonzeroes for spy2 (from pattern matrix): " << grb::nnz( spy2 ) << ", expected " << grb::nnz( spy ) << "\n";
		rc = grb::FAILED;
	}

	// also check re-entrance by reusing spy2 (we should overwrite)
	rc = rc ? rc : grb::algorithms::spy< true >( spy2, pattern );
	if( rc == grb::SUCCESS && grb::nnz( spy2 ) != grb::nnz( spy ) ) {
		std::cerr << "Unexpected number of nonzeroes for spy2 (from pattern matrix, normalised): " << grb::nnz( spy2 ) << ", expected " << grb::nnz( spy ) << "\n";
		rc = grb::FAILED;
	}
	if( rc == grb::SUCCESS ) {
		grb::Matrix< double > chk( p, q );
		rc = rc ? rc : grb::resize( chk, nnz( spy ) );
		rc = rc ? rc : grb::eWiseApply( chk, spy, spy2, ring.getMultiplicativeOperator() );
		if( rc == grb::SUCCESS && grb::nnz( chk ) != grb::nnz( spy ) ) {
			std::cerr << "Unexpected number of nonzeroes for chk (pattern): " << grb::nnz(chk) << ", expected " << grb::nnz(spy) << "\n";
			rc = grb::FAILED;
		}
		for( const auto &triple : chk ) {
			if( !grb::utils::equals( triple.second, 1.0, 1 ) ) {
				std::cerr << "Verification of normalised spy (pattern) failed at ( " << triple.first.first << ", " << triple.first.second << " ): " << triple.second << ", expected 1\n";
				rc = grb::FAILED;
			}
		}
	}

	// check if we can have boolean input to spy
	grb::Matrix< bool > boolean( m, n );
	rc = rc ? rc : grb::resize( boolean, grb::nnz( original ) );
	rc = rc ? rc : grb::set< grb::descriptors::structural >( boolean, pattern, true );
	rc = rc ? rc : grb::algorithms::spy( spy2, boolean );
	if( rc == grb::SUCCESS && grb::nnz( spy2 ) != grb::nnz( spy ) ) {
		std::cerr << "Unexpected number of nonzeroes for spy2 (from boolean matrix): " << grb::nnz( spy2 ) << ", expected " << grb::nnz( spy ) << "\n";
		rc = grb::FAILED;
	}

	// also check re-entrance
	rc = rc ? rc : grb::algorithms::spy< true >( spy2, boolean );
	if( rc == grb::SUCCESS && grb::nnz( spy2 ) != grb::nnz( spy ) ) {
		std::cerr << "Unexpected number of nonzeroes for spy2 (from boolean matrix, normalised): " << grb::nnz( spy2 ) << ", expected " << grb::nnz( spy ) << "\n";
		rc = grb::FAILED;
	}
	if( rc == grb::SUCCESS ) {
		grb::Matrix< double > chk( p, q );
		rc = rc ? rc : grb::resize( chk, nnz( spy ) );
		rc = rc ? rc : grb::eWiseApply( chk, spy, spy2, ring.getMultiplicativeOperator() );
		if( rc == grb::SUCCESS && grb::nnz( chk ) != grb::nnz( spy ) ) {
			std::cerr << "Unexpected number of nonzeroes for chk (boolean): " << grb::nnz(chk) << ", expected " << grb::nnz(spy) << "\n";
			rc = grb::FAILED;
		}
		for( const auto &triple : chk ) {
			if( !grb::utils::equals( triple.second, 1.0, 1 ) ) {
				std::cerr << "Verification of normalised spy (boolean) failed at ( " << triple.first.first << ", " << triple.first.second << " ): " << triple.second << ", expected 1\n";
				rc = grb::FAILED;
			}
		}
	}
}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;

	// error checking
	if( argc < 2 || argc > 2 ) {
		printUsage = true;
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [matrix file]\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, static_cast< const void * >(argv[ 1 ]), strlen( argv[ 1 ] ) + 1, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

