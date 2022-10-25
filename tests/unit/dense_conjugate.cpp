
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
#include <string>
#include <type_traits>
#include <vector>
#include <memory>
#include <complex>

#include <alp.hpp>
#include "../utils/print_alp_containers.hpp"

typedef float BaseScalarType;
constexpr BaseScalarType tol = 1.e-10;

template< typename T >
T random_value();

template<>
BaseScalarType random_value< BaseScalarType >() {
	return static_cast< BaseScalarType >( rand() ) / RAND_MAX;
}

template<>
std::complex< BaseScalarType > random_value< std::complex< BaseScalarType > >() {
	const BaseScalarType re = random_value< BaseScalarType >();
	const BaseScalarType im = random_value< BaseScalarType >();
	return std::complex< BaseScalarType >( re, im );
}

template< typename MatrixType >
void init_matrix( MatrixType &M ) {
	// Temporary until proper matrix building is implemented
	typedef typename MatrixType::value_type value_type;
	alp::internal::setInitialized( M, true );
	const size_t height = alp::ncols( M );
	const size_t width = alp::nrows( M );
	for( size_t r = 0; r < height; ++r ) {
		for( size_t c = 0; c < width; ++c ) {
			const value_type val = random_value< value_type >();
			if( r < c ) {
				alp::internal::access( M, alp::internal::getStorageIndex( M, r, c ) ) = val;
				if( r != c ) {
					alp::internal::access( M, alp::internal::getStorageIndex( M, c, r ) ) = grb::utils::is_complex< value_type >::conjugate( val );
				}
			} else if ( r == c ) {
				alp::internal::access( M, alp::internal::getStorageIndex( M, r, c ) ) = std::real( val );
			}
		}
	}
}

template< typename T >
std::vector< T > generate_random_data(
	size_t N
) {
	std::vector< T > data( N );
	for( size_t i = 0; i < N; ++i ) {
		const T val = random_value< T >();
		data[ i ] = val;
	}
	return data;
}

template<
	typename MatrixType1,
	typename MatrixType2,
	typename T = typename MatrixType1::value_type,
	typename Ring
>
alp::RC check_if_same( const MatrixType1 &A, const MatrixType2 &B, const Ring &ring ) {

	alp::RC rc = alp::SUCCESS;

	alp::Matrix< T, alp::structures::Square > E( nrows( A ) );
	rc = rc ? rc : set( E, alp::Scalar< T >( ring.template getZero< T >() ) );

	rc = rc ? rc : alp::foldl( E, A, ring.getAdditiveOperator() );
	rc = rc ? rc : alp::foldl( E, B, alp::operators::subtract< T >() );

	BaseScalarType fnorm = ring.template getZero< BaseScalarType >();
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm, &ring ]( const size_t i, const size_t j, T &val ) {
			(void) i;
			(void) j;
			const BaseScalarType valsquare = std::norm( val );
			alp::internal::foldl(
				fnorm,
				valsquare,
				alp::operators::add< BaseScalarType >()
			);
		},
		E
	);
	fnorm = std::sqrt( fnorm );

	if( fnorm < tol ) {
		return alp::SUCCESS;
	} else {
		return alp::FAILED;
	}
}

template<
	typename T,
	typename Structure = typename std::conditional<
		grb::utils::is_complex< T >::value,
		alp::structures::Hermitian,
		// Should be Symmetric.
		// Temporarily using Square until fold is fixed to support folding symmetric onto more general structures
		alp::structures::Square
	>::type
>
alp::RC test_conjugate_matrix( const size_t n ) {

	const alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	alp::RC rc = alp::SUCCESS;

	// create the original matrix
	alp::Matrix< T, Structure > H( n, n );
	// set matrix elements using the internal interface
	init_matrix( H );

	// create a conjugated matrix
	auto H_conj = alp::conjugate( H );

	// create a transpose view over original matrix (used for error checking)
	auto H_T = alp::get_view< alp::view::transpose >( H );

	// check if conjugated and transposed matrix are the same
	rc = rc ? rc : check_if_same( H_conj, H_T, ring );

	return rc;
}

template<
	typename T
>
alp::RC test_conjugate_vector( const size_t n ) {

	const alp::Semiring< alp::operators::add< T >, alp::operators::mul< T >, alp::identities::zero, alp::identities::one > ring;

	alp::RC rc = alp::SUCCESS;

	std::srand( 1 );
	auto randdata = generate_random_data< T >( n );
	alp::Vector< T > x( n );
	rc = rc ? rc : alp::buildVector( x, randdata.begin(), randdata.end() );

	auto x_conj = conjugate( x );

	print_vector( " x ", x );
	print_vector( " x* ", x_conj );


	// // check if conjugated and transposed matrix are the same
	// rc = rc ? rc : check_if_same( H_conj, H_T, ring );

	return rc;
}

void alp_program( const size_t &n, alp::RC &rc ) {

	rc = alp::SUCCESS;

	rc = rc ? rc : test_conjugate_matrix< std::complex< BaseScalarType > >( n );
	rc = rc ? rc : test_conjugate_matrix< BaseScalarType >( n );
	rc = rc ? rc : test_conjugate_vector< std::complex< BaseScalarType > >( n );
	rc = rc ? rc : test_conjugate_vector< BaseScalarType >( n );

}

int main( int argc, char ** argv ) {
	// defaults
	bool printUsage = false;
	size_t in = 5;

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
		} else if( read % 2 != 0 ) {
			std::cerr << "Given value for n is odd\n";
			printUsage = true;
		} else {
			// all OK
			in = read;
		}
	}
	if( printUsage ) {
		std::cerr << "Usage: " << argv[ 0 ] << " [n]\n";
		std::cerr << "  -n (optional, default is 100): an even integer, the "
					 "test size.\n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< alp::AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != alp::SUCCESS ) {
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
