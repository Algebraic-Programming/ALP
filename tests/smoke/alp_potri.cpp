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

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef _COMPLEX
#include <complex>
#include <cmath>
#include <iomanip>
#endif

#include <alp.hpp>
#include <graphblas/utils/iscomplex.hpp> // use from grb
#include <alp/algorithms/symherm_posdef_inverse.hpp>
#include <alp/utils/parser/MatrixFileReader.hpp>
#ifdef DEBUG
#include "../utils/print_alp_containers.hpp"
#endif

using namespace alp;

using BaseScalarType = double;

#ifdef _COMPLEX
using ScalarType = std::complex< BaseScalarType >;
//not fully implemented structures
using HermitianOrSymmetricPD = structures::HermitianPositiveDefinite;
#else
using ScalarType = BaseScalarType;
//fully implemented structures
using HermitianOrSymmetricPD = structures::SymmetricPositiveDefinite;
#endif
constexpr BaseScalarType tol = 1.e-10;
constexpr size_t RNDSEED = 1;

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


struct inpdata {
	std::string fname = "";
	size_t N = 0;
};


/** Generate full storage Symmetric or Hermitian
 *   positive definite matrix for in-place tests
 */
template< typename T >
void generate_symmherm_pos_def_mat_data_full(
	size_t N,
	std::vector< T > &mat_data
) {
	std::fill( mat_data.begin(), mat_data.end(), static_cast< T >( 0 ) );
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			mat_data[ i * N + j ] = random_value< T >();
			mat_data[ j * N + i ] += grb::utils::is_complex< T >::conjugate( mat_data[ i * N + j ] );
			if( i == j ) {
				mat_data[ j * N + i ] += static_cast< T >( N );
			}
		}
	}
}

/** Generate symmetric-hermitian positive
 *  definite matrix in a full-storage container
 */
template< typename T >
void generate_symmherm_pos_def_mat_data(
	size_t N,
	std::vector< T > &mat_data,
	const typename std::enable_if<
		grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	generate_symmherm_pos_def_mat_data_full< T >( N, mat_data );
}

/** Generate upper/lower triangular part of a
 *  Symmetric positive definite matrix
 */
template< typename T >
void generate_symmherm_pos_def_mat_data(
	size_t N,
	std::vector< T > &mat_data,
	const typename std::enable_if<
		!grb::utils::is_complex< T >::value,
		void
	>::type * const = nullptr
) {
	std::fill( mat_data.begin(), mat_data.end(), static_cast< T >( 0 ) );
	size_t k = 0;
	for( size_t i = 0; i < N; ++i ) {
		for( size_t j = i; j < N; ++j ) {
			mat_data[ k ] = random_value< T >();
			if( i == j ) {
				mat_data[ k ] += grb::utils::is_complex< T >::conjugate( mat_data[ k ] );
				mat_data[ k ] += static_cast< T >( N );
			}
			++k;
		}
	}
}


//** check the solution by calculating the Frobenius norm of (I - H^-1 x H) */
template<
	typename MatH,
	typename D = typename MatH::value_type,
	typename Ring = Semiring< operators::add< D >, operators::mul< D >, identities::zero, identities::one >,
	typename Minus = operators::subtract< D >,
	std::enable_if_t<
		is_matrix< MatH >::value &&
		// TODO: structures::Symmetric should be replced
		//       rewith structures::SymmetricPositiveDefinite
		(
			(
				!grb::utils::is_complex< D >::value &&
				structures::is_a< typename MatH::structure, structures::Symmetric >::value
			) || (
				grb::utils::is_complex< D >::value &&
				structures::is_a< typename MatH::structure, structures::Hermitian >::value
			)
		) &&
		is_semiring< Ring >::value &&
		is_operator< Minus >::value
	> * = nullptr
>
alp::RC check_inverse_solution(
	const MatH &Hinv,
	const MatH &H,
	const Ring &ring = Ring(),
	const Minus &minus = Minus()
) {
	alp::RC rc = SUCCESS;
	const Scalar< D > zero( ring.template getZero< D >() );
	const Scalar< D > one( ring.template getOne< D >() );
	const size_t N = nrows( H );

	alp::Matrix< D, structures::Square, Dense > HxHinv( N );
	rc = rc ? rc : alp::set( HxHinv, zero );
	rc = rc ? rc : alp::mxm( HxHinv, H, Hinv, ring );
#ifdef DEBUG
	print_matrix( std::string("  HxHinv  "), HxHinv );
#endif

	auto HxHinvdiag = alp::get_view< alp::view::diagonal >( HxHinv );
	rc = rc ? rc : foldl( HxHinvdiag, one, minus );

	//Frobenius norm
	D fnorm = 0;
	rc = rc ? rc : alp::eWiseLambda(
		[ &fnorm ]( const size_t i, const size_t j, D &val ) {
			(void) i;
			(void) j;
			fnorm += val * val;
		},
		HxHinv
	);
	fnorm = std::sqrt( fnorm );
#ifdef DEBUG
	std::cout << " FrobeniusNorm(I - H^-1 x H) = " << fnorm << "\n";
#endif
	if( tol < std::abs( fnorm ) ) {
		std::cout << "The Frobenius norm is too large. "
			"Make sure that you have used SPD matrix as input.\n";
		return FAILED;
	}

	return rc;
}

void alp_program( const inpdata &unit, alp::RC &rc ) {
	rc = SUCCESS;

	alp::Semiring<
		alp::operators::add< ScalarType >,
		alp::operators::mul< ScalarType >,
		alp::identities::zero,
		alp::identities::one
	> ring;
	const alp::Scalar< ScalarType > zero_scalar( ring.getZero< ScalarType >() );
	const alp::Scalar< ScalarType > one_scalar( ring.getOne< ScalarType >() );

	size_t N = 0;
	if( !unit.fname.empty() ) {
		alp::utils::MatrixFileReader< ScalarType > parser_A( unit.fname );
		N = parser_A.n();
		if( !parser_A.isSymmetric() ) {
			std::cout << "Symmetric matrix epxected as input!\n";
			rc = ILLEGAL;
			return;
		}
	} else if( unit.N != 0 )  {
		N = unit.N;
	}

	alp::Matrix< ScalarType, HermitianOrSymmetricPD, Dense > H( N );
	alp::Matrix< ScalarType, HermitianOrSymmetricPD, Dense > Hinv( N );

	if( !unit.fname.empty() ) {
		alp::utils::MatrixFileReader< ScalarType > parser_A( unit.fname );
		rc = rc ? rc : alp::buildMatrix( H, parser_A.begin(), parser_A.end() );
	} else if( unit.N != 0 )  {
		std::srand( RNDSEED );
		std::vector< ScalarType > matrix_data( ( N * ( N + 1 ) ) / 2 );
		// Hermitian is currently using full storage
		if( grb::utils::is_complex< ScalarType >::value ) {
			matrix_data.resize( N * N );
		}
		generate_symmherm_pos_def_mat_data< ScalarType >( N, matrix_data );
		rc = rc ? rc : alp::buildMatrix( H, matrix_data.begin(), matrix_data.end() );
	}

	if( !internal::getInitialized( H ) ) {
		std::cout << " Matrix H is not initialized\n";
		return;
	}

#ifdef DEBUG
	print_matrix( std::string(" << H >> "), H );
#endif

	rc = rc ? rc : algorithms::symherm_posdef_inverse( Hinv, H, ring );
 	rc = rc ? rc : check_inverse_solution( Hinv, H, ring );
}

int main( int argc, char **argv ) {
	// defaults
	bool printUsage = false;
	inpdata in;

	// error checking
	if( argc == 3 ) {
		std::string readflag;
		std::istringstream ss1( argv[ 1 ] );
		std::istringstream ss2( argv[ 2 ] );
		if( ! ( ( ss1 >> readflag ) &&  ss1.eof() ) ) {
			std::cerr << "Error parsing first argument\n";
			printUsage = true;
		} else if(
			( readflag != std::string( "-fname" ) ) &&
			( readflag != std::string( "-n" ) )
		) {
			std::cerr << "Given first argument is unknown\n";
			printUsage = true;
		} else {
			if( readflag == std::string( "-fname" ) ) {
				if( ! ( ( ss2 >> in.fname ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing second argument\n";
					printUsage = true;
				}
			}

			if( readflag == std::string( "-n" ) ) {
				if( ! ( ( ss2 >> in.N ) &&  ss2.eof() ) ) {
					std::cerr << "Error parsing second argument\n";
					printUsage = true;
				}
			}

		}
	} else {
		std::cout << "Wrong number of arguments\n" ;
		printUsage = true;
	}

	if( printUsage ) {
		std::cerr << "Usage: \n";
		std::cerr << "       " << argv[ 0 ] << " -fname FILENAME.mtx \n";
		std::cerr << "      or  \n";
		std::cerr << "       " << argv[ 0 ] << " -n N \n";
		return 1;
	}

	std::cout << "This is functional test " << argv[ 0 ] << "\n";
	alp::Launcher< AUTOMATIC > launcher;
	alp::RC out;
	if( launcher.exec( &alp_program, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		std::cerr << "Test FAILED (" << alp::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}
