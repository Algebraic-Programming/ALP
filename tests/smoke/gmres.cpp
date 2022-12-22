
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
#include <iterator>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#ifdef _GMRES_COMPLEX
 #include <complex>
#endif
#include <inttypes.h>

#include <graphblas.hpp>
#include <graphblas/algorithms/gmres.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <utils/output_verification.hpp>


using BaseScalarType = double;
#ifdef _GMRES_COMPLEX
 using ScalarType = std::complex< BaseScalarType >;
#else
 using ScalarType = BaseScalarType;
#endif

constexpr BaseScalarType TOL = 1.e-9;

using namespace grb;
using namespace algorithms;

class input {
public:
	bool generate_random = true;
	size_t rep = grb::config::BENCHMARKING::inner();
	size_t max_iterations = 1;
	size_t n = 0;
	size_t nz_per_row = 10;
	std::string filename = "";
	std::string precond_filename = "";
	std::string rhs_filename = "";
	bool rhs = false;
	bool no_preconditioning = false;
	bool direct = true;
	size_t rep_outer = grb::config::BENCHMARKING::outer();
	BaseScalarType tol = TOL;
	BaseScalarType max_residual_norm = TOL;
	size_t gmres_restart = 10;
};

struct output {
	int rc;
	size_t rep;
	size_t iterations;
	size_t iterations_arnoldi;
	size_t iterations_gmres;
	double residual;
	double residual_relative;
	grb::utils::TimerResults times;
	double time_gmres;
	double time_x_update;
	double time_preamble;
	double time_io;
	double time_residual;
	PinnedVector< ScalarType > pinnedVector;
};

template< typename T > T random_value();

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

/**
 * Generate random lienar problem matrix data (A) and
 * the corresponding preconditioner matrix data (P).
 */
template< typename NonzeroType, typename DimensionType >
void generate_random_data(
	const DimensionType &n,
	const DimensionType &nz_per_row,
	std::vector< DimensionType > &MatAveci,
	std::vector< DimensionType > &MatAvecj,
	std::vector< NonzeroType > &MatAvecv,
	std::vector< DimensionType > &MatPveci,
	std::vector< DimensionType > &MatPvecj,
	std::vector< NonzeroType > &MatPvecv
) {
	Semiring<
		grb::operators::add< NonzeroType >, grb::operators::mul< NonzeroType >,
		grb::identities::zero, grb::identities::one
	> ring;
	operators::divide< NonzeroType > divide;
	const NonzeroType one = ring.template getOne< NonzeroType >();

	MatAveci.resize( n * nz_per_row );
	MatAvecj.resize( n * nz_per_row );
	MatAvecv.resize( n * nz_per_row );
	MatPveci.resize( n );
	MatPvecj.resize( n );
	MatPvecv.resize( n );

	DimensionType acout = 0;
	DimensionType pcout = 0;
	for( DimensionType i = 0; i < n; ++i ){
		for(
			DimensionType j = i - std::min( nz_per_row / 2, i );
			j < std::min( i + nz_per_row / 2, n );
			++j
		) {
			NonzeroType tmp = random_value< NonzeroType >();
			NonzeroType tmp2 = random_value< NonzeroType >();
			// make sure preconditioner is useful
			if( i == j ) {
				tmp2 = tmp2 * static_cast< NonzeroType >( n * n );
				tmp = tmp + tmp2;
			}
			MatAveci[ acout ] = i;
			MatAvecj[ acout ] = j;
			MatAvecv[ acout ] = tmp;
			++acout;
			if( i == j ) {
				MatPveci[ pcout ] = i;
				MatPvecj[ pcout ] = j;
				grb::foldr( one, tmp2, divide );
				MatPvecv[ pcout ] = tmp2;
				++pcout;
			}
		}
	}
	MatAveci.resize( acout );
	MatAvecj.resize( acout );
	MatAvecv.resize( acout );
}

/**
 * Build random matrix A and the preconditioner P.
 */
RC make_matrices(
	Matrix< ScalarType > &A,
	Matrix< ScalarType > &P,
	const size_t &n,
	const size_t &nz_per_row
){
	RC rc = SUCCESS;

	{
		std::vector< size_t > MatAveci;
		std::vector< size_t > MatAvecj;
		std::vector< ScalarType > MatAvecv;
		std::vector< size_t > MatPveci;
		std::vector< size_t > MatPvecj;
		std::vector< ScalarType > MatPvecv;
		std::srand( 0 );
		generate_random_data(
			n,
			nz_per_row,
			MatAveci,
			MatAvecj,
			MatAvecv,
			MatPveci,
			MatPvecj,
			MatPvecv
		);
		rc = rc ? rc : buildMatrixUnique( A, MatAveci.begin(), MatAvecj.begin(), MatAvecv.begin(), MatAveci.size(), SEQUENTIAL );
		rc = rc ? rc : buildMatrixUnique( P, MatPveci.begin(), MatPvecj.begin(), MatPvecv.begin(), MatPveci.size(), SEQUENTIAL );
	}

	return rc;
}

/**
 * Solves the  least linear square problem defined by vector H[1:n] x =  H[ 0 ],
 * using Givens rotations and backsubstitution. The results is stored in H[ 0 ],
 * which is sused to update GMRES solution, vector x.
 */
// todo: this is a temp implementations and
// should be replace by the equivalent ALP function.
template<
	typename NonzeroType,
	typename DimensionType
>
void hessolve(
	std::vector< std::vector< NonzeroType > > &H,
	const DimensionType n,
	const DimensionType &kspspacesize
) {
	std::vector< NonzeroType > rhs = H[ 0 ];

	size_t n_ksp = std::min( kspspacesize, n - 1 );

	// for i in range(n):
	for( size_t i = 0; i < n_ksp; ++i ) {
		NonzeroType a, b, c, s;

		// a,b=H[i:i+2,i]
		a = H[ i + 1 ][ i ];
		b = H[ i + 1 ][ i + 1 ];
		// tmp1=sqrt(norm(a)**2+norm(b)**2)
		NonzeroType tmp1 = std::sqrt(
			std::norm( a ) +
			std::norm( b )
		);
		c = grb::utils::is_complex< NonzeroType >::modulus( a ) / tmp1 ;
		if( std::norm( a ) != 0 ) {
			// s = a / std::norm(a) * std::conj(b) / tmp1;
			s = a / grb::utils::is_complex< ScalarType >::modulus( a )
				* grb::utils::is_complex< NonzeroType >::conjugate( b ) / tmp1;
		}
		else {
			// s = std::conj(b) / tmp1;
			s = grb::utils::is_complex< NonzeroType >::conjugate( b ) / tmp1;
		}

		NonzeroType tmp2;
		// for k in range(i,n):
		for( size_t k = i; k < n_ksp; ++k ) {
			// tmp2       =   s * H[i+1,k]
			tmp2 = s * H[ k + 1 ][ i + 1 ];
			// H[i+1,k] = -conjugate(s) * H[i,k] + c * H[i+1,k]
			H[ k + 1 ][ i + 1 ] = - grb::utils::is_complex< NonzeroType >::conjugate( s )
				* H[ k + 1 ][ i ] + c * H[ k + 1 ][ i + 1 ];
			// H[i,k]   = c * H[i,k] + tmp2
			H[ k + 1 ][ i ] = c * H[ k + 1 ][ i ] + tmp2;
		}

		// tmp3 = rhs[i]
		NonzeroType tmp3;
		tmp3 = rhs[ i ];
		// rhs[i] =  c * tmp3 + s * rhs[i+1]
		rhs[ i ]  =  c * tmp3 + s * rhs[ i + 1 ] ;
		// rhs[i+1]  =  -conjugate(s) * tmp3 + c * rhs[i+1]
		rhs[ i + 1 ]  =  - grb::utils::is_complex< NonzeroType >::conjugate( s )
			* tmp3 + c * rhs[ i + 1 ];

	}


#ifdef _DEBUG
	std::cout << "hessolve rhs vector before inversion, vector = ";
	for( size_t k = 0; k < n_ksp; ++k ) {
		std::cout << rhs[ k ] << " ";
	}
	std::cout << "\n";
#endif

	// for i in range(n-1,-1,-1):
	for( size_t m = 0; m < n_ksp; ++m ) {
	 	size_t i = n_ksp - 1 - m;
		// for j in range(i+1,n):
		for( size_t j = i + 1; j < n_ksp; ++j ) {
			// rhs[i]=rhs[i]-rhs[j]*H[i,j]
			rhs[ i ] = rhs[ i ] - rhs[ j ] * H[ j + 1 ][ i ];
		}
		// rhs[i]=rhs[i]/H[i,i]
		if( std::abs( H[ i + 1 ][ i ] ) < TOL ) {
			std::cout << "---> small number in hessolve\n";
		}
		rhs[ i ] = rhs[ i ] / H[ i + 1 ][ i ];
	}

	H[ 0 ] = rhs;
}



void grbProgram( const struct input &data_in, struct output &out ) {
	out.rc = 1;
	out.time_gmres = 0;
	out.time_x_update = 0;
	out.time_preamble = 0;
	out.time_io = 0;
	out.time_residual = 0;
	out.iterations_gmres = 0;
	out.iterations_arnoldi = 0;
	out.iterations = 0;

	grb::utils::Timer timer;
	timer.reset();

	grb::RC rc = grb::SUCCESS;

	size_t n; // problem matrix size

	Semiring<
		grb::operators::add< ScalarType >, grb::operators::mul< ScalarType >,
		grb::identities::zero, grb::identities::one
	> ring;
	operators::subtract< ScalarType > minus;
	const ScalarType zero = ring.template getZero< ScalarType >();
	const ScalarType one = ring.template getOne< ScalarType >();

	if( data_in.generate_random ) {
		// get size for random matrix
		n = data_in.n;
	} else {
		// get size for input matrix
		grb::utils::MatrixFileReader<
			ScalarType,
			std::conditional<
				( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
				grb::config::RowIndexType,
				grb::config::ColIndexType
			>::type
		> parser( data_in.filename, data_in.direct );
		if( parser.m() != parser.n() ) {
			std::cerr << " matrix in " << data_in.filename << " file, is not rectangular!";
			rc = grb::ILLEGAL;
		}
		n = parser.n();
	}

#ifdef DEBUG
	if( rc == grb::SUCCESS ) {
		std::cout << "Problem size n = " << n << " \n";
	}
#endif

	grb::Matrix< ScalarType > A( n, n );
	grb::Matrix< ScalarType > P( n, n );
	grb::Vector< ScalarType > x( n ), b( n ), temp( n );
	grb::set( temp, zero );

	// initialize Matrix A, P and RHS vector, set x = 0
	if( data_in.generate_random ) {
		rc = rc ? rc : make_matrices( A, P, n, data_in.nz_per_row );
#ifdef DEBUG
		if( rc == grb::SUCCESS ) {
			std::cout << "Random matrices generated successfully\n";
		}
#endif
	} else {
		// read matrix A from file
		grb::utils::MatrixFileReader<
			ScalarType,
			std::conditional<
				( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
			grb::config::RowIndexType,
			grb::config::ColIndexType
			>::type
		> parser( data_in.filename, data_in.direct );
		rc = rc ? rc : buildMatrixUnique(
			A,
			parser.begin( SEQUENTIAL ),
			parser.end( SEQUENTIAL ),
			SEQUENTIAL
		);
#ifdef DEBUG
		if( rc == grb::SUCCESS ) {
			std::cout << "Matrix A built from " << data_in.filename << "file successfully\n";
		}
#endif
		// read preconditioner P from file
		if( !data_in.no_preconditioning ) {
#ifdef DEBUG
			std::cout << "Reading preconditioning matrix from " << data_in.precond_filename << " file \n";
#endif
			grb::utils::MatrixFileReader<
				ScalarType,
				std::conditional<
					( sizeof( grb::config::RowIndexType ) > sizeof( grb::config::ColIndexType ) ),
					grb::config::RowIndexType,
					grb::config::ColIndexType
				>::type
			> parser_precond( data_in.precond_filename, data_in.direct );
			if( parser_precond.m() != parser_precond.n() ) {
				std::cerr << " matrix in " << data_in.precond_filename << " file, is not rectangular!";
				rc = grb::ILLEGAL;
			} else if( parser_precond.n() != n ) {
				std::cerr << " Preconditioner P("<< parser_precond.n() << ") mast have same dimensions as matrix A(" << n << ") !\n";
				rc = grb::ILLEGAL;
			}
			rc = rc ? rc : buildMatrixUnique(
				P,
				parser_precond.begin( SEQUENTIAL ),
				parser_precond.end( SEQUENTIAL ),
				SEQUENTIAL
			);
#ifdef DEBUG
			if( rc == grb::SUCCESS ) {
				std::cout << "Matrix P built from " << data_in.precond_filename << "file successfully\n";
			}
#endif
		}

	}

	out.time_io += timer.time();
	timer.reset();

	// read RHS vector
	if( data_in.rhs ) {
#ifdef DEBUG
		std::cout << "Reading RHS vector from " << data_in.rhs_filename << " file \n";
#endif
		std::vector< ScalarType > buffer( n, zero );
		std::ifstream inFile( data_in.rhs_filename );
		if( inFile.is_open() ) {
			for(size_t i = 0; i < n; ++i) {
				if( !( inFile >> buffer[ i ] ) ){
					std::cerr << "Error reading from: " << data_in.rhs_filename << "\n";
					rc = grb::ILLEGAL;
					break;
				};
			}
			inFile.close(); // cloose input file

			rc = rc ? rc : grb::buildVector( b, buffer.begin(), buffer.end(), SEQUENTIAL );
			if( rc != SUCCESS ) {
				std::cout << "RHS vector: buildVector failed!\n ";
			}
		}
		out.time_io += timer.time();
		timer.reset();

	} else {
		grb::set( x, one );
		grb::set( b, zero );
		rc = rc ? rc : grb::mxv( b, A, x, ring );
		grb::set( x, zero );
		out.time_preamble += timer.time();
		timer.reset();
	}

	// get RHS vector norm
	ScalarType bnorm = zero;
	rc = grb::set( temp, b );
	if( grb::utils::is_complex< ScalarType >::value ) {
		Vector< ScalarType > temp2( n );
		rc = grb::set( temp2, zero );
		rc = rc ? rc : grb::eWiseLambda(
			[ &, temp ] ( const size_t i ) {
				temp2[ i ] = grb::utils::is_complex< ScalarType >::conjugate( temp [ i ] );
			}, temp
		);
		rc = rc ? rc : grb::dot( bnorm, temp, temp2, ring );
	} else {
		rc = rc ? rc : grb::dot( bnorm, temp, temp, ring );
	}
	bnorm =std::sqrt( bnorm );

#ifdef DEBUG
	std::cout << "RHS norm = " << std::abs( bnorm ) << " \n";

	out.pinnedVector = PinnedVector< ScalarType >( b, SEQUENTIAL );
	std::cout << "RHS vector = ";
	for( size_t k = 0; k < 10; ++k ) {
		const ScalarType &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
		std::cout << nonzeroValue << " ";
	}
	std::cout << " ...  ";
	for( size_t k = n - 10; k < n; ++k ) {
		const ScalarType &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
		std::cout << nonzeroValue << " ";
	}
	std::cout << "\n";
#endif


	out.time_preamble += timer.time();
	timer.reset();

	// todo: this loop shoud be moved into gmres() after
	// hessolve is used from ALP

	// inner iterations
	for( size_t i_inner = 0; i_inner < data_in.rep; ++i_inner ) {
		out.residual = std::abs( bnorm );
		out.residual_relative = std::norm( one );
		grb::set( x, zero );

		std::vector< std::vector< ScalarType > > Hmatrix(
			data_in.gmres_restart + 1 ,
			std::vector< ScalarType >( data_in.gmres_restart + 1, zero )
		);
		std::vector< grb::Vector< ScalarType > > Q;
		for( size_t i = 0; i < data_in.gmres_restart + 1; ++i ) {
			Q.push_back(x);
		}
		size_t kspspacesize = 0;

		// gmres iterations
		for( size_t gmres_iter = 0; gmres_iter < data_in.max_iterations; ++gmres_iter ) {
			++out.iterations;
			timer.reset();
			++out.iterations_gmres;
			kspspacesize = 0;
			if( data_in.no_preconditioning ) {
#ifdef DEBUG
				std::cout << "Call gmres without preconditioner.\n";
#endif
				rc = rc ? rc : gmres(
					x, A, b,
					Hmatrix, Q,
					data_in.gmres_restart, TOL,
					kspspacesize,
					temp
				);
			} else {
#ifdef DEBUG
				std::cout << "Call gmres with preconditioner.\n";
#endif
				rc = rc ? rc : gmres(
					x, A, b,
					Hmatrix, Q,
					data_in.gmres_restart, TOL,
					kspspacesize,
					temp,
					P
				);
			}
#ifdef DEBUG
			if( rc == grb::SUCCESS ) {
				std::cout << "gmres iteration finished successfully, kspspacesize = " << kspspacesize << "  \n";
			}
#endif
			out.iterations_arnoldi += kspspacesize;
			out.time_gmres += timer.time();
			timer.reset();

			hessolve( Hmatrix, data_in.gmres_restart + 1, kspspacesize );
			// update x
			for( size_t i = 0; i < kspspacesize; ++i ) {
				rc = rc ? rc : grb::eWiseMul( x, Hmatrix[ 0 ][ i ], Q [ i ], ring );
#ifdef DEBUG
				if( rc != grb::SUCCESS ) {
					std::cout << "grb::eWiseMul( x, Hmatrix[ 0 ][ " << i << " ], Q [ " << i << " ], ring ); failed\n";
				}
#endif
			}

			out.time_x_update += timer.time();
			timer.reset();

#ifdef DEBUG
			if( rc == grb::SUCCESS ) {
				std::cout << "vector x updated successfully\n";
				out.pinnedVector = PinnedVector< ScalarType >( x, SEQUENTIAL );
				std::cout << "x vector = ";
				for( size_t k = 0; k < 10; ++k ) {
					const ScalarType &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
					std::cout << nonzeroValue << " ";
				}
				std::cout << " ...  ";
				for( size_t k = n-10; k < n; ++k ) {
					const ScalarType &nonzeroValue = out.pinnedVector.getNonzeroValue( k );
					std::cout << nonzeroValue << " ";
				}
				std::cout << "\n";
			}
#endif

			// calculate residual
			rc = rc ? rc : grb::set( temp, zero );
			rc = rc ? rc : grb::mxv( temp, A, x, ring );
			rc = rc ? rc : grb::foldl( temp, b, minus );
			ScalarType residualnorm = zero;
			if( grb::utils::is_complex< ScalarType >::value ) {
				Vector< ScalarType > temp2( n );
				rc = grb::set( temp2, zero );
				rc = rc ? rc : grb::eWiseLambda(
					[ &, temp ] ( const size_t i ) {
						temp2[ i ] = grb::utils::is_complex< ScalarType >::conjugate( temp [ i ] );
					}, temp
				);
				rc = rc ? rc : grb::dot( residualnorm, temp, temp2, ring );
			} else {
				rc = rc ? rc : grb::dot( residualnorm, temp, temp, ring );
			}
			if( rc != grb::SUCCESS ) {
				std::cout << "Residual norm not calculated properly.\n";
			}
			residualnorm = std::sqrt( residualnorm );

			out.residual = std::abs( residualnorm );
			out.residual_relative = out.residual / std::abs( bnorm );

#ifdef DEBUG
			std::cout << "Residual norm = " << out.residual << " \n";
			std::cout << "Residual norm (relative) = " << out.residual_relative << " \n";
#endif

			out.time_residual += timer.time();
			timer.reset();

			if( out.residual_relative < data_in.max_residual_norm ) {
#ifdef DEBUG
				std::cout << "Convergence reached\n";
#endif
				break;
			}
		} // gmres iterations

		out.pinnedVector = PinnedVector< ScalarType >( x, SEQUENTIAL );

		timer.reset();

		out.time_io += timer.time();
		timer.reset();

		out.times.io += out.time_io;
		out.times.useful += out.time_x_update + out.time_gmres;
		out.times.postamble += out.time_residual;
		out.times.preamble += out.time_preamble;

		if( i_inner + 1 >   data_in.rep ) {
			std::cout << "Residual norm = " << out.residual << " \n";
			std::cout << "Residual norm (relative) = " << out.residual_relative << " \n";
			std::cout << "X update time = " << out.time_x_update << "\n";
			std::cout << "Residual time = " << out.time_residual << "\n";
			std::cout << "IO time = " << out.time_io << "\n";
			std::cout << "GMRES iterations = " << out.iterations_gmres << "\n";
			std::cout << "Arnoldi iterations = " << out.iterations_arnoldi << "\n";
			std::cout << "GMRES time = " << out.time_gmres << "\n";
			std::cout << "GMRES time per iteration  = " << out.time_gmres / out.iterations_gmres << "\n";
		}

	} // inner iterations

	if( rc == grb::SUCCESS ) {
		out.rc = 0;
	}
}


void printhelp( char *progname ) {
	std::cout << " Use: \n";
	std::cout << "     --n INT                  random generated matrix size, default 0\n";
	std::cout << "                              cannot be used with --matA-fname\n";
	std::cout << "     --nz-per-row INT         numer of nz per row in a random generated matrix, defaiult  10\n";
	std::cout << "                              can only be used when --n is present\n";
	std::cout << "     --test-rep  INT          consecutive test inner algorithm repetitions, default 1\n";
	std::cout << "     --test-outer-rep  INT    consecutive test outer (including IO) algorithm repetitions, default 1\n";
	std::cout << "     --gmres-restart INT      gmres restart (max size of KSP space), default 10\n";
	std::cout << "     --max-gmres-iter INT     maximum number of GMRES iterations, default 1\n";
	std::cout << "     --matA-fname STR         matrix A filename in matrix market format\n";
	std::cout << "                              cannot be used with --n\n";
	std::cout << "     --matP-fname STR         preconditioning matrix P filename in matrix market format\n";
	std::cout << "                              can only be used when --matA-fname is present\n";
	std::cout << "     --rhs-fname  STR         RHS vector filename, where vector elements are stored line-by-line\n";
	std::cout << "     --tol  DBL               convergence tolerance within GMRES, default 1.e-9\n";
	std::cout << "     --max-residual-norm  DBL max residual norm, default 1.e-9\n";
	std::cout << "     --no-preconditioning     disable pre-conditioning\n";
	std::cout << "     --no-direct              disable direct addressing\n";
	std::cout << "Examples\n";
	std::cout << "\n";
	std::cout << "         " << progname << " --n 100 --gmres-restart 50 \n";
	std::cout << "\n";
	std::cout << "         " << progname << " --matA-fname /path/tp/MatA.mtx  --matP-fname /path/to/matP.mtx \n";
}

bool parse_arguments(
	input &in,
	int argc,
	char **argv
) {
	for( int i = 1; i < argc; ++i ){
		if( std::string( argv[ i ] ) == std::string( "--n" ) ) {
			if( in.filename != std::string( "" ) ){
				std::cerr << " input matrix fname already given, cannot use --matA-fname with --n flag\n";
				return false;
			}
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.n ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: n = " << in.n << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--nz-per-row" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.nz_per_row ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: nz_per_row = " << in.nz_per_row << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--test-rep" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.rep ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: rep = " << in.rep << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--test-outer-rep" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.rep_outer ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: rep_outer = " << in.rep_outer << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--gmres-restart" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.gmres_restart ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
		} else if( std::string( argv[ i ] ) == std::string( "--max-gmres-iter" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.max_iterations ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: max_iterations = " << in.max_iterations << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--matA-fname" ) ) {
			if( in.n != 0 ){
				std::cerr << "randomly generated matrix already requested, cannot use --matA-fname with --n flag\n";
				return false;
			}
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.filename ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
			in.generate_random = false;
#ifdef DEBUG
			std::cout << " set: filename = " << in.filename << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--matP-fname" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.precond_filename ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: precond_filename = " << in.precond_filename << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--rhs-fname" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.rhs_filename ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
			in.rhs = true;
#ifdef DEBUG
			std::cout << " set: rhs_filename = " << in.rhs_filename << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--tol" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.tol ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: tol = " << in.tol << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--max-residual-norm" ) ) {
			std::stringstream s( argv[ ++i ] );
			if( !( s >> in.max_residual_norm ) ){
				std::cerr << "error parsing: " << argv[ i ] << "\n";
				return false;
			}
#ifdef DEBUG
			std::cout << " set: max_residual_norm = " << in.max_residual_norm << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--no-preconditioning" ) ) {
			in.no_preconditioning = true;
#ifdef DEBUG
			std::cout << " set: no_preconditioning = " << in.no_preconditioning << "\n";
#endif
		} else if( std::string( argv[ i ] ) == std::string( "--no-direct" ) ) {
			in.direct = false;
#ifdef DEBUG
			std::cout << " set: direct = " << in.direct << "\n";
#endif
		} else {
			std::cerr << "unknown command line argument: " << argv[ i ] << "\n";
			return false;
		}
	}

	if( in.precond_filename == std::string( "" ) && in.filename != std::string( "" ) ) {
		in.no_preconditioning = true;
	}
	if( in.precond_filename != std::string( "" ) && in.filename == std::string( "" ) ) {
		std::cerr << " --matP-fname can be used only if --matA-fname is present";
		return false;
	}

	if( in.n == 0 && in.filename == std::string( "" ) ) {
		std::cerr << "No input!\n";
		return false;
	}

	return true;
}

int main( int argc, char **argv ) {

	grb::RC rc = grb::SUCCESS;
	std::cout << "Test executable: " << argv[ 0 ] << std::endl;

	input  in;
	output out;

	if( !parse_arguments( in, argc, argv ) ) {
		std::cerr << "error parsing command line arguments\n";
		printhelp( argv[0] );
		return 1;
	};

	std::cout << "Executable called with parameters " << in.filename << ", "
		  << "inner repititions = " << in.rep << ", "
		  << "and outer reptitions = " << in.rep_outer << std::endl;

	// launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code "
				<< (int)rc << std::endl;
			return 6;
		}
	}

	// launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, in.rep_outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code "
			<< grb::toString( rc ) << std::endl;
		return 8;
	} else if( out.rc == 0 ) {
		std::cout << "Benchmark completed successfully and took "
			<< out.iterations << " iterations to converge "
			<< "with residual " << out.residual << ".\n";
	}

	const size_t n = out.pinnedVector.size();
	std::cout << "Error code is " << out.rc << ".\n";
	std::cout << "Size of pr is " << n << ".\n";
	if( out.rc == 0 && n > 0 ) {
		std::cout << "First 10 nonzeroes of pr are: ( ";
		for( size_t k = 0; k < out.pinnedVector.nonzeroes() && k < 10; ++k ) {
			const auto &value = out.pinnedVector.getNonzeroValue( k );
			std::cout << value << " ";
		}
		std::cout << ")" << std::endl;
	}

	if( out.rc != 0 ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED\n";
	} else {
		std::cout << "Test OK\n";
	}
	std::cout << std::endl;

	// done
	return out.rc;
}



