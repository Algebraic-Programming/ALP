
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
		size_t gmres_restart = 10;
};

struct output {
	int rc;
	size_t rep;
	size_t iterations;
	size_t iterations_arnoldi;
	size_t iterations_gmres;
	double residual;
	grb::utils::TimerResults times;
	double time_gmres;
	double time_preamble;
	double time_io;
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

BaseScalarType sqrt_generic( BaseScalarType x ) {
	//return std::sqrt( x );
	return( pow( x, 0.5 ) );

	// //test Quake sqrt
	// double y = x;
	// double x2 = y * 0.5;
	// size_t * ip = reinterpret_cast< size_t * >( &y );
	// *ip = 0x5fe6eb50c7b537a9 - ( *ip / 2 );
	// y = y * ( 1.5 - x2 * y * y  );   // 1st iteration
	// y = y * ( 1.5 - x2 * y * y  );   // 2nd iteration
	// y = y * ( 1.5 - x2 * y * y  );   // 3nd iteration
	// y = y * ( 1.5 - x2 * y * y  );   // 4nd iteration
	// return 1./y;
}

/**
 * Generate random linear problem matrix data (A) and
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
			(void) ++j
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
			(void) ++acout;
			if( i == j ) {
				MatPveci[ pcout ] = i;
				MatPvecj[ pcout ] = j;
				grb::foldr( one, tmp2, divide );
				MatPvecv[ pcout ] = tmp2;
				(void) ++pcout;
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

void grbProgram( const struct input &data_in, struct output &out ) {
	out.rc = 1;
	out.time_gmres = 0;
	out.time_preamble = 0;
	out.time_io = 0;
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
	operators::divide< ScalarType > divide;
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

	// inner iterations
	for( size_t i_inner = 0; i_inner < data_in.rep; ++i_inner ) {

		grb::set( temp, zero );
		std::vector< ScalarType > Hmatrix(
			( data_in.gmres_restart + 1 ) * ( data_in.gmres_restart + 1 ),
			zero
		);
		std::vector< ScalarType > temp3( n, 0 );

		std::vector< grb::Vector< ScalarType > > Q;
		for( size_t i = 0; i < data_in.gmres_restart + 1; ++i ) {
			Q.push_back(x);
		}
		grb::set( x, zero );

		out.time_preamble += timer.time();
		timer.reset();

		const std::function< BaseScalarType( BaseScalarType ) > &my_sqrt = sqrt_generic;
		if( data_in.no_preconditioning ) {
			rc = rc ? rc : grb::algorithms::gmres(
				x, A, b,
				data_in.gmres_restart, data_in.max_iterations,
				data_in.tol,
				out.iterations, out.iterations_gmres, out.iterations_arnoldi,
				out.residual,
				Q, Hmatrix,
				temp, temp3,
				ring, minus, divide, my_sqrt
			);
		} else {
			rc = rc ? rc : grb::algorithms::preconditioned_gmres(
				x, P, A, b,
				data_in.gmres_restart, data_in.max_iterations,
				data_in.tol,
				out.iterations, out.iterations_gmres, out.iterations_arnoldi,
				out.residual,
				Q, Hmatrix,
				temp, temp3,
				ring, minus, divide, my_sqrt
			);
		}

		out.time_gmres += timer.time();
		timer.reset();

		if( i_inner + 1 > data_in.rep ) {
			std::cout << "Residual norm = " << out.residual << " \n";
			std::cout << "IO time = " << out.time_io << "\n";
			std::cout << "GMRES iterations = " << out.iterations_gmres << "\n";
			std::cout << "Arnoldi iterations = " << out.iterations_arnoldi << "\n";
			std::cout << "GMRES time = " << out.time_gmres << "\n";
			std::cout << "GMRES time per iteration  = " << out.time_gmres / out.iterations_gmres << "\n";
		}

	} // inner iterations

	out.times.postamble += timer.time();
	out.times.useful += out.time_gmres;
	out.times.io += out.time_io;
	out.times.preamble += out.time_preamble;

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

