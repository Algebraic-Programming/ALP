/*
  Minimal scaffold adapted from ising_machine_sb.cpp to drive a replica-exchange
  simulated-annealing (RE-SA) solver.  Algorithmic parts are intentionally left
  unimplemented (stubs).  This file mirrors the existing IO / launcher /
  program structure and replaces numpy arrays with grb::Vector and lists of
  numpy vectors with std::vector< grb::Vector<...> >. Sparse matrices are
  represented as grb::Matrix< JType >.

  Purpose: allow running internal tests or an external-run mode while the RE-SA
  algorithm is implemented separately.
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>
#include <memory>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstdlib>
#include <unistd.h>

#include <graphblas/algorithms/simulated_annealing_re.hpp>
#include <graphblas/nonzeroStorage.hpp>
#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>
#include <graphblas/utils/iterators/nonzeroIterator.hpp>
#include <utils/output_verification.hpp>
#include <graphblas.hpp>
#include <utils/print_vec_mat.hpp>

using namespace grb;

#define DEBUG_IMSB 1
#define ISCLOSE(a,b) (std::abs((b)-(a))/std::abs(a) < 1e-4) || (std::abs((b)-(a)) < 1e-4)


// Types
using IOType = double;   // scalar/vector element type
using JType  = double;   // coupling (matrix) value type
using EnergyType  = double;   // coupling (matrix) value type

/** Parser type */
typedef grb::utils::MatrixFileReader<
	JType,
	std::conditional<
		(sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
		grb::config::RowIndexType,
		grb::config::ColIndexType
	>::type
> Parser;

/** Nonzero type */
typedef internal::NonzeroStorage<
	grb::config::RowIndexType,
	grb::config::ColIndexType,
	JType
> NonzeroT;

/** In-memory storage type using tuple */
typedef grb::utils::Singleton<
    std::tuple<
        size_t,                    // n (rows/columns)
        size_t,                    // nz (nonzeros)
        size_t,                    // nsweeps
        size_t,                    // n_replicas
        bool,                      // use_pt
        unsigned,                  // seed
        std::string,               // sweep_name
        std::vector<NonzeroT>,     // matrix data
        std::vector<JType>         // h vector
    >
> Storage;

namespace test_data {
    constexpr size_t n = 16;
    constexpr size_t nsweeps = 2;
    constexpr size_t n_replicas = 3;
    constexpr bool use_pt = true; 
    constexpr unsigned seed = 8;

    const std::vector< std::pair< std::pair< grb::config::RowIndexType, grb::config::ColIndexType >, JType > > j_matrix_data = {
		{{0, 1}, -0.2752300610319546},
		{{1, 0}, -0.2752300610319546},
		{{1, 2}, -0.10636508505639508},
		{{2, 1}, -0.10636508505639508},
		{{2, 3}, 0.3961450048806352},
		{{3, 2}, 0.3961450048806352},
		{{3, 4}, -0.15453838800213293},
		{{3, 5}, 0.4847494372852713},
		{{4, 3}, -0.15453838800213293},
		{{4, 5}, -0.4712679510367046},
		{{5, 3}, 0.4847494372852713},
		{{5, 4}, -0.4712679510367046},
		{{5, 6}, -0.1483152637298799},
		{{6, 5}, -0.1483152637298799},
		{{7, 8}, -0.11904111079614699},
		{{8, 7}, -0.11904111079614699},
		{{9, 10}, -0.18031020353297234},
		{{10, 9}, -0.18031020353297234},
		{{10, 11}, -0.22985425840853468},
		{{11, 10}, -0.22985425840853468},
		{{11, 12}, 0.30105588632639446},
		{{11, 13}, 0.13823880612312134},
		{{12, 11}, 0.30105588632639446},
		{{13, 11}, 0.13823880612312134},
		{{13, 14}, 0.10364447636911123},
		{{14, 13}, 0.10364447636911123},
		{{14, 15}, 0.2955745584289766},
		{{15, 14}, 0.2955745584289766},
    };


    const size_t nnz = j_matrix_data.size();

    const std::vector< JType > h_array_data = {
        -0.08910436,  0.58034508,  0.97719304,  0.16792909,
		-0.9221754 , -0.10715418, -0.62365497,  0.25411129,
		-0.5693644 , -0.69805978,  0.07228861, -0.79922641,
		0.46231686 , 0.87930208 ,  0.88663637, -0.25052299,
    };

	const std::vector< std::vector< size_t > > row_blocks = {
		// {3, 1, 6, 7, 9, 11, 12, 13, 14, 15}, {5, 2, 0, 8, 10}, {4} // for python data files
		{0, 2, 4, 7, 9, 12, 13, 15}, {1, 3, 6, 8, 11}, {5, 10, 14},
		// {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}
	};

}
// --- New, minimal runner configuration and result types ---
struct input {
    bool use_default_data = false;
    std::string filename_Jmatrix;
    std::string filename_h;
    size_t n_replicas = test_data::n_replicas;
    size_t nsweeps = test_data::nsweeps;
    bool use_pt = test_data::use_pt;
    unsigned seed = test_data::seed;
    std::string sweep_name = "sequential_sweep_immediate";
    bool verify = false;
    std::string filename_ref_solution;
	bool direct;
    size_t rep = 0;
    size_t outer = 1;
};

struct output {
    int error_code = 0;
    // TODO: remove itrations if not applicable
    size_t iterations = 10; // total number of iterations performed does not make sense since the code does not have convergence criteria
    EnergyType best_energy = std::numeric_limits< EnergyType >::max();
	size_t rep;
	grb::utils::TimerResults times;
    std::unique_ptr< PinnedVector< JType > > pinnedSolutionVector;
    std::unique_ptr< PinnedVector< JType > > pinnedRefSolutionVector;
    // other things like eg: best replicas ...
};

template< typename Dtype >
void read_matrix_data(const std::string &filename, std::vector<Dtype> &data, bool direct) {
    // Implementation for reading matrix data from file
	try {
		Parser parser( filename, direct );
		assert( parser.m() == parser.n() );
		std::get<0>(Storage::getData()) = parser.n();
		try {
			std::get<1>(Storage::getData()) = parser.nz();
		} catch( ... ) {
			std::get<1>(Storage::getData()) = parser.entries();
		}
		/* Once internal issue #342 is resolved this can be re-enabled
		for(
			auto it = parser.begin( PARALLEL );
			it != parser.end( PARALLEL );
			++it
		) {
			data.push_back( *it );
		}*/
		for(
			auto it = parser.begin( SEQUENTIAL );
			it != parser.end( SEQUENTIAL );
			++it
		) {
			data.push_back( Dtype( *it ) );
#ifdef DEBUG_IMSB
			if( spmd<>::pid() == 0 ){
				// print last data element from std::vector<NonzeroT> data
				std::cout << "readmatrix_data: " << data.back().first.first << ", "
					<< data.back().first.second << ", " << data.back().second << "\n";
			}
#endif
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}
}

template< typename NonzeroT, typename IType, typename VType >
void read_matrix_data_from_array(
	const std::vector<std::pair< std::pair< IType, IType >, VType > > &array,
	std::vector<NonzeroT> &data
) {
	// Implementation for reading matrix data from array
    try {
        for (const auto &entry : array) {
            data.emplace_back(
                NonzeroT( entry.first.first, entry.first.second, entry.second )
            );
#ifdef DEBUG_IMSB
			if( spmd<>::pid() < 2 ){
				// print last data element from std::vector<NonzeroT> data
				std::cout << "read_matrix_data_from_array: " << data.back().first.first << ", "
					<< data.back().first.second << ", " << data.back().second << "\n";
			}
#endif
        }
        std::get<0>(Storage::getData()) = test_data::n;
        std::get<1>(Storage::getData()) = data.size();
    } catch (const std::exception &e) {
        std::cerr << "Failed to read matrix data from array: " << e.what() << "\n";
        return;
    }
}

template< typename Dtype >
void read_vector_data(const std::string &filename, std::vector<Dtype> &data) {
    // Implementation for reading vector data from file
    try {
        std::ifstream file( filename );
        if( !file.is_open() ) {
            std::cerr << "Failed to open vector file: " << filename << "\n";
            return;
        }
        std::string line;
        while( std::getline( file, line ) ) {
            if( line.empty() ) continue; // skip empty lines
            std::istringstream iss( line );
            Dtype v;
            if( !(iss >> v) ) {
                throw std::runtime_error( "Failed to parse line in vector file" );
            }
            data.push_back( v );
        }
    } catch( std::exception &e ) {
        std::cerr << "I/O program failed: " << e.what() << "\n";
        return;
    }
}

template< typename Dtype >
void read_vector_data_from_array(
	const std::vector<Dtype> &array, std::vector<Dtype> &data
) {
	// Implementation for reading vector data from array
	try {
		for (size_t i = 0; i < array.size(); ++i) {
			data.push_back(array[i]);
		}
	} catch (const std::exception &e) {
		std::cerr << "Failed to read vector data from array: " << e.what() << "\n";
		return;
	}
}

template<
		class Ring = Semiring<
			grb::operators::add< JType >, grb::operators::mul< JType >,
			grb::identities::zero, grb::identities::one
		> >
EnergyType get_energy(
				 const grb::Matrix< JType >& couplings,
				 const grb::Vector< JType > &local_fields,
				 const grb::Vector< IOType > &state,
				 const Ring &ring = Ring()
			  ){
	static grb::Vector< JType > tmp ( grb::size( local_fields ) );
	grb::RC rc = grb::clear( tmp );
	EnergyType energy = 0.0;

	rc = rc ? rc : grb::mxv( tmp, couplings, state, ring );
	rc = rc ? rc : grb::foldl( tmp, static_cast< JType >( 0.5 ), ring.getMultiplicativeMonoid() );
	rc = rc ? rc : grb::foldl( tmp, local_fields, ring.getAdditiveMonoid() );
	rc = rc ? rc : grb::dot<>( energy, tmp, state, ring );
	assert( rc == grb::SUCCESS );

	return energy;
}

template<
		typename SweepDataType = std::tuple<
				 	 const grb::Matrix< JType >&,
				 	 const grb::Vector< JType >&,
					 grb::Vector< JType >&,
					 grb::Vector< JType >&,
					 grb::Vector< IOType >&,
					 const std::vector< grb::Vector< bool > >&,
					 grb::Vector< EnergyType >&,
					 grb::Vector< bool >&
					 >,
		grb::Descriptor descr = grb::descriptors::no_operation,
		class Ring = Semiring<
			grb::operators::add< JType >, grb::operators::mul< JType >,
			grb::identities::zero, grb::identities::one
		>
	>
static EnergyType sequential_sweep_immediate(
				 grb::Vector< IOType > &state,
				 const JType &beta,
				 std::tuple<
				 	 const grb::Matrix< JType > &,
				 	 const grb::Vector< JType > &,
					 grb::Vector< JType >&,
					 grb::Vector< JType >&,
					 grb::Vector< IOType >&,
					 const std::vector< grb::Vector< bool > >&,
					 grb::Vector< EnergyType >&,
					 grb::Vector< bool >&,
					 std::minstd_rand&
					 > &data
			  ){
		const Ring ring = Ring();


		grb::RC rc = grb::SUCCESS;
		const size_t n = grb::size( state );
		EnergyType delta_energy = static_cast< JType >(0.0);

		const auto &couplings 	= std::get<0>(data);
		const auto &local_fields = std::get<1>(data);
		auto &h 		= std::get<2>(data);
		auto &log_rand	= std::get<3>(data);
		auto &delta		= std::get<4>(data);
		const auto &masks = std::get<5>(data);
		auto &dn		= std::get<6>(data);
		auto &accept	= std::get<7>(data);
		auto &rng       = std::get<8>(data);

		rc = rc ? rc : grb::wait();
		rc = rc ? rc : grb::resize( h, n );
		rc = rc ? rc : grb::resize( log_rand, n );
		rc = rc ? rc : grb::resize( delta, n );
		rc = rc ? rc : grb::resize( dn, n );
		rc = rc ? rc : grb::resize( accept, n );

		rc = rc ? rc : grb::set( h, local_fields );
		rc = rc ? rc : grb::mxv( h, couplings, state , ring );

		std::uniform_real_distribution< JType > rand ( 0.0, 1.0 );
		for( size_t j = 0 ; j < n ; ++j ){
			const auto rnd = rand( rng );
			rc = rc ? rc : grb::setElement(log_rand,  std::log( rnd ), j );
		}
		// rc = rc ? rc : grb::wait();
		// print_vector( log_rand, 30, "log_rand" );

#ifndef NDEBUG
		const grb::Vector< IOType > old_state = state;
#endif
		for(const auto &mask : masks ){

			rc = rc ? rc : grb::clear( accept  );
			rc = rc ? rc : grb::clear( delta  );
			rc = rc ? rc : grb::clear( dn );

			// dn = (2*state_slice - 1) * h_slice
			rc = rc ? rc : grb::set( dn, mask, state );
			rc = rc ? rc : grb::foldl( dn, static_cast< EnergyType >( 2 ), ring.getMultiplicativeMonoid()  );
			rc = rc ? rc : grb::foldl( dn, static_cast< EnergyType >( -1 ), ring.getAdditiveMonoid() );
			rc = rc ? rc : grb::foldl( dn, h, ring.getMultiplicativeMonoid() );

			// ( dn >= 0 ) | ( log_rand < beta * dn )
			rc = rc ? rc : grb::set( accept, mask );
			rc = rc ? rc : grb::wait(); // ERROR: Segmentation Fault with nonblocking backend
			rc = rc ? rc : grb::eWiseLambda<>(
					[ &mask, &accept, &dn, &log_rand, beta ]( const size_t i ){
						(void) i;
						if( mask[i] ){
							accept[i] = ( dn[i] >= 0 ) || ( log_rand[i] < beta * dn[i] );
						}
					}, mask, log_rand, dn, accept );
			// print_vector( log_rand, 30, "log_rand" );
			// print_vector( mask, 30, "mask" );
			// print_vector( accept, 30, "accept" );

			// new_state = np.where(accept, 1 - old, old)
			rc = rc ? rc : grb::foldl( state, accept, static_cast< IOType >( -1 ), ring.getMultiplicativeMonoid() );
			rc = rc ? rc : grb::foldl( state, accept, static_cast< IOType >( 1 ), ring.getAdditiveMonoid() );
			// print_vector( state, 30, "state" );
			
			// delta = new - old ==> delta[accept] = 2*new_state[accept]-1
			rc = rc ? rc : grb::clear( delta  );
			rc = rc ? rc : grb::set( delta, accept, state );
			rc = rc ? rc : grb::foldl( delta, accept, static_cast< IOType >( 2 ), ring.getMultiplicativeMonoid() );
			rc = rc ? rc : grb::foldl( delta, accept, static_cast< IOType >( -1 ), ring.getAdditiveMonoid() );
			
			// Update delta_energy -= dot(dn, accept)
			rc = rc ? rc : grb::dot< descr >( delta_energy, delta, h, ring );
			// rc = rc ? rc : grb::wait();

			// update h
			rc = rc ? rc : grb::mxv( h, couplings, delta, ring );
			
		}
		rc = rc ? rc : grb::wait();

#ifndef NDEBUG
		if( rc != grb::SUCCESS ){
			std::cerr << "\n\t Error in some GraphBLAS function " << rc << " : " << grb::toString( rc ) << std::endl;
			abort();
		}
		assert( rc == grb::SUCCESS );
		const auto new_state = state;
		rc = rc ? rc : grb::wait();

		const auto real_delta = get_energy(couplings, local_fields, new_state) - get_energy(couplings, local_fields, old_state);
		std::cerr << "\n\t Delta_energy: " << delta_energy;
		std::cerr << "\n\t Real delta: " << real_delta;
		std::cerr << "\n\t Discrepancy: " << real_delta - delta_energy;
		// std::cerr << "\n\t Old energy: " << get_energy(couplings, local_fields, old_state) ;
		// std::cerr << "\n\t New energy: " << get_energy(couplings, local_fields, new_state);
		std::cerr << std::endl;

		assert( ISCLOSE(real_delta, delta_energy ) );
		// TODO: assert fails with nonblocking backend -> see issue #397
#endif

		return delta_energy;
}


template<
		typename SweepDataType = std::tuple<
				 	 const grb::Matrix< JType >&,
				 	 const grb::Vector< JType >&,
					 grb::Vector< JType >&,
					 grb::Vector< JType >&,
					 grb::Vector< IOType >&,
					 const std::vector< grb::Vector< bool > >&,
					 grb::Vector< EnergyType >&,
					 grb::Vector< bool >&,
					 std::minstd_rand&
					 >,
		typename SweepFuncType = std::function< EnergyType(
					 grb::Vector< IOType >&,
					 const JType&,
					 SweepDataType&
				 ) >,
		class Ring = Semiring<
			grb::operators::add< JType >, grb::operators::mul< JType >,
			grb::identities::zero, grb::identities::one
		>
	>
SweepFuncType get_sweep_function( std::string sweep_name ){
	if( sweep_name != "sequential_sweep_immediate" ){
			std::cerr << "Warning: unknown sweep setting. Falling back to  \"sequential_sweep_immediate\"" << std::endl;
	}
	 return sequential_sweep_immediate< Ring >;
}

void ioProgram( const struct input &data_in, bool &success ) {

    using namespace test_data;
	success = false;

	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	try {
		// Parse and store matrix in singleton class
		// Map Storage tuple fields to meaningful names and wire up default data
		auto &storage = Storage::getData();
		// auto &n           = std::get<0>(storage); // n (rows/cols)
		// auto &nnz         = std::get<1>(storage); // nz (nonzeros)
		auto &nsweeps_st  = std::get<2>(storage); // nsweeps
		auto &n_replicas_st = std::get<3>(storage); // n_replicas
		auto &use_pt      = std::get<4>(storage); // use_pt
		auto &seed_st     = std::get<5>(storage); // seed
		auto &sweep_name  = std::get<6>(storage); // sweep_name
		auto &Jdata       = std::get<7>(storage); // std::vector<NonzeroT>
		auto &h           = std::get<8>(storage); // std::vector<JType>

		// Initialize metadata from input (allow CLI to override defaults)
		nsweeps_st    = data_in.nsweeps;
		n_replicas_st = data_in.n_replicas;
		use_pt        = data_in.use_pt;
		seed_st       = data_in.seed;
		sweep_name    = data_in.sweep_name; // TODO: makes bsp1d backend crash!?


		if ( data_in.use_default_data ) {
			// if no file provided, use default data from file_content
			read_matrix_data_from_array<NonzeroT>( test_data::j_matrix_data, Jdata );
			read_vector_data_from_array<JType>( test_data::h_array_data, h );
			// other data
		} else {
			// read from files if provided
			read_matrix_data<NonzeroT>( data_in.filename_Jmatrix, Jdata, data_in.direct );
			read_vector_data<JType>( data_in.filename_h, h );
			if(data_in.verify) {
				if(data_in.filename_ref_solution.empty()) {
					std::cerr << "Reference solution file not provided for verification\n";
					return;
				}
			}
		}
	} catch( std::exception &e ) {
		std::cerr << "I/O program failed: " << e.what() << "\n";
		return;
	}

	success = true;
}


void grbProgram(
    const struct input &data_in, 
    struct output &out
) {
    std::cout<< "grbProgram: running simulated-annealing RE solver (stub)\n";

	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// std::cerr << "Process " << s <<  " running at line " << __LINE__ << std::endl;

    grb::utils::Timer timer;
	timer.reset();

    /* --- Problem setup --- */
    const size_t n = std::get<0>(Storage::getData());
	if( s == 0 ){
		std::cout << "problem size n = " << n << "\n";
	}
    grb::Vector< JType > h( n );

    // populate J with test (random) values
    grb::RC rc = grb::SUCCESS;

    // load into GraphBLAS
    grb::Matrix< JType > J( n, n );
	{
		const auto &data = std::get<7>(Storage::getData());
		RC io_rc = buildMatrixUnique(
			J,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cend() ),
			SEQUENTIAL
		);
		/* Once internal issue #342 is resolved this can be re-enabled
		RC io_rc = buildMatrixUnique(
			J,
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cbegin() ),
			utils::makeNonzeroIterator<
				grb::config::RowIndexType, grb::config::ColIndexType, JType
			>( data.cend() ),
			PARALLEL
		);*/
		io_rc = io_rc ? io_rc : wait();
		if( io_rc != SUCCESS ) {
			std::cerr << "Failure: call to buildMatrixUnique did not succeed "
				<< "(" << toString( io_rc ) << ")." << std::endl;
			out.error_code = 5;
			return;
		}

#ifdef DEBUG_IMSB
		if( s == 0 && grb::ncols( J ) < 40 ) {
			std::cout << "Matrix J:\n";
			print_matrix( J );
		}
#endif
	}

    // build vector h with data from singleton
    {
        const auto &h_data = std::get<8>(Storage::getData());
		rc = rc ? rc : buildVector(
			h,
			h_data.cbegin(),
			h_data.cend(),
			SEQUENTIAL
		);
    }

	assert( grb::nnz( grb::Vector< bool >( n ) ) == 0 );

	// build masks from row block indices
    std::vector< grb::Vector< bool > > masks;
	for(const auto&v : test_data::row_blocks ){
		masks.emplace_back( grb::Vector< bool >( n ) );
		for(const auto&i : v ){
			grb::setElement( masks.back(), 1, i );
		}
		if( s == 0 ){
			print_vector( masks.back(), 30, "MASK" );
		}
	}

    // seed RNGs (C and C++ engines) using requested seed (hardcoded default 8 if not provided)
    std::srand( static_cast<unsigned>( data_in.seed + s ) );
    std::minstd_rand rng ( data_in.seed + s ); // rng or std::mt19937

    // create states storage and initialize with random 1/0 values
    const size_t n_replicas = std::get<3>(Storage::getData());
    std::vector< grb::Vector<IOType> > states;
    for ( size_t r = 0; r < n_replicas; ++r ) {
        states.emplace_back( grb::Vector<IOType>(n) );
        // initialize with random values
        std::uniform_int_distribution< unsigned short > randint(0,1);
        // we use buildvectorUnique with a random set of indices
        std::vector< IOType > rand_data;
        for ( size_t i = 0; i < n; ++i ) {
            rand_data.emplace_back( static_cast<IOType>(
                randint( rng ) ) );
        }
        rc = rc ? rc : grb::buildVector(
            states.back(),
            rand_data.cbegin(),
            rand_data.cend(),
            SEQUENTIAL
        );
    }
	
	const auto sweep = get_sweep_function( data_in.sweep_name );


    #ifdef DEBUG_IMSB
    if( s == 0 ) {
        for ( size_t r = 0; r < n_replicas; ++r ) {
            std::cout << "Initial state replica " << r << ":\n";
            print_vector( states[r], 30 ,"states values" );  
			std::cout << "With energy " << get_energy(  J, h, states[r] ) << "\n";
            std::cout << std::endl;
        }

		// assert( std::abs(get_energy(  J, h, zero ) - 0.5803450826765713) < 1e-4 );
    }
    #endif


    // also make betas vector os size n_replicas and initialize with 10.0
    grb::Vector< JType > betas( n_replicas );
    grb::Vector< EnergyType > energies( n_replicas );
    grb::Vector< EnergyType > temp_energies( n_replicas );
    for ( size_t r = 0; rc == grb::SUCCESS && r < n_replicas; ++r ) {
        rc = rc ? rc : grb::setElement( betas, static_cast< JType >(10.0), r );
        rc = rc ? rc : grb::setElement( energies, get_energy(  J, h, states[r] ), r );
    }
    rc = rc ? rc : wait();


    std::vector< grb::Vector<IOType> > temp_states;
	grb::Vector< JType > temp_h ( n );
	grb::Vector< JType > temp_log_rand ( n );
	grb::Vector< EnergyType > temp_dn ( n );
	grb::Vector< bool > temp_accept ( n );
	grb::Vector< IOType > temp_delta ( n );
	auto sweep_data = std::tie(
			(const typeof(J)&) J,
			(const typeof(h)&) h,
 			temp_h,
			temp_log_rand,
			temp_delta,
			(const typeof(masks)&) masks,
			temp_dn,
			temp_accept,
			rng
			);
	grb::wait();


	out.rep = data_in.rep;
	// time a single call
	if( out.rep == 0 ) {
		timer.reset();
		rc = grb::algorithms::simulated_annealing_RE(
				sweep, sweep_data, states, energies, betas, temp_states, temp_energies, data_in.nsweeps, data_in.use_pt
        );

		rc = rc ? rc : wait();
		double single_time = timer.time();
		if( !(rc == SUCCESS || rc == FAILED) ) {
			std::cerr << "Failure: call to Simulated Annealing RE did not succeed ("
				<< toString( rc ) << ")." << std::endl;
			out.error_code = 20;
		}
		if( rc == FAILED ) {
			std::cout << "Warning: call to Simulated Annealing RE did not converge\n";
		}
		if( rc == SUCCESS ) {
			rc = collectives<>::reduce( single_time, 0, operators::max< double >() );

			for(size_t i = 0 ; i < n_replicas ; ++i ){
				out.best_energy = std::min( out.best_energy, energies[ i ] );
			}
		}
		if( rc != SUCCESS ) {
			out.error_code = 25;
		}
		out.times.useful = single_time;
		out.rep = static_cast< size_t >( 1000.0 / single_time ) + 1;
		if( rc == SUCCESS || rc == FAILED ) {
			if( s == 0 ) {
				if( rc == FAILED ) {
					std::cout << "Info: cold Simulated Annealing RE did not converge within ";
				} else {
					std::cout << "Info: cold Simulated Annealing RE completed within ";
				}
				std::cout << out.iterations << " iterations. "
					<< "Time taken was " << single_time << " ms. "
					<< "Deduced inner repetitions parameter of " << out.rep << " "
					<< "to take 1 second or more per inner benchmark.\n";
			}
		}
	} else {
		// do benchmark
		timer.reset();
		for( size_t i = 0; i < out.rep && rc == SUCCESS; ++i ) {
			if( rc == SUCCESS ) {
				out.iterations = data_in.nsweeps;

                rc = grb::algorithms::simulated_annealing_RE(
				sweep, sweep_data, states, energies, betas, temp_states, temp_energies, data_in.nsweeps, data_in.use_pt
                );
			}
			if( grb::Properties<>::isNonblockingExecution ) {
				rc = rc ? rc : wait();
			}
		}
		const double time_taken = timer.time();
		if( s == 0 ) {
			for ( size_t r = 0; r < n_replicas; ++r ) {
				std::cout << "Final state replica " << r << ":\n";
				print_vector( states[r], 50 ,"states values" );  
				std::cout << "With energy " << energies[ r ] << "\n";
				std::cout << "With energy " << get_energy(  J, h, states[r] ) << "\n";
				std::cout << std::endl;
				assert( ISCLOSE( get_energy( J, h, states[r] ), energies[ r ] ) );
			}
		}
		for(size_t i = 0 ; i < n_replicas ; ++i ){
			out.best_energy = std::min( out.best_energy, energies[ i ] );
		}

		out.times.useful = time_taken / static_cast< double >( out.rep );
		// print timing at root process
		if( s == 0 ) {
			std::cout << "Time taken for " << out.rep << " "
				<< "Simulated Annealing RE calls (hot start): " << out.times.useful << ". "
				<< "Error code is " << grb::toString( rc ) << std::endl;
			std::cout << "\tnumber of IM-SB iterations: " << out.iterations << "\n";
			std::cout << "\tmilliseconds per iteration: "
				<< ( out.times.useful / static_cast< double >( out.iterations ) )
				<< "\n";
		}
		sleep( 1 );
	}

	// start postamble
	timer.reset();

	// set error code
	if( rc == FAILED ) {
		out.error_code = 30;
	} else if( rc != SUCCESS ) {
		std::cerr << "Benchmark run returned error: " << toString( rc ) << "\n";
		out.error_code = 35;
		return;
	}
}


// --- Simple help / CLI parser for the new runner (no backward compatibility) ---
void printhelp( char *progname ) {
    std::cout << "Usage: " << progname << " [--use-default-data] [--j-matrix-fname STR] [--h-fname STR]\n"
              << "       [--n-replicas INT] [--nsweeps INT] [--seed INT] [--sweep STR]\n"
              << "       [--verify] [--ref-solution-fname STR] [--help]\n\n"
              << "Options:\n"
              << "  --use-default-data         Use embedded default test data\n"
              << "  --j-matrix-fname STR       Path to J matrix file (matrix-market or supported)\n"
              << "  --h-fname STR              Path to h (local fields) vector (whitespace separated)\n"
              << "  --n-replicas INT           Number of replicas (default: 3)\n"
              << "  --nsweeps INT              Number of sweeps (default: 2)\n"
              << "  --use-pt BOOL              Use Parallel Tampering (default: 1)\n"
              << "  --seed INT                 RNG seed (default: 8)\n"
              << "  --sweep STR                Sweep selector (default: sequential_sweep_immediate)\n"
              << "  --verify                   Verify output against reference solution\n"
              << "  --ref-solution-fname STR   Reference solution file (required with --verify unless using default data)\n"
              << "  --help, -h                 Print this help message\n";
}

bool parse_arguments( input &in, int argc, char ** argv ) {
    in.filename_Jmatrix.clear();
    in.filename_h.clear();
    in.filename_ref_solution.clear();
    in.direct = true;
    // map benchmarking configuration to the runner's fields
    in.rep = grb::config::BENCHMARKING::inner();
    in.outer = grb::config::BENCHMARKING::outer();
    // keep verify default (false) unless overridden via CLI
    in.verify = false;

    for ( int i = 1; i < argc; ++i ) {
        std::string a = argv[i];
        if ( a == "--use-default-data" ) {
            in.use_default_data = true;
        } else if ( a == "--j-matrix-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--j-matrix-fname requires an argument\n"; return false; }
            in.filename_Jmatrix = argv[++i];
        } else if ( a == "--h-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--h-fname requires an argument\n"; return false; }
            in.filename_h = argv[++i];
        } else if ( a == "--n-replicas" ) {
            if ( i+1 >= argc ) { std::cerr << "--n-replicas requires an argument\n"; return false; }
            in.n_replicas = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--nsweeps" ) {
            if ( i+1 >= argc ) { std::cerr << "--nsweeps requires an argument\n"; return false; }
            in.nsweeps = static_cast<size_t>( std::stoul(argv[++i]) );
        } else if ( a == "--use-pt" ) {
            if ( i+1 >= argc ) { std::cerr << "--use-pt requires an argument\n"; return false; }
            in.use_pt = static_cast<bool>( std::stoul(argv[++i]) );
        } else if ( a == "--seed" ) {
            if ( i+1 >= argc ) { std::cerr << "--seed requires an argument\n"; return false; }
            in.seed = static_cast<unsigned>( std::stoul(argv[++i]) );
        } else if ( a == "--sweep" ) {
            if ( i+1 >= argc ) { std::cerr << "--sweep requires an argument\n"; return false; }
            in.sweep_name = argv[++i];
        } else if ( a == "--verify" ) {
            in.verify = true;
        } else if ( a == "--ref-solution-fname" ) {
            if ( i+1 >= argc ) { std::cerr << "--ref-solution-fname requires an argument\n"; return false; }
            in.filename_ref_solution = argv[++i];
        } else if ( a == "--help" || a == "-h" ) {
            printhelp( argv[0] );
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }

    // basic validation
    if ( !in.use_default_data ) {
        if ( in.filename_Jmatrix.empty() || in.filename_h.empty() ) {
            std::cerr << "Either --use-default-data or both --j-matrix-fname and --h-fname must be provided\n";
            return false;
        }
    }
    if ( in.verify && !in.use_default_data && in.filename_ref_solution.empty() ) {
        std::cerr << "--ref-solution-fname required when --verify is used without --use-default-data\n";
        return false;
    }
    return true;
}

// --- Minimal main that uses the existing ioProgram / grbProgram entrypoints ---
int main( int argc, char ** argv ) {
    std::cout << "simulated_anealing_re runner\n";
    input in;
    output out;

    if ( !parse_arguments( in, argc, argv ) ) {
        printhelp( argv[0] );
        return 1;
    }


    std::cout << "seed=" << in.seed << " n_replicas=" << in.n_replicas << " nsweeps=" << in.nsweeps << " sweep=" << in.sweep_name << "\n";

    // Run IO program (populates Storage or similar)
    {
        bool success = false;
        grb::Launcher< AUTOMATIC > launcher;
        grb::RC rc = launcher.exec( &ioProgram, in, success, true );
        if ( rc != SUCCESS ) {
            std::cerr << "I/O launcher failed: " << toString(rc) << "\n";
            return 2;
        }
        if ( !success ) {
            std::cerr << "I/O program reported failure\n";
            return 3;
        }
    }

    // Run main GraphBLAS program that builds data and calls reSA stub
    {
        grb::Launcher< AUTOMATIC > launcher;
        grb::RC rc = launcher.exec( &grbProgram, in, out, true );
        if ( rc != SUCCESS ) {
            std::cerr << "grbProgram launcher failed: " << toString(rc) << "\n";
            return 4;
        }
    }

    std::cout << "Finished: error_code=" << out.error_code << " iterations=" << out.iterations << " best_energy=" << out.best_energy << "\n";
    return out.error_code;
}
