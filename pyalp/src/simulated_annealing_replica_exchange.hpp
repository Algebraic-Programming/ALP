#include <iostream>
#include <stdexcept>
#include <vector>
#include <exception>
#include <iostream>
#include <vector>

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>
#include <graphblas/algorithms/simulated_annealing_re.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

using ScalarType = double;
using StateType = int8_t;

ScalarType SARE_QUBO(
    const grb::Matrix< ScalarType > & Q,
    std::vector< grb::Vector< StateType > > & states,
    grb::Vector< ScalarType > & energies,
    grb::Vector< ScalarType > & betas,
    grb::Vector< StateType > & best_state,
    const size_t solver_iterations = 100,
    const size_t seed = 0,
    size_t verbose = 0
    ) {
    ScalarType best_energy = std::numeric_limits<ScalarType>::max();

	if( !verbose ){
	    std::cout << "SARE_QUBO:  start \n";
	}
	// get user process ID
	const size_t s = grb::spmd<>::pid();
	(void)s;
	assert( s < grb::spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	grb::RC rc = grb::SUCCESS;
	rc = grb::algorithms::simulated_annealing_RE_QUBO<
		grb::_GRB_BACKEND,
		grb::descriptors::no_operation,
		StateType,
		ScalarType,
		ScalarType,
		ScalarType
		>(
	    Q, states, energies,
	    betas, best_state,
	    best_energy, solver_iterations,
	    0, 1, seed
	    );

	double single_time = timer.time();
	if( !(rc == grb::SUCCESS || rc == grb::FAILED) ) {
	    std::cerr << "Failure: call to SARE_QUBO did not succeed ("
		      << grb::toString( rc ) << ")." << std::endl;
	}
	if( rc == grb::FAILED ) {
	    if( !verbose ) {
		std::cout << "Warning: call to SARE_QUBO did not converge\n";
	    }
	}
	if( rc == grb::SUCCESS ) {
	    rc = grb::collectives<>::reduce( single_time, 0, grb::operators::max< double >() );
	}

	if( !verbose ) {
	    // output
	    std::cout << " seed = " << seed << "\n";
	    std::cout << " #iterations = " << solver_iterations << "\n";
	    std::cout << " best_energy = " << best_energy << "\n";
	    std::cout << " solver time = " << single_time << " s \n";
	}

	if( !verbose ) {
	    std::cout << "SARE_QUBO:  end \n";
	}

	return best_energy;
}

ScalarType SARE_Ising(
    const grb::Matrix< ScalarType > & Q,
    const grb::Vector< ScalarType > & h,
    std::vector< grb::Vector< StateType > > & states,
    grb::Vector< ScalarType > & energies,
    grb::Vector< ScalarType > & betas,
    grb::Vector< StateType > & best_state,
    const size_t solver_iterations = 100,
    const size_t seed = 0,
    size_t verbose = 0
    ) {
    ScalarType best_energy = 0;

	if( !verbose )
	    std::cout << "SARE_Ising:  start \n";
	// get user process ID
	const size_t s = grb::spmd<>::pid();
	(void)s;
	assert( s < grb::spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	grb::RC rc = grb::SUCCESS;
	rc = grb::algorithms::simulated_annealing_RE_Ising<
		grb::_GRB_BACKEND,
		grb::descriptors::no_operation,
		false,
		StateType,
		ScalarType,
		ScalarType,
		ScalarType
		>(
	    Q, h, states, energies,
	    betas, best_state,
	    best_energy, solver_iterations,
	    0, 1, seed
	    );

	double single_time = timer.time();
	if( !(rc == grb::SUCCESS || rc == grb::FAILED) ) {
	    std::cerr << "Failure: call to SARE_Ising did not succeed ("
		      << grb::toString( rc ) << ")." << std::endl;
	}
	if( rc == grb::FAILED ) {
	    if( !verbose ) {
		std::cout << "Warning: call to SARE_Ising did not converge\n";
	    }
	}
	if( rc == grb::SUCCESS ) {
	    rc = grb::collectives<>::reduce( single_time, 0, grb::operators::max< double >() );
	}

	if( !verbose ) {
	    // output
	    std::cout << " seed = " << seed << "\n";
	    std::cout << " #iterations = " << solver_iterations << "\n";
	    std::cout << " best_energy = " << best_energy << "\n";
	    std::cout << " solver time = " << single_time << " s \n";
	}

	if( !verbose ) {
	    std::cout << "SARE_Ising:  end \n";
	}

	return best_energy;
}

