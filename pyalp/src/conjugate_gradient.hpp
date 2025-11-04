#include <iostream>
#include <stdexcept>
#include <vector>
#include <exception>
#include <iostream>
#include <vector>

#ifdef _CG_COMPLEX
 #include <complex>
#endif

#include <inttypes.h>

#include <graphblas.hpp>

#include <graphblas/nonzeroStorage.hpp>

#include <graphblas/algorithms/conjugate_gradient.hpp>

#include <graphblas/utils/timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/utils/singleton.hpp>

#include <graphblas/utils/iterators/nonzeroIterator.hpp>

using BaseScalarType = double;
#ifdef _CG_COMPLEX
 using ScalarType = std::complex< BaseScalarType >;
#else
 using ScalarType = BaseScalarType;
#endif


constexpr const BaseScalarType tol = 0.000001;

/** The default number of maximum iterations. */
constexpr const size_t max_iters = 10000;

constexpr const double c1 = 0.0001;
constexpr const double c2 = 0.0001;

std::tuple<size_t, ScalarType>
conjugate_gradient(
    grb::Matrix< ScalarType > & L,
    grb::Vector< ScalarType > & x,
    grb::Vector< ScalarType > & b,
    grb::Vector< ScalarType > & r,
    grb::Vector< ScalarType > & u,
    grb::Vector< ScalarType > & temp,
    size_t solver_iterations = 1000,
    size_t verbose = 0
    //const struct input &data_in, struct output &out
    ) {
	size_t iterations = 0;
	BaseScalarType residual;

	if( !verbose )
	    std::cout << "conjugate_gradient:  start \n";
	// get user process ID
	const size_t s = grb::spmd<>::pid();
	(void)s;
	assert( s < grb::spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	grb::RC rc = grb::SUCCESS;
	rc = grb::algorithms::conjugate_gradient(
	    x, L, b,
	    solver_iterations, tol,
	    iterations, residual,
	    r, u, temp
	    );
	double single_time = timer.time();
	if( !(rc == grb::SUCCESS || rc == grb::FAILED) ) {
	    std::cerr << "Failure: call to conjugate_gradient did not succeed ("
		      << grb::toString( rc ) << ")." << std::endl;
	}
	if( rc == grb::FAILED ) {
	    if( !verbose ) {
		std::cout << "Warning: call to conjugate_gradient did not converge\n";
	    }
	}
	if( rc == grb::SUCCESS ) {
	    rc = grb::collectives<>::reduce( single_time, 0, grb::operators::max< double >() );
	}

	if( !verbose ) {
	    // output
	    std::cout << " solver_iterations = " << solver_iterations << "\n";
	    std::cout << " tol = " << tol << "\n";
	    std::cout << " iterations = " << iterations << "\n";
	    std::cout << " residual = " << residual << "\n";
	}

	if( !verbose ) {
	    std::cout << "conjugate_gradient:  end \n";
	}

	// Return as a tuple: (int, float)
	return std::make_tuple(iterations, residual);
}


