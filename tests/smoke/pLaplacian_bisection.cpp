//test simple functions in spectral_partition.hpp

#include <graphblas.hpp>
#include <graphblas/algorithms/pLaplacian_spectral_partition.hpp>
#include <graphblas/algorithms/spec_part_utils.hpp>

#include <iostream>
#include <exception>

using namespace grb;
using namespace algorithms;

int main(int argc, char ** argv ){
    (void) argc;
    (void) argv;


    // [Example data]
    const size_t m = 6;
    const size_t n = 6;
    static const long incidence_entries[] = { 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 };
    static const size_t I[2*m] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 };
    static const size_t J[2*m] = { 0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 3, 5 };
    static const double raw_initial[n] = { 0.1, -0.1, 0.2, -0.2, 0.1, 0.2 };
    // [Example data]


    //declare the rings
    const grb::Semiring<double> reals_ring;
	const grb::Semiring<long> integers_ring;

    //define and initialise the p-eigenvector
    grb::Vector< double > x(n);
    grb::operators::right_assign< double, double, double > accum;
    buildVector( x, accum, &(raw_initial[0]), &(raw_initial[n]), SEQUENTIAL );

    //build the incidence matrix
    grb::Matrix< long > A( m, n );
	grb::resize( A, 2*m );
    buildMatrixUnique( A, &(I[0]), &(J[0]), incidence_entries, 2*m, SEQUENTIAL );

    //run the pLaplacian procedure to obtain an approximation to a 1-eigenvector

    grb::algorithms::pLaplacian_bisection( x, A, 2.0, 5.0, 0.05, 3);

    //use the 1-eigenvector to generate the partition

    grb::Vector< bool > par(n);
    grb::algorithms::spec_part_utils::general_rounding( par, x, (bool)1, (bool)0 );

    //print out the partition
    
    std::cout << "Partition: ";
    for (const std::pair< size_t, long > &pair : par ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl << std::endl;

    grb::finalize();
    return 0;
}