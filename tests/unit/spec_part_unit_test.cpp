//test simple functions in spectral_partition.hpp

#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/algorithms/spectral_partition.hpp>
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
    static const size_t incidence_I[2*m] = { 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5 };
    static const size_t incidence_J[2*m] = { 0, 1, 0, 2, 1, 2, 2, 3, 3, 4, 3, 5 };

     static const long adjacency_entries[] = {  1, 1,
                                             1,    1,
                                             1, 1,    1,
                                                   1,    1, 1,
                                                      1,  
                                                      1       };
    static const size_t adjacency_I[] = { 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5 };
    static const size_t adjacency_J[] = { 1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 3 };

    static const double raw_initial[n] = { 0.1, -0.1, 0.2, -0.2, 0.1, 0.2 };
    // [Example data]


    //declare the rings and the partition vector
    const grb::Semiring<double> reals_ring;
	const grb::Semiring<long> integers_ring;

    //-----testing Fiedler_vector_incidence-----

    //define and initialise the Fiedler vector
    grb::Vector< double > x_1(n);
    grb::operators::right_assign< double, double, double > accum;
    buildVector( x_1, accum, &(raw_initial[0]), &(raw_initial[n]), SEQUENTIAL );

    //build the incidence matrix
    grb::Matrix< long > Inc( m, n );
	grb::resize( Inc, 2*m );
    buildMatrixUnique( Inc, &(incidence_I[0]), &(incidence_J[0]), incidence_entries, 2*m, SEQUENTIAL );

    //minimise Rayleigh quotient to get the Fiedler vector

    grb::algorithms::Fiedler_vector_incidence( x_1, Inc, 0.01 );

    //use the Fiedler vector to generate the partition
    grb::Vector< bool > par_1(n);
    grb::algorithms::spec_part_utils::general_rounding( par_1, x_1, (bool)1, (bool)0 );


    //-----testing Fiedler_vector_laplacian-----

    //define and initialise the Fiedler vector
    grb::Vector< double > x_2(n);
    buildVector( x_2, accum, &(raw_initial[0]), &(raw_initial[n]), SEQUENTIAL );

    //build the incidence matrix
    grb::Matrix< void > A( n, n );
	grb::resize( A, sizeof(adjacency_entries)/sizeof(long) );
    buildMatrixUnique( A, &(adjacency_I[0]), &(adjacency_J[0]), 2*m, SEQUENTIAL );

    //minimise Rayleigh quotient to get the Fiedler vector

    grb::algorithms::Fiedler_vector_laplacian( x_2, A, 0.01 );

    //use the Fiedler vector to generate the partition
    grb::Vector< bool > par_2(n);
    grb::algorithms::spec_part_utils::general_rounding( par_2, x_2, (bool)1, (bool)0 );

    //print out the partitions
    
    std::cout << "Partition from Fiedler_vector_incidence: ";
    for (const std::pair< size_t, long > &pair : par_1 ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl;
    std::cout << "Partition from Fiedler_vector_laplacian: ";
    for (const std::pair< size_t, long > &pair : par_2 ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl;

    grb::finalize();
    return 0;
}