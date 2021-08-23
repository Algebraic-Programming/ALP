//test simple functions in spectral_partition.hpp

#include <graphblas.hpp>
#include <graphblas/utils/parser.hpp>
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
    static const double raw_norm[n] = { 3.0, 4.0, 0.0, 0.0, 0.0, 0.0 };
    static const double raw_rounding[n] = { 1.0, 2.0, 1.0, -2.0, -4.0, -2.0};
    // [Example data]


    //declare the rings
    const grb::Semiring<double> reals_ring;
	const grb::Semiring<long> integers_ring;

    //test p-norm

    grb::Vector<double> x(n);

    double nor = grb::p_norm( x, 2.0, reals_ring.getAdditiveMonoid() );
    std::cout << nor << std::endl;

    nor = grb::p_norm( x, 1.0, reals_ring.getAdditiveMonoid() );
    std::cout << nor << std::endl;

    nor = grb::p_norm( x, 1.5, reals_ring.getAdditiveMonoid() );
    std::cout << nor << std::endl;

    std::cout << std::endl;

    //test generalised rounding 

    grb::operators::right_assign< double, double, double > accum;

    buildVector(x, accum, &(raw_rounding[0]), &(raw_rounding[n]), SEQUENTIAL);

    grb::Vector< long > par(n);

    grb::algorithms::spec_part_utils::general_rounding( par, x, static_cast<long>(1), static_cast<long>(0) );
    
    for (const std::pair< size_t, long > &pair : par ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl << std::endl;

    //test the function computing the ratio Cheeger cut

    grb::Matrix< long > A( m, n );
	grb::resize( A, 2*m );

    buildMatrixUnique( A, &(I[0]), &(J[0]), incidence_entries, 2*m, SEQUENTIAL );

    double cut;
    grb::algorithms::spec_part_utils::ratio_cheeger_cut( cut, par, A, m, n, integers_ring );

    std::cout << cut << std::endl; //should be 1/min{3,6-3}=1/3

    std::cout << std::endl;

    //test the in-place elementwise phi_p function

    grb::algorithms::spec_part_utils::phi_p_normalize( x, 2.0, n, reals_ring.getAdditiveMonoid() );

    for (const std::pair< size_t, double > &pair : x ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl << std::endl;

    grb::finalize();
    return 0;
}