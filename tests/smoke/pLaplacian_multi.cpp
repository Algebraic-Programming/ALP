#include <graphblas.hpp>
#include <graphblas/algorithms/pLaplacian_spectral_partition.hpp>

#include <iostream>
#include <exception>

using namespace grb;
using namespace algorithms;

void grbProgram( const void *, const size_t in_size, grb::RC &ret ) {
	
	if( in_size != 0 ) {
		(void) fprintf( stderr, "Unit test called with unexpected input\n" );
		ret = FAILED;
		return;
	}
/*
    const size_t n = 3;
    const size_t e = 3;
    const double weights_entries[] = { 1, 1, 1, 1, 1, 1 };
    const size_t I[] = { 0, 1, 0, 2, 1, 2 };
    const size_t J[] = { 1, 0, 2, 0, 2, 1 };
    // [Example data]

    grb::Matrix< double > W( n, n );
	grb::resize( W, 2*e );
    buildMatrixUnique( W, I, J, weights_entries, 2*e, SEQUENTIAL );

    const double test_vector[] = { 0.1, 2.1, -2.3 };
    grb::Vector< double > vec( n );
    buildVector( vec, test_vector, test_vector + 3, SEQUENTIAL );

    grb::eWiseLambda( [ &W, &vec ]( const size_t i, const size_t j, double &v ) {
        v = v*( vec[ i ] - vec[ j ] );
    }, W );

*/

    // [Example data]
    const size_t n = 10;
    const size_t e = 12;
    const size_t k = 3;
    const double weights_entries[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    const size_t I[] = { 0, 1, 0, 2, 1, 2, 3, 5, 4, 5, 6, 7, 7, 8, 7, 9, 8, 9, 6, 9, 2, 6, 5, 6 };
    const size_t J[] = { 1, 0, 2, 0, 2, 1, 5, 3, 5, 4, 7, 6, 8, 7, 9, 7, 9, 8, 9, 6, 6, 2, 6, 5 };

    //const double init_entries[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    //const size_t init_I[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    //const size_t init_J[] = { 0, 0, 0, 1, 1, 1, 2, 2, 2, 2 };
    // [Example data]

    //define the labels vector
    grb::Vector< size_t > x( n );

    //build the incidence matrix
    grb::Matrix< double > W( n, n );
	grb::resize( W, 2*e );
    buildMatrixUnique( W, I, J, weights_entries, 2*e, SEQUENTIAL );

    //run the pLaplacian procedure to obtain an approximation to a 1-eigenvector
    grb::algorithms::pLaplacian_multi( x, W, k, 1.01, 0.9 );

    // print out the partition
    std::cout << "Partition: ";
    for (const std::pair< size_t, long > &pair : x ){
        std::cout << pair.second << " "; 
    }
    std::cout << std::endl << std::endl;
}

int main( int argc, char **argv ) {
	(void) argc;
	(void) printf( "Functional test executable: %s\n", argv[ 0 ] );

	grb::RC rc = SUCCESS;
	grb::Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, rc ) != SUCCESS ) {
		(void) fprintf( stderr, "Test failed to launch\n" );
		rc = FAILED;
	}
	if( rc == SUCCESS ) {
		(void) printf( "Test OK.\n\n" );
	} else {
		fflush( stderr );
		(void) printf( "Test FAILED.\n\n" );
	}

	//done
	return 0;
}