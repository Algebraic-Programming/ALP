

#include <iostream>

#include <graphblas.hpp>

static constexpr size_t num_elements = 6UL;
size_t indices[num_elements] = { 2, 5, 15, 47, 77, 94 };
double values[num_elements] = { 2.0, 5.0, 15.0, 47.0, 77.0, 94.0 };

void grb_program( const size_t &n, grb::RC &rc ) {
	(void) rc;

	grb::Vector< double > sparse_in( n ), sparse_out( n );
	grb::buildVector( sparse_in, indices, indices + num_elements,
		values, values + num_elements, grb::SEQUENTIAL );

	grb::Monoid< grb::operators::add< double >, grb::identities::zero > plusM;

	grb::eWiseApply( sparse_out,  0.25, sparse_in, plusM.getOperator() );

	if( grb::nnz( sparse_out ) != num_elements ) {
		std::cerr << "wrong number of nonzeroes" << std::endl;
	}
	for( const auto& v : sparse_out ) {
		std::cout << v.first << ": " << v.second << std::endl;
	}
}

int main( int argc, char ** argv ) {
	(void) argc;
	(void) argv;
	size_t in = 100;


	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC out;
	if( launcher.exec( &grb_program, in, out, true ) != grb::SUCCESS ) {
		std::cerr << "Launching test FAILED\n";
		return 255;
	}
	if( out != grb::SUCCESS ) {
		std::cerr << std::flush;
		std::cout << "Test FAILED (" << grb::toString( out ) << ")" << std::endl;
	} else {
		std::cout << "Test OK" << std::endl;
	}
	return 0;
}

