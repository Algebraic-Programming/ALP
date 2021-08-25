#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/algorithms/multilevel_partition.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <exception>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

using namespace grb;
using namespace algorithms;



struct output {
    int error_code;
    char filename[1024];
    grb::utils::TimerResults times;
    PinnedVector<size_t> pinnedVector;
};

struct input {
    char filename[1024];
    bool direct;
    bool unweighted;
    size_t num_clusters;
};

// written by Gabriel
void hM2grbM( std::ifstream& infile, int rows, int cols, grb::Matrix< int > &A ) {
	// Transform a hypergraph in hMETIS format into row-net incidence matrix as a grb::Matrix
	std::string line;	
	std::vector< int > Ivec, Jvec;
	std::vector< int > Vvec;
	int i = 0;
	while( std::getline( infile, line ) ) {
		int number;
		int line_length = line.length()/2 + 1;
		std::stringstream numbers(line);

		for( int j = 0; j < line_length; ++j ) {
			numbers >> number;
			number -= 1;
			Ivec.push_back( i );
			Jvec.push_back( number );
			Vvec.push_back( 1 );
		}
		i++;
	}

	int* I = &Ivec[0];
	int* J = &Jvec[0];
	int* V = &Vvec[0];
	grb::resize( A, Vvec.size()) ;
	grb::buildMatrixUnique( A, &(I[0]), &(J[0]), &(V[0]), Vvec.size(), SEQUENTIAL );

}


void grbProgram(const struct input &data_in, struct output &out) {
    grb::utils::Timer timer;

    timer.reset();

    if (data_in.filename[0] == '\0') {
        std::cerr << "no file name given as input." << std::endl;
        out.error_code = ILLEGAL;
        return;
    }

    out.error_code = 0;
    std::ifstream myfile;
    out.times.io = timer.time();
    timer.reset();

    myfile.open(data_in.filename);
    std::string line;
    std::getline(myfile, line);

    std::stringstream numbers(line);
    Matrix< int > A( rows, cols );
    int cols, rows; 
    numbers >> rows >> cols;

    hM2grbM(myfile, rows, cols, A);
    myfile.close();

	out.times.preamble = timer.time();


    RC rc = SUCCESS;

    timer.reset();

    // initialize hgraph partitioner

    rc = grb::algorithms::partition(A, 2, 1.1);
    double single_time = timer.time();

	if (rc != SUCCESS)
	{
		std::cerr << "Failure: call to pLaplacian_multi did not succeed (" << toString(rc) << ")." << std::endl;
		out.error_code = 20;
	}
	if (rc == SUCCESS)
	{
		rc = collectives<>::reduce(single_time, 0, operators::max<double>());
	}
	if (rc != SUCCESS)
	{
		out.error_code = 25;
	}
	out.times.useful = single_time;

	//start postamble
	timer.reset();

	//set error code
	if (rc == FAILED)
	{
		out.error_code = 30;
		//no convergence, but will print output
	}
	else if (rc != SUCCESS)
	{
		std::cerr << "Benchmark run returned error: " << toString(rc) << "\n";
		out.error_code = 35;
		return;
	}

	//output
	out.pinnedVector = PinnedVector<size_t>(x, SEQUENTIAL);

	//finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	//done
	return;

}


int main(int argc, char **argv) {
    return 1;
}