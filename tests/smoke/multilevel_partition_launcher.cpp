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
#include <string>
#include <inttypes.h>
#include <unordered_map>

using namespace grb;
using namespace algorithms;

struct input
{
	char filename[1024];
	bool direct;
	bool unweighted;
	//size_t rep;
	unsigned int k; // number of parts to compute
	double eps;     // maximum allowed load imbalance
};

struct output
{
	int error_code;
	char filename[1024];
	//size_t rep;
	grb::utils::TimerResults times;
	PinnedVector<size_t> pinnedVector;
};

// written by Gabriel
void hM2grbM( std::ifstream& infile, size_t rows, size_t cols,
	std::vector< size_t > &Ivec, std::vector< size_t > &Jvec, std::vector< unsigned int > &Vvec
) {
	// Transform a hypergraph in hMETIS format into row-net incidence matrix as a grb::Matrix
	std::string line;	

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

	
}

void matrixMarket2RowHyperGraph( Matrix< int > &A, int &num_edges, int &num_cols, std::vector< int > &Ivec, std::vector< int > &Jvec, std::vector< int > &Vvec ) {

	std::unordered_map< int, std::vector< int > > matrix_rows;
	int rows = grb::nrows( A );
	int cols = grb::ncols( A );
	std::vector< int > taken;
	for(const std::pair< std::pair< size_t, size_t >, int > &pair : A ) {
		int row = pair.first.first;
		int col = pair.first.second;

		matrix_rows[ row ].push_back( col );
		taken.push_back( row );
	}
	num_edges = matrix_rows.size();
	for( const int &i : taken ) {
		for( const int &v : matrix_rows[i] ) {
			Ivec.push_back( i );
			Jvec.push_back( v );
			Vvec.push_back( 1 );
			num_cols = std::max(num_cols, v);

		}
	}	

}

void grbProgram( const struct input &data_in, struct output &out) {
	grb::utils::Timer timer;
	const double c = 1.0 + data_in.eps;
	timer.reset();
	// very hacky fix for now
	std::string fileType = "hmetis";

	if (data_in.filename[0] == '\0') {
		std::cerr << "no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}
	out.error_code = 0;
	std::ifstream myfile;
	out.times.io = timer.time();
	timer.reset();
	int cols, rows, nnz;

	std::vector< size_t > Ivec, Jvec;
       	std::vector< unsigned int > Vvec;
	if ( fileType == "hmetis" ) {
		myfile.open(data_in.filename);
		std::string line;
		std::getline(myfile, line);
		std::stringstream numbers(line);
		
		numbers >> rows >> cols;
		
		hM2grbM( myfile, rows, cols, Ivec, Jvec, Vvec );
	} else {
		std::ifstream file( data_in.filename );
		std::string line;
		int count = 0;
		while(std::getline(file, line)) {
			std::istringstream iss(line);
			int r, c, v;
			if (count == 0) {
				iss >> rows >> cols >> nnz;

			} else {
				iss >> r >> c;
				std::cout << "r: " << int(r) << std::endl;
				std::cout << "c: " << int(c) << std::endl;
				Ivec.push_back( r-1 );
				Jvec.push_back( c-1 );
				Vvec.push_back( 1.0 );
				
			}
			count++;
		}
		
		myfile.close();
	}
	std::cout << rows << std::endl;
	Matrix< unsigned int > A( rows, cols );
	size_t * const I = &Ivec[0];
	size_t * const J = &Jvec[0];
	unsigned int * const V = &Vvec[0];
	grb::resize( A, Vvec.size() );
	grb::buildMatrixUnique( A , &(I[0]), &(J[0]), &(V[0]), Vvec.size(), PARALLEL );

	out.times.preamble = timer.time();

	RC rc = SUCCESS;


	timer.reset();

	// initialize hgraph partitioner
	rc = grb::algorithms::partition( A, data_in.k, c );
	double single_time = timer.time();

	if (rc != SUCCESS)
	{
		std::cerr << "Failure: call to multilevel_partition did not succeed (" << toString(rc) << ")." << std::endl;
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
	// out.pinnedVector = PinnedVector<size_t>(x, SEQUENTIAL);

	//finish timing
	const double time_taken = timer.time();
	out.times.postamble = time_taken;

	//done
	return;
}


int main(int argc, char **argv) {
	std::cout << "@@@@  =======================  @@@ " << std::endl;
	std::cout << "@@@@  Multilevel partitioning @@@ " << std::endl;
	std::cout << "@@@@  ======================= @@@ " << std::endl
			  << std::endl;

	//sanity check
	if (argc < 5 || argc > 7)
	{
		std::cout << "Usage: " << argv[0] << " <dataset> <direct/indirect> <weighted/unweighted> <out_filename> <num_clusters> <epsilon>" << std::endl;
		std::cout << " -------------------------------------------------------------------------------- " << std::endl;
		std::cout << "INPUT" << std::endl;
		std::cout << "Mandatory: <dataset>, <direct/indirect>, <weighted/unweighted>, and <out_filename> are mandatory arguments" << std::endl;
		std::cout << "Optional : <num_clusters> integer >= 2. Default value is 2." << std::endl;
		std::cout << "           <epsilon> double > 0, the maximum relative load imbalance. Default value is 0.1." << std::endl;
		std::cout << " -------------------------------------------------------------------------------- " << std::endl;
		return 0;
	}

	std::cout << "Running executable: " << argv[0] << std::endl;
	std::cout << " -------------------------------------------------------------------------------- " << std::endl;
	//the input struct
	struct input in;

	//the output struct
	struct output out;

	//get file name
	(void)strncpy(in.filename, argv[1], 1023);
	in.filename[1023] = '\0';

	//get direct or indirect addressing
	if (strncmp(argv[2], "direct", 6) == 0)
	{
		in.direct = true;
	}
	else
	{
		in.direct = false;
	}

	//get weighted or unweighted graoh
	if (strncmp(argv[3], "weighted", 8) == 0)
	{
		in.unweighted = false;
	}
	else
	{
		in.unweighted = true;
	}

	//get output file name
	(void)strncpy(out.filename, argv[4], 1023);
	in.filename[1023] = '\0';

	char *end = NULL;
	in.k = 2;
	if (argc >= 6)
	{
		in.k = strtoumax(argv[5], &end, 10);
		if (argv[5] == end)
		{
			std::cerr << "Could not parse argument " << argv[5] << " for number of clusters." << std::endl;
			return 102;
		}
	}

	in.eps = 0.1;
	if( argc >= 7 ) {
		in.eps = strtod( argv[6], &end );
		if( end == argv[6] ) {
			std::cerr << "Could not parse argument " << argv[6] << " for maximum relative load imbalance." << std::endl;
			return 103;
		}
	}

	//set standard exit code
	grb::RC rc = SUCCESS;

	//launch
	grb::Launcher<AUTOMATIC> launcher;
	grb::utils::Timer timer;
	double grob_time = 0;

	rc = launcher.exec(&grbProgram, in, out, true);
	grob_time += timer.time();
	if (rc != SUCCESS)
	{
		std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
		return 6;
	}

	std::ofstream outfile(out.filename, std::ios::out | std::ios::trunc);
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Exit with error code" << out.error_code << std::endl;
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Size of x is " << out.pinnedVector.size() << std::endl;
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Writing partition vector to file " << out.filename << std::endl;
	for (size_t i = 0; i < out.pinnedVector.size(); ++i)
	{
		outfile << out.pinnedVector.getNonzeroValue(i) << std::endl;
	}
	outfile.close();

	if (out.error_code != 0)
	{
		std::cout << "Test FAILED." << std::endl;
		;
	}
	else
	{
		std::cout << "Test SUCCEEDED." << std::endl;
		;
	}
	std::cout << std::endl;

	//done
	return 0;
}
