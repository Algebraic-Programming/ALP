#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <armadillo>
#include <graphblas/algorithms/spectral_graph_partition.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdlib.h> /* srand, rand */
#include <time.h>
#include <inttypes.h>


using namespace arma;
using namespace grb;
using namespace algorithms;

struct input
{
	char filename[1024];
	bool direct;
	bool unweighted;
	size_t num_clusters;	
	//size_t rep;
};

struct output
{
	int error_code;
	char filename[1024];
	//size_t rep;
	grb::utils::TimerResults times;
	PinnedVector<size_t> pinnedVector;
};

// Extract the data as an Armadillo matrix Mat of type T, if there is no data the matrix will be empty
  template<typename T>
  arma::Mat<T> load_mat(std::ifstream &file, const std::string &keyword) {
 	std::string line;
 	std::stringstream ss;
 	bool process_data = false;
 	bool has_data = false;
 	while (std::getline(file, line)) {
 		if (line.find(keyword) != std::string::npos) {
 			process_data = !process_data;
 			if (process_data == false) break;
 			continue;
 		}
 		if (process_data) {
 			ss << line << '\n';
 			has_data = true;
 		}
 	}
 
 	arma::Mat<T> val;
 	if (has_data) {
 		val.load(ss);
 	}
 	return val;
 }


void grbProgram(const struct input &data_in, struct output &out)
{
	//get input n
	grb::utils::Timer timer;
	timer.reset();

	//sanity checks on input
	if (data_in.filename[0] == '\0')
	{
		std::cerr << "no file name given as input." << std::endl;
		out.error_code = ILLEGAL;
		return;
	}

	//assume successful run
	out.error_code = 0;

	//create local parser
	grb::utils::MatrixFileReader<double, std::conditional<
											 (sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
											 grb::config::RowIndexType,
											 grb::config::ColIndexType>::type>
		parser(data_in.filename, data_in.direct);
	assert(parser.m() == parser.n());
	const size_t n = parser.n();
	out.times.io   = timer.time();
	timer.reset();

	//load into GraphBLAS
	Matrix<double> W(n, n);
	{
		const RC rc = buildMatrixUnique(W, parser.begin(SEQUENTIAL), parser.end(SEQUENTIAL), SEQUENTIAL);
		if (rc != SUCCESS)
		{
			std::cerr << "Failure: call to buildMatrixUnique did not succeed (" << toString(rc) << ")." << std::endl;
			out.error_code = 10;
			return;
		}
	}

	//check number of nonzeroes
	try
	{
		const size_t global_nnz = nnz(W);
		const size_t parser_nnz = parser.nz();
		if (global_nnz != parser_nnz)
		{
			std::cerr << "Failure: global nnz (" << global_nnz << ") does not equal parser nnz (" << parser_nnz << ")." << std::endl;
			out.error_code = 15;
			return;
		}
	}
	catch (const std::runtime_error &)
	{
		std::cout << "Info: nonzero check skipped as the number of nonzeroes cannot be derived from the matrix file header. The grb::Matrix reports " << nnz(W) << " nonzeroes.\n";
	}

	//if the input is unweighted, the weights of W need to be set to 1
	if (data_in.unweighted)
	{
		grb::set(W, W, 1.0);
	}

	//labels vector
	Vector<size_t> x(n);
	grb::set(x, 0); //make x dense

	out.times.preamble = timer.time();

    Matrix<double> A( n, n );

	//by default, copy input requested repetitions to output repetitions performed
	//out.rep = data_in.rep;


	//time a single call
	RC rc = SUCCESS;
	timer.reset();

	// Initialize parameters for the partitioner
	int kmeans_iters = 30;      // kmeans iterations 
	float final_p = 1.1;       // final value of p 
	float factor_reduce = 0.7; // reduction factor for the value of p

	// Load the eigenvecs from a txt file
	// std::ifstream file("datasets/V_Rect_5pt_4.txt");
    // arma::Mat<double> V = load_mat<double>(file, "V");
    // file.close();

	// Call the Multiway p-spectral partitioner 
	// rc = grb::algorithms::pLaplacian_multi(x, W, data_in.num_clusters, final_p, factor_reduce, kmeans_iters);
	// rc = grb::algorithms::pLaplacian_multi(x, W, V, data_in.num_clusters, final_p, factor_reduce, kmeans_iters);
	rc = grb::algorithms::spectral_graph_partitioner(x, W, data_in.num_clusters);
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

int main(int argc, char **argv)
{
	std::cout << "@@@@  ================================ @@@ " << std::endl;
	std::cout << "@@@@  Vanilla Spectral Clustering      @@@ " << std::endl;
	std::cout << "@@@@  ================================ @@@ " << std::endl
			  << std::endl;

	//sanity check
	if (argc < 5 || argc > 6)
	{
		std::cout << "Usage: " << argv[0] << " <dataset> <direct/indirect> <weighted/unweighted> <out_filename> <num_clusters> " << std::endl;
		std::cout << " -------------------------------------------------------------------------------- " << std::endl;
		//std::cout << "Usage: " << argv[0] << " <dataset> <direct/indirect> (inner iterations) (outer iterations)\n";
		std::cout << "INPUT" << std::endl;
		std::cout << "Mandatory: <dataset>, <direct/indirect>, <weighted/unweighted>, and <out_filename> are mandatory arguments" << std::endl;
		std::cout << "Optional : <num_clusters> integer >= 2. Default value is 2." << std::endl;
		std::cout << " -------------------------------------------------------------------------------- " << std::endl;
		//std::cout << "(inner iterations) is optional, the default is " << grb::config::BENCHMARKING::inner() << ". If set to zero, the program will select a number of iterations approximately required to take at least one second to complete.\n";
		//std::cout << "(outer iterations) is optional, the default is " << grb::config::BENCHMARKING::outer() << ". This value must be strictly larger than 0." << std::endl;
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
	if (argc >= 5)
	{
		in.num_clusters = strtoumax(argv[5], &end, 10);
		if (argv[5] == end)
		{
			std::cerr << "Could not parse argument " << argv[5] << " for number of clusters." << std::endl;
			return 102;
		}
	}

	/*
	//get inner number of iterations
	in.rep = grb::config::BENCHMARKING::inner();
	char * end = NULL;
	if( argc >= 4 ) {
		in.rep = strtoumax( argv[ 3 ], &end, 10 );
		if( argv[ 3 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 2 ] << " for number of inner experiment repititions." << std::endl;
			return 2;
		}
	}

	//get outer number of iterations
	size_t outer = grb::config::BENCHMARKING::outer();
	if( argc >= 5 ) {
		outer = strtoumax( argv[ 4 ], &end, 10 );
		if( argv[ 4 ] == end ) {
			std::cerr << "Could not parse argument " << argv[ 3 ] << " for number of outer experiment repititions." << std::endl;
			return 4;
		}
	}
    

	std::cout << "Executable called with parameters " << in.filename << ", inner repititions = " << in.rep << ", and outer reptitions = " << outer << std::endl;
    */

	//set standard exit code
	grb::RC rc = SUCCESS;

	//launch
	grb::Launcher<AUTOMATIC> launcher;
	rc = launcher.exec(&grbProgram, in, out, true);

	if (rc != SUCCESS)
	{
		std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
		return 6;
	}

	/*
	//launch estimator (if requested)
	if( in.rep == 0 ) {
		grb::Launcher< AUTOMATIC > launcher;
		rc = launcher.exec( &grbProgram, in, out, true );
		if( rc == SUCCESS ) {
			in.rep = out.rep;
		}
		if( rc != SUCCESS ) {
			std::cerr << "launcher.exec returns with non-SUCCESS error code " << (int)rc << std::endl;
			return 6;
		}
	}

	//launch benchmark
	if( rc == SUCCESS ) {
		grb::Benchmarker< AUTOMATIC > benchmarker;
		rc = benchmarker.exec( &grbProgram, in, out, 1, outer, true );
	}
	if( rc != SUCCESS ) {
		std::cerr << "benchmarker.exec returns with non-SUCCESS error code " << grb::toString(rc) << std::endl;
		return 8;
	} else if( out.error_code == 0 ) {
		std::cout << "Benchmark completed successfully and took " << out.iterations << " iterations to converge with residual " << out.residual << ".\n";
	}
    */

	std::ofstream outfile(out.filename, std::ios::out | std::ios::trunc);
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Exit with error code" << out.error_code << std::endl;
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Size of x is " << out.pinnedVector.length() << std::endl;
	std::cout << " @@@@@@@@@@@@@@@@@@@@ " << std::endl;
	std::cout << "Writing partition vector to file " << out.filename << std::endl;
	for (size_t i = 0; i < out.pinnedVector.length(); ++i)
	{
		outfile << out.pinnedVector[i] << std::endl;
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
