#include <graphblas.hpp>
#include <graphblas/utils/Timer.hpp>
#include <graphblas/utils/parser.hpp>
#include <graphblas/algorithms/pLaplacian_poweriter_partition.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdlib.h> /* srand, rand */
#include <time.h>
#include <inttypes.h>

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
		parser( data_in.filename, data_in.direct, false );
	const size_t n = parser.n();
	const size_t m = parser.m();
	out.times.io = timer.time();
	timer.reset();

	//load into GraphBLAS
	Matrix<double> A_hyper( m, n );
	{
		const RC rc = buildMatrixUnique( A_hyper, parser.begin(SEQUENTIAL), parser.end(SEQUENTIAL), SEQUENTIAL );
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
		const size_t global_nnz = nnz( A_hyper );
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
		std::cout << "Info: nonzero check skipped as the number of nonzeroes cannot be derived from the matrix file header. The grb::Matrix reports " << nnz(A_hyper) << " nonzeroes.\n";
	}

	//if the input is unweighted, the weights of W need to be set to 1
	if (data_in.unweighted)
	{
		grb::set( A_hyper, A_hyper, 1.0);
	}

	//labels vector
	Vector< size_t > x( n );
	grb::set(x, 0); //make x dense

	out.times.preamble = timer.time();

	//by default, copy input requested repetitions to output repetitions performed
	//out.rep = data_in.rep;

	//time a single call
	RC rc = SUCCESS;
	timer.reset();

	// Initialize parameters for the partitioner
	int kmeans_iters_ortho = 200;	// kortho iterations
	int kmeans_iters_kpp = 50;	   	// k++ iterations
	float final_p = 1.05;	   // final value of p
	float factor_reduce = 0.97; // reduction factor for the value of p

	// Call the Multiway p-spectral partitioner
	rc = grb::algorithms::pLaplacian_poweriter( x, A_hyper, data_in.num_clusters, final_p, factor_reduce, kmeans_iters_ortho, kmeans_iters_kpp );
	

	double single_time = timer.time();

	if (rc != SUCCESS)
	{
		std::cerr << "Failure: call to pLaplacian_poweriter did not succeed (" << toString(rc) << ")." << std::endl;
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
	std::cout << "@@@@  ================================================ @@@@ " << std::endl;
	std::cout << "@@@@  Multiway p-spectral power iteration partitioning @@@@ " << std::endl;
	std::cout << "@@@@  ================================================ @@@@ " << std::endl
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
