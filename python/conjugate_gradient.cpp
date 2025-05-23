#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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

namespace py = pybind11;

using namespace grb;
using namespace algorithms;

using BaseScalarType = double;
#ifdef _CG_COMPLEX
 using ScalarType = std::complex< BaseScalarType >;
#else
 using ScalarType = BaseScalarType;
#endif

/** Parser type */
typedef grb::utils::MatrixFileReader<
	ScalarType,
	std::conditional<
		(sizeof(grb::config::RowIndexType) > sizeof(grb::config::ColIndexType)),
		grb::config::RowIndexType,
		grb::config::ColIndexType
	>::type
> Parser;

/** Nonzero type */
typedef internal::NonzeroStorage<
	grb::config::RowIndexType,
	grb::config::ColIndexType,
	ScalarType
> NonzeroT;

/** In-memory storage type */
typedef grb::utils::Singleton<
	std::pair<
		// stores n and nz (according to parser)
		std::pair< size_t, size_t >,
		// stores the actual nonzeroes
		std::vector< NonzeroT >
	>
> Storage;

constexpr const BaseScalarType tol = 0.000001;

/** The default number of maximum iterations. */
constexpr const size_t max_iters = 10000;

constexpr const double c1 = 0.0001;
constexpr const double c2 = 0.0001;

// struct input {
// 	char filename[ 1024 ];
// 	bool direct;
// 	bool jacobi_precond;
// 	size_t rep;
// 	size_t solver_iterations;
// };

// struct output {
// 	int error_code;
// 	size_t rep;
// 	size_t iterations;
// 	double residual;
// 	grb::utils::TimerResults times;

// };

// void ioProgram( const struct input &data_in, bool &success ) {
// 	success = false;
// 	// Parse and store matrix in singleton class
// 	auto &data = Storage::getData().second;
// 	try {
// 		Parser parser( data_in.filename, data_in.direct );
// 		assert( parser.m() == parser.n() );
// 		Storage::getData().first.first = parser.n();
// 		try {
// 			Storage::getData().first.second = parser.nz();
// 		} catch( ... ) {
// 			Storage::getData().first.second = parser.entries();
// 		}
// 		/* Once internal issue #342 is resolved this can be re-enabled
// 		for(
// 			auto it = parser.begin( PARALLEL );
// 			it != parser.end( PARALLEL );
// 			++it
// 		) {
// 			data.push_back( *it );
// 		}*/
// 		for(
// 			auto it = parser.begin( SEQUENTIAL );
// 			it != parser.end( SEQUENTIAL );
// 			++it
// 		) {
// 			data.push_back( NonzeroT( *it ) );
// 		}
// 	} catch( std::exception &e ) {
// 		std::cerr << "I/O program failed: " << e.what() << "\n";
// 		return;
// 	}
// 	success = true;
// }

std::tuple<py::array_t<double>, int, double>
conjugate_gradient_numpy2alp(
    int N,
    int M,
    py::array_t<int> Ipy,
    py::array_t<int> Jpy,
    py::array_t<double> Vpy,
    py::array_t<double> Bpy,
    py::array_t<double> Xpy,
    size_t solver_iterations
    //const struct input &data_in, struct output &out
    ) {

	int n = N;
	int error_code;
	PinnedVector< ScalarType > pinnedVector;
	//size_t solver_iterations = max_iters;
	size_t iterations = 0;
	double residual;

	std::cout << "conjugate_gradient_numpy2alp:  start \n";
	// get user process ID
	const size_t s = spmd<>::pid();
	assert( s < spmd<>::nprocs() );

	// get input n
	grb::utils::Timer timer;
	timer.reset();

	std::cout << " n = " << n << "\n";

	py::buffer_info buf_i = Ipy.request();
	int* ptr_i = static_cast<int*>(buf_i.ptr);
	std::vector<int> vec_i(ptr_i, ptr_i + buf_i.size);
	for (int i : vec_i) {
	    std::cout << i << " ";
	}
	std::cout << std::endl;

	py::buffer_info buf_j = Jpy.request();
	int* ptr_j = static_cast<int*>(buf_j.ptr);
	std::vector<int> vec_j(ptr_j, ptr_j + buf_j.size);
	for (int j : vec_j) {
	    std::cout << j << " ";
	}
	std::cout << std::endl;

	py::buffer_info buf_v = Vpy.request();
	ScalarType* ptr_v = static_cast<ScalarType*>(buf_v.ptr);
	std::vector<ScalarType> vec_v(ptr_v, ptr_v + buf_v.size);
	std::cout << "Vector V contents: ";
	for (ScalarType v : vec_v) {
	    std::cout << v << " ";
	}
	std::cout << std::endl;

	size_t nz = vec_i.size();
	std::cout << " nz = " << nz << "\n";
	assert( nz == vec_j.size() );
	assert( nz == vec_v.size() );

	grb::RC io_rc;
	grb::Matrix< ScalarType > L( n, n );
	io_rc = grb::buildMatrixUnique( L, vec_i.data(), vec_j.data() , vec_v.data(), nz, SEQUENTIAL );
	assert( io_rc == grb::SUCCESS );

	std::cout << "Matrix L has been built \n";

	Vector< ScalarType > x( n ), b( n ), r( n ), u( n ), temp( n );

	RC rc = SUCCESS;
	double* buf_b = static_cast<double*>(Bpy.request().ptr);
	rc = rc ? rc : grb::buildVector( b, buf_b, buf_b + Bpy.request().size, SEQUENTIAL );
	if( rc != SUCCESS ) { std::cout << "RHS vector: buildVector failed!\n "; }

	pinnedVector = PinnedVector< ScalarType >( b, SEQUENTIAL );
	std::cout << "First 10 nonzeroes of b are: ( ";
	for( size_t k = 0; k < n && k < 10; ++k ) {
	    const auto &value = pinnedVector.getNonzeroValue( k );
	    std::cout << value << " ";
	}
	std::cout << ")" << std::endl;


	double* buf_x = static_cast<double*>(Xpy.request().ptr);
	rc = rc ? rc : grb::buildVector( x, buf_x, buf_x + Xpy.request().size, SEQUENTIAL );
	if( rc != SUCCESS ) { std::cout << "X vector: buildVector failed!\n "; }

	pinnedVector = PinnedVector< ScalarType >( x, SEQUENTIAL );
	std::cout << "First 10 nonzeroes of x are: ( ";
	for( size_t k = 0; k < n && k < 10; ++k ) {
	    const auto &value = pinnedVector.getNonzeroValue( k );
	    std::cout << value << " ";
	}
	std::cout << ")" << std::endl;

	timer.reset();
	rc = conjugate_gradient(
	    x, L, b,
	    solver_iterations, tol,
	    iterations, residual,
	    r, u, temp
	    );
	double single_time = timer.time();
	if( !(rc == SUCCESS || rc == FAILED) ) {
	    std::cerr << "Failure: call to conjugate_gradient did not succeed ("
		      << toString( rc ) << ")." << std::endl;
	    error_code = 20;
	}
	if( rc == FAILED ) {
	    std::cout << "Warning: call to conjugate_gradient did not converge\n";
	}
	if( rc == SUCCESS ) {
	    rc = collectives<>::reduce( single_time, 0, operators::max< double >() );
	}
	if( rc != SUCCESS ) {
	    error_code = 25;
	}
	timer.reset();


	// output
	std::cout << " solver_iterations = " << solver_iterations << "\n";
	std::cout << " tol = " << tol << "\n";
	std::cout << " iterations = " << iterations << "\n";
	std::cout << " residual = " << residual << "\n";

	pinnedVector = PinnedVector< ScalarType >( x, SEQUENTIAL );
	std::cout << "First 10 nonzeroes of x are: ( ";
	for( size_t k = 0; k < n && k < 10; ++k ) {
	    const auto &value = pinnedVector.getNonzeroValue( k );
	    std::cout << value << " ";
	}
	std::cout << ")" << std::endl;

	// finish timing
	const double time_taken = timer.time();

	double* data = new double[n];
	for( size_t k = 0; k < n; ++k ) {
	    const auto &value = pinnedVector.getNonzeroValue( k );
	    data[k]=value;
	}

	// Capsule to manage memory (will delete[] when array is destroyed in Python)
	py::capsule free_when_done(data, [](void *f) {
	    delete[] reinterpret_cast<double*>(f);
	});

	// Create NumPy array that shares memory with C++
	py::array_t<double> arr({n}, {sizeof(double)}, data, free_when_done);


	std::cout << "conjugate_gradient_numpy2alp:  end \n";

	// Return as a tuple: (array, int, float)
	return std::make_tuple(arr, iterations, residual);
}


// Print a NumPy array as a std::vector (flattened)
void print_my_numpy_array(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::vector<double> vec(ptr, ptr + buf.size);

    std::cout << "Vector contents (flattened): ";
    for (double v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

PYBIND11_MODULE(conjugate_gradient_python, m) {
    m.def("print_my_numpy_array", &print_my_numpy_array, "Print a numpy array as a flattened std::vector");
    m.def("conjugate_gradient_numpy2alp", &conjugate_gradient_numpy2alp, "Pass numpy data to alp CG");
}
