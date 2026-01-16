#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <cassert>

#include <graphblas/algorithms/simulated_annealing_re.hpp>
#include <graphblas.hpp>

using QType = double;
using StateType = int8_t;
using EnergyType = double;

constexpr EnergyType EPS = 1e-6;

template< typename T >
inline bool ISCLOSE( const T a, const T b ){
	return (std::abs<T>(a-b) < EPS) || (std::abs<T>((a-b)/a) < EPS);
}

struct data_in {
	// instance settings
	size_t n = 5;	// size of small instance
	size_t k = 3;	// number of small instances
	size_t degree = 5;
	// solver settings
	size_t n_replicas = 8;
	size_t nsweeps = 5;
	// global setting
	int seed = 0;
};

template<
	grb::Backend backend,
	grb::Descriptor descr = grb::descriptors::no_operation,
	class Ring = grb::Semiring<
		grb::operators::add< QType >, grb::operators::mul< QType >,
		grb::identities::zero, grb::identities::one
	>,
	typename Ttmp
	>
EnergyType get_energy(
				 const grb::Matrix< QType, backend >& couplings,
				 const grb::Vector< QType, backend > &local_fields,
				 const grb::Vector< StateType,backend > &state,
				 grb::Vector< Ttmp, backend > &tmp,
				 const Ring &ring = Ring()
			  ){
	const size_t n = grb::size( local_fields );
	assert( n == grb::size( state ) );
	assert( n == grb::ncols( couplings ) );
	assert( n == grb::nrows( couplings ) );
	grb::RC rc = grb::SUCCESS;
	EnergyType energy = 0.0;
	constexpr auto dense_descr = descr | grb::descriptors::dense;

	rc = rc ? rc : grb::resize( tmp, n );
	rc = rc ? rc : grb::set< descr >( tmp, 0.0 );
	rc = rc ? rc : grb::mxv< dense_descr >( tmp, couplings, state, ring );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, static_cast< QType >( 0.5 ), ring.getMultiplicativeMonoid() );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, local_fields, ring.getAdditiveMonoid() );
	rc = rc ? rc : grb::dot< dense_descr >( energy, tmp, state, ring );
	assert( rc == grb::SUCCESS );

	return energy;
}


template< grb::Backend backend >
void generate_random_qubo(
    const size_t n,
    const size_t k,
    grb::Vector< QType, backend >  &Q_diag,
    grb::Matrix< QType, backend > &Q_off,
    grb::Vector< StateType, backend >  &x_star,
    unsigned int seed = 0
) {
	grb::RC rc = grb::SUCCESS;
	rc = rc ? rc : grb::clear( Q_diag );
	rc = rc ? rc : grb::clear( Q_off );

	std::minstd_rand rng( seed );
	std::uniform_real_distribution< QType > weight_dist( -1, 1 );

	std::map< std::pair<size_t,size_t>, QType > Q;
	std::vector< QType > Qdiag ( n*k, 0 );
	for(size_t kk = 0; kk < k ; ++kk){
		for (size_t i = 0; i < n; ++i) {
			for (size_t j = i; j < n; ++j) {
				const QType val = weight_dist(rng);
				if (i == j){
					Qdiag[n*kk+i] = val;
				} else{
					Q[{n*kk+i, n*kk+j}] = val;
					Q[{n*kk+j, n*kk+i}] = val;
				}
			}
		}
	}
	std::vector< size_t > i, j;
	std::vector< QType > v;
	for(const auto &x : Q ){
		i.push_back( x.first.first );
		j.push_back( x.first.second );
		v.push_back( x.second );
	}

	rc = rc ?  rc : grb::buildVector( Q_diag, Qdiag.begin(), Qdiag.end(), grb::SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( Q_off,
			i.begin(), i.end(),
			j.begin(), j.end(),
			v.begin(), v.end(),
			grb::SEQUENTIAL );
	assert( rc == grb::SUCCESS );

	grb::Vector< StateType, backend > x ( n*k );
	grb::Vector< QType, backend > tmp ( n*k );
	rc = rc ? rc : grb::set( tmp, static_cast<StateType>( 0 ) );
	rc = rc ? rc : grb::set( x_star, static_cast<StateType>( 0 ) );

	QType min_energy = 0 ;

	for(size_t kk = 0; kk < k ; ++kk){
		rc = rc ? rc : grb::set( x, x_star );
		for (int64_t bits = 0; bits < (1 << n); ++bits) {
			for (size_t i = 0; i < n; ++i) {
				grb::setElement( x, (bits >> i) & 1, i + kk*n );
			}
			const double E = get_energy( Q_off, Q_diag, x, tmp );

			if ( E < min_energy - 1e-9 ) {
				min_energy = E;
				x_star = x;
			}
		}
	}
	assert( get_energy( Q_off, Q_diag, x_star, tmp ) == min_energy );
	assert( rc == grb::SUCCESS );
	return;
}

template< grb::Backend backend >
void generate_sparse_planted_qubo(
    const size_t n,
    const size_t degree,
    std::pair< QType, QType > weight_range,
    grb::Vector< QType, backend >  &Q_diag,
    grb::Matrix< QType, backend > &Q_off,
    const grb::Vector< StateType, backend > &x_star,
    double &E_star,
    unsigned int seed = 0
) {
	std::minstd_rand rng( seed );
	std::uniform_real_distribution< QType > weight_dist(weight_range.first, weight_range.second);

	grb::RC rc = grb::SUCCESS;
	rc = rc ? rc : grb::clear( Q_diag );
	rc = rc ? rc : grb::clear( Q_off );
	E_star = 0.0;

	std::map< std::pair<size_t,size_t>, QType > Q;
	std::vector< QType > Qdiag ( n, 0 );

    for (size_t i = 0; i < n; ++i) {
		std::vector< size_t > neighbors;
        for (size_t j = 0; j < n; ++j) {
            if (j != i) neighbors.push_back(j);
        }
		std::shuffle( neighbors.begin(), neighbors.end(), rng );
        neighbors.resize( degree );

        for (const auto&j : neighbors) {
            if (j < i) continue;

            const double w = weight_dist( rng );
            const int b = x_star[i] ^ x_star[j];

            if (b == 0) {
                Qdiag[i] += w;
                Qdiag[j] += w;
                Q[{i, j}] -= 2*w;
                Q[{j, i}] -= 2*w;
            } else {
                Qdiag[i] -= w;
                Qdiag[j] -= w;
                Q[{i, j}] += 2*w;
                Q[{j, i}] += 2*w;
                E_star += w;
            }
        }
    }
	std::vector< size_t > i, j;
	std::vector< QType > v;
	for(const auto &x : Q ){
		i.push_back( x.first.first );
		j.push_back( x.first.second );
		v.push_back( x.second );
	}

	rc = rc ? rc : grb::buildVector( Q_diag, Qdiag.begin(), Qdiag.end(), grb::SEQUENTIAL );
	rc = rc ? rc : grb::buildMatrixUnique( Q_off,
			i.begin(), i.end(),
			j.begin(), j.end(),
			v.begin(), v.end(),
			grb::SEQUENTIAL );
	assert( rc == grb::SUCCESS );
}

template< grb::Backend backend >
bool brute_force_check(
    const grb::Vector< QType, backend > &Q_diag,
    const grb::Matrix< QType, backend > &Q_off,
    const grb::Vector< StateType, backend > &x_star,
    const double opt_energy
) {
    const size_t n = grb::size( x_star );
    EnergyType min_energy = 1e7;
	std::vector< grb::Vector< StateType > > argmins;
	assert( n < 8 * sizeof( int64_t ) );

	grb::Vector< double, backend > tmp ( n );
	grb::Vector< StateType, backend > x ( n );
    for (int64_t bits = 0; bits < (1 << n); ++bits) {
        for (size_t i = 0; i < n; ++i) {
			grb::setElement( x, (bits >> i) & 1, i);
        }
        const double E = get_energy( Q_off, Q_diag, x, tmp );
		if(n < 6){
			std::bitset<5> x (bits);
			std::cerr << x << " --> " << E << std::endl;
		}

        if (E < min_energy - 1e-9) {
            min_energy = E;
            argmins = {x};
        } else if (abs(E - min_energy) < 1e-9) {
            argmins.push_back(x);
        }
    }

	std::cout << "Planted energy   : " << opt_energy << std::endl;
	std::cout << "Minimum found    : " << min_energy << std::endl;
	std::cout << "# ground states  : " << argmins.size() << std::endl;

    bool planted_ok = false;
    for (const auto &x : argmins) {
        if ( std::equal(x.begin(), x.end(), x_star.begin()) ) {
            planted_ok = true;
            break;
        }
    }

	std::cout << std::boolalpha;
	std::cout << "planted_is_optimal: " << planted_ok << std::endl;
    std::cout << "energy_matches: " << (std::abs(min_energy - opt_energy) < 1e-9) << std::endl;
    std::cout << "degeneracy " << argmins.size() << std::endl;
	return planted_ok;
}

void grbProgram( const struct data_in &in, grb::RC &rc ) {
	rc = grb::SUCCESS;
	const auto n = in.n;
	const auto k = in.k;
    const int degree = in.degree;
    const std::pair< QType, QType > weight_range = {0.1, 1.0};
    const unsigned int seed = in.seed;

    grb::Vector< QType > Q_diag ( n*k ), Q_diag_rand ( n*k );
    grb::Matrix< QType > Q_off ( n*k, n*k ), Q_off_rand ( n*k, n*k );
	grb::Vector< StateType > x_star ( n*k );
    double opt_energy = 0.0;

	generate_random_qubo( n, k, Q_diag_rand, Q_off_rand, x_star, seed );
    // generate_sparse_planted_qubo( n, degree, weight_range, Q_diag, Q_off, x_star, opt_energy, seed );

	const grb::Monoid< grb::operators::add<QType>, grb::identities::zero > addMonoid;
	// rc = rc ? rc :grb::foldl( Q_diag, Q_diag_rand, addMonoid );
	// rc = rc ? rc :grb::foldl( Q_off, Q_off_rand, addMonoid );
	
	rc = rc ? rc : grb::set( Q_diag, Q_diag_rand );
	// rc = rc ? rc : grb::set( Q_off, Q_off_rand );
	std::swap( Q_off, Q_off_rand );
	assert( rc == grb::SUCCESS );

    grb::Vector< QType > tmp ( n*k );
	rc = rc ? rc : grb::set( tmp, 0 );
    
    opt_energy = get_energy( Q_off, Q_diag, x_star, tmp );

	std::cout << "Optimal value: " << opt_energy << std::endl;

	if( n*k < 22 ){
		const bool optimal = brute_force_check(Q_diag, Q_off, x_star, opt_energy);
		if( !optimal ){
			rc = grb::FAILED;
			std::cerr << "Constructed solution is not optimal." << std::endl;
			return;
		}
	}
	assert( rc == grb::SUCCESS );

	std::cout << "------------------ Test with SA-RE ----------------------" << std::endl;
	grb::Vector< StateType > best_state ( n*k );
	EnergyType best_energy = 42;
	constexpr bool use_pt = true;
	constexpr EnergyType reference_energy = 0;
	const size_t nsweeps = in.nsweeps;
	const size_t n_replicas = in.n_replicas;
	const size_t s = grb::spmd<>::pid();

    std::minstd_rand rng ( seed + s ); // rng or std::mt19937

    // create states storage and initialize with random 1/0 values
    std::vector< grb::Vector< StateType > > states;
    for ( size_t r = 0; r < n_replicas; ++r ) {
        std::uniform_int_distribution< StateType > randint(0,1);
        std::vector< StateType > rand_data;
        for ( size_t i = 0; i < n*k; ++i ) {
            rand_data.emplace_back( static_cast< StateType >(
                randint( rng ) ) );
        }
        states.emplace_back( n*k );
        rc = rc ? rc : grb::buildVector(
            states.back(),
            rand_data.cbegin(),
            rand_data.cend(),
            grb::SEQUENTIAL
        );
    }
	assert( rc == grb::SUCCESS );

    // also make betas vector of size n_replicas and initialize with 10.0
    grb::Vector< QType > betas( n_replicas );
    grb::Vector< EnergyType > energies( n_replicas );
    for ( size_t r = 0; rc == grb::SUCCESS && r < n_replicas; ++r ) {
        rc = rc ? rc : grb::setElement( betas, static_cast< QType >( (10.0) * std::pow<QType>( 2, r ) ), r );
        rc = rc ? rc : grb::setElement( energies, get_energy( Q_off, Q_diag, states[r], tmp ), r );
    }
	assert( rc == grb::SUCCESS );

	rc = grb::algorithms::simulated_annealing_RE_Ising(
		 Q_off, Q_diag, states, energies, betas, best_state, best_energy, nsweeps, reference_energy, use_pt, seed
	);
	std::cout << "Optimized SA-RE value: " << best_energy << std::endl;
	std::cout << "Absolute error: " << best_energy-opt_energy << std::endl;
	std::cout << "Relative error: " << (best_energy-opt_energy)/best_energy << std::endl;

	if( !ISCLOSE(best_energy, opt_energy) ){
		rc = grb::FAILED;
	}
}

int main( int argc, char **argv ){
	struct data_in in;
	in.n = argc > 1 ? atoi(argv[1]) : in.n ;
	assert( in.n > 0 );
	in.k = argc > 2 ? atoi(argv[2]) : in.k ;
	assert( in.k > 0 );
	in.degree = argc > 3 ? atoi(argv[3]) : in.degree ;
	if( in.degree >= in.n ) in.degree = in.n-1;
	in.n_replicas = argc > 4 ? atoi(argv[4]) : in.n_replicas ;
	in.nsweeps = argc > 5 ? atoi(argv[5]) : in.nsweeps ;
	in.seed = argc > 6 ? atoi(argv[6]) : in.seed ;

	if( in.n == 0 || in.degree == 0 || in.n_replicas == 0 || argc > 7 ){
		std::cout << "Usage: " << std::endl;
		std::cout << argv[0] << " [n] [degree] [n_replicas] [nsweeps] [seed]" << std::endl;
		exit( 0 );
	}
	std::cout << "\tn = " << in.n << std::endl;
	std::cout << "\tk = " << in.k << std::endl;
	std::cout << "\ttotal size = " << in.n*in.k << std::endl;
	std::cout << "\tdegree = " << in.degree << std::endl;
	std::cout << "\tn_replicas = " << in.n_replicas << std::endl;
	std::cout << "\tnsweeps = " << in.nsweeps << std::endl;
	std::cout << "\tseed = " << in.seed << std::endl;
	assert( in.n < 22 );


	grb::RC out;
	grb::Launcher< grb::AUTOMATIC > launcher;
	grb::RC rc = launcher.exec( &grbProgram, in, out, true );
	if ( rc != grb::SUCCESS ) {
		std::cerr << "grbProgram launcher failed: " << toString(rc) << "\n";
		return 4;
	}
	std::cout << "Test " << (( out == grb::SUCCESS )? "OK" : "FAILED") << std::endl;
    return 0;

}
