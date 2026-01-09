#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>

#include <graphblas/algorithms/simulated_annealing_re.hpp>
#include <graphblas.hpp>

using QType = float;
using StateType = int8_t;
using EnergyType = double;

template< grb::Backend backend >
void generate_sparse_planted_qubo(
    int n,
    int degree,
    std::pair< QType, QType > weight_range,
    grb::Vector< QType, backend >  &Q_diag,
    grb::Matrix< QType, backend > &Q_off,
    grb::Vector< StateType, backend > &x_star,
    double &E_star,
    unsigned int seed = 0
) {
	std::minstd_rand rng( seed );
	std::uniform_int_distribution< StateType > int_dist(0, 1);
	std::uniform_real_distribution< QType > weight_dist(weight_range.first, weight_range.second);

	std::vector< StateType > x ( n );
	for( auto  &y : x ){
		y = int_dist( rng );
	}
	grb::resize( x_star, n );
	grb::buildVector( x_star, x.begin(), x.end(), grb::SEQUENTIAL );

	grb::clear( Q_diag );
	grb::clear( Q_off );
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

	grb::buildVector( Q_diag, Qdiag.begin(), Qdiag.end(), grb::SEQUENTIAL );
	grb::buildMatrixUnique( Q_off,
			i.begin(), i.end(),
			j.begin(), j.end(),
			v.begin(), v.end(),
			grb::SEQUENTIAL );
}

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
	rc = rc ? rc : grb::resize( tmp, n );
	EnergyType energy = 0.0;
	constexpr auto dense_descr = descr | grb::descriptors::dense;

	rc = rc ? rc : grb::set< descr >( tmp, 0.0 );
	rc = rc ? rc : grb::mxv< dense_descr >( tmp, couplings, state, ring );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, static_cast< QType >( 0.5 ), ring.getMultiplicativeMonoid() );
	rc = rc ? rc : grb::foldl< dense_descr >( tmp, local_fields, ring.getAdditiveMonoid() );
	rc = rc ? rc : grb::dot< dense_descr >( energy, tmp, state, ring );
	assert( rc == grb::SUCCESS );

	return energy;
}

template< grb::Backend backend >
bool brute_force_check(
    const grb::Vector< QType, backend > &Q_diag,
    const grb::Matrix< QType, backend > &Q_off,
    const grb::Vector< StateType, backend > &x_star,
    double E_star
) {
    const int n = grb::size( x_star );
    EnergyType min_energy = 1e7;
	std::vector< grb::Vector< StateType > > argmins;
	assert( n < 8 * sizeof( int64_t ) );

	grb::Vector< double, backend > tmp ( n );
	grb::Vector< StateType, backend > x ( n );
    for (int64_t bits = 0; bits < (1 << n); ++bits) {
        for (int i = 0; i < n; ++i) {
			grb::setElement( x, (bits >> i) & 1, i);
        }
        double E = get_energy( Q_off, Q_diag, x, tmp );

        if (E < min_energy - 1e-9) {
            min_energy = E;
            argmins = {x};
        } else if (abs(E - min_energy) < 1e-9) {
            argmins.push_back(x);
        }
    }

	std::cout << "Planted energy   : " << -E_star << std::endl;
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
    std::cout << "energy_matches: " << (std::abs(min_energy + E_star) < 1e-9) << std::endl;
    std::cout << "degeneracy " << argmins.size() << std::endl;
	return planted_ok;
}

void grbProgram( const size_t&n, grb::RC &rc ) {
	rc = grb::SUCCESS;
    const int degree = 4;
    const std::pair< QType, QType > weight_range = {1.0, 1.0};
    const unsigned int seed = 1;

    grb::Vector< QType > Q_diag ( n );
    grb::Matrix< QType > Q_off ( n, n );
	grb::Vector< StateType > x_star ( n );
    double E_star = 0.0;

    generate_sparse_planted_qubo( n, degree, weight_range, Q_diag, Q_off, x_star, E_star, seed );

    const bool optimal = brute_force_check(Q_diag, Q_off, x_star, E_star);
	// assert( optimal );

	std::cout << "------------------ Test with SA-RE ----------------------" << std::endl;
	grb::Vector< StateType > best_state ( n );
	EnergyType best_energy = 42;
	constexpr bool use_pt = true;
	constexpr EnergyType reference_energy = 0;
	constexpr size_t nsweeps = 100;
	constexpr size_t n_replicas = 16;
	const size_t s = grb::spmd<>::pid();

    std::minstd_rand rng ( seed + s ); // rng or std::mt19937

    // create states storage and initialize with random 1/0 values
    std::vector< grb::Vector< StateType > > states;
    for ( size_t r = 0; r < n_replicas; ++r ) {
        std::uniform_int_distribution< StateType > randint(0,1);
        std::vector< StateType > rand_data;
        for ( size_t i = 0; i < n; ++i ) {
            rand_data.emplace_back( static_cast< StateType >(
                randint( rng ) ) );
        }
        states.emplace_back( n );
        rc = rc ? rc : grb::buildVector(
            states.back(),
            rand_data.cbegin(),
            rand_data.cend(),
            grb::SEQUENTIAL
        );
    }

    // also make betas vector of size n_replicas and initialize with 10.0
    grb::Vector< QType > betas( n_replicas );
    grb::Vector< EnergyType > energies( n_replicas );
    grb::Vector< EnergyType > tmp_energy( n );
    for ( size_t r = 0; rc == grb::SUCCESS && r < n_replicas; ++r ) {
        rc = rc ? rc : grb::setElement( betas, static_cast< QType >( (10.0) * std::pow<QType>( 2, r ) ), r );
        rc = rc ? rc : grb::setElement( energies, get_energy( Q_off, Q_diag, states[r], tmp_energy ), r );
    }
	assert( rc == grb::SUCCESS );

	rc = grb::algorithms::simulated_annealing_RE_Ising(
		 Q_off, Q_diag, states, energies, betas, best_state, best_energy, nsweeps, reference_energy, use_pt, seed
	);
	assert( get_energy( Q_off, Q_diag, best_state, tmp_energy ) == best_energy );
	std::cout << "Optimized SA-RE value: " << best_energy << std::endl;

	if( best_energy != -E_star ){
		rc = grb::FAILED;
	}
}

int main() {
	const size_t in = 18;
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
