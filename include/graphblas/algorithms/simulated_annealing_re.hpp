/*
 *   Copyright 2025 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 *
 * Provides a Simulated Annealing-Replica Exchange QUBO optimizator.
 *
 * @author Giovanni Gaio
 * @date TODO 2025
 */

#ifndef _H_GRB_ALGORITHMS_SA_RE
#define _H_GRB_ALGORITHMS_SA_RE

#include <vector>
#include <type_traits>
#include <tuple>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <cmath>

#ifndef NDEBUG
#include <iostream>
#endif

#include <graphblas.hpp>

#define ISCLOSE(a,b) (std::abs((b)-(a))/std::abs(a) < 1e-4) || (std::abs((b)-(a)) < 1e-4)

namespace grb {
	namespace internal {
		/*
		 * The following functions are used to ensure the correct type of the value in
		 * in the exponential function.
		 */
		template< typename T >
		inline T exp(T x ){
			static_assert(
					std::is_same<T, float>::value
				 || std::is_same<T, double>::value
				 || std::is_same<T, long double>::value
					);
			return std::exp( x );
		}

		template< typename T >
		inline T log(T x ){
			static_assert(
					std::is_same<T, float>::value
				 || std::is_same<T, double>::value
				 || std::is_same<T, long double>::value
					);
			return std::log( x );
		}
	} // namespace internal

	namespace algorithms {

		/*
		 * Do a Parallel Tempering pass.
		 * This means exchanging states at low temperature with states at higher temperature.
		 * To make the code simpler, this will be done by exchanging the temperatures instead.
		 *
		 * TODO: Fix this documentation.
		 *
		 * @param[in,out] states        On input: initial states.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     Inverse temperature of each state.
		 * 							The betas may be permuted.
		 *
		 * @tparam StateType	The state variable type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 *
		 * This implementation of parallel tempering does not use any spmd characteristics.
		 */
		template<
			Backend backend,
			typename StateType, 
			typename EnergyType,
			typename TempType
			>
	typename std::enable_if<
		(grb::_GRB_BACKEND != grb::BSP1D) || (backend == grb::BSP1D),
		grb::RC >::type
	pt(
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				const grb::Vector< TempType, backend > &betas
				){

			const size_t n_replicas = states.size();
			// const size_t s 		= spmd<>::pid();
			// const size_t nprocs = spmd<>::nprocs();
			grb::RC rc = grb::SUCCESS;

			for( size_t i = n_replicas - 1 ; i > 0 ; --i ){
				const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);

				if( de >= 0 || std::rand() < RAND_MAX * internal::exp( de ) ){
					std::swap( states[i], states[i-1] );
					std::swap( energies[i], energies[i-1] );
				}
			}

			return rc;
		}

		/*
		 * Implementation of parallel tempering using spmd.
		 */
		template<
			Backend backend,
			typename StateType, 
			typename EnergyType,
			typename TempType
			>
			typename std::enable_if<
				(grb::_GRB_BACKEND == grb::BSP1D) && (backend != grb::BSP1D),
				grb::RC >::type
		pt(
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				const grb::Vector< TempType, backend > &betas
				){
			static_assert( backend != grb::BSP1D );
			// static_assert( grb::_GRB_BACKEND == grb::BSP1D );

			const size_t n = grb::size( states[0] );
			const size_t n_replicas = states.size();
			const size_t s 		= spmd<>::pid();
			const size_t nprocs = spmd<>::nprocs();
			grb::RC rc = grb::SUCCESS;

#ifndef NDEBUG
			assert( grb::size(energies) == n_replicas );
			assert( grb::size(betas) == n_replicas );
#endif
			struct data {
					EnergyType e;
					TempType b;
					int r;
				};
			grb::Vector< StateType, backend > s0 ( n );
			grb::Vector< StateType, backend > s1 ( n );
			grb::set( s0, static_cast< StateType >( 0 ) );
			grb::set( s1, static_cast< StateType >( 0 ) );


			struct data msg[ 2 ];
			rc = rc ? rc : grb::resize( s0, n );
			rc = rc ? rc : grb::resize( s1, n );
			if( rc != grb::SUCCESS ) return rc;
			int rand = std::rand();

			for( size_t si = nprocs ; rc == grb::SUCCESS && si > 0; --si ){
				if( si-1 == s ){
					for( size_t i = n_replicas - 1 ; i > 0 ; --i ){
						const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);

						if( de >= 0 || std::rand() < RAND_MAX * internal::exp( de ) ){
							std::swap( states[i], states[i-1] );
							std::swap( energies[i], energies[i-1] );
						}
					}
					grb::set( s1, states[0] );
					msg[ 1 ].e = energies[ 0 ];
					msg[ 1 ].b = betas[0];
					// msg[ 1 ].r = rand;
				}else if( si-2 == s ){
					grb::set( s0, states[ n_replicas - 1 ] );
					msg[ 0 ].e = energies[ n_replicas - 1 ];
					msg[ 0 ].b = betas[ n_replicas - 1 ];
					msg[ 0 ].r = rand;
				}
				if( si == 1 ) continue;

#ifdef _GRB_WITH_LPF
				rc = rc ? rc : grb::internal::broadcast( s0, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].e, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].b, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].r, si-2 );
				rc = rc ? rc : grb::internal::broadcast( s1, si-1 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ].e, si-1 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ].b, si-1 );

				assert( grb::nnz(s0) == n ); // state has to be dense!
				assert( grb::nnz(s1) == n ); // state has to be dense!
#else
				assert( false ); // this should never run
#endif

#ifndef NDEBUG
	
				if( rc != grb::SUCCESS ){
					std::cerr << "\n\t Error in a collective broadcast " << rc << " : " <<
						grb::toString( rc ) << std::endl;
				}
				assert( rc == grb::SUCCESS );
#endif

				const EnergyType de = ( msg[ 1 ].e - msg[ 0 ].e ) * ( msg[ 1 ].b - msg[ 0 ].b );

				if( rc == grb::SUCCESS && ( de >= 0 || msg[ 0 ].r < RAND_MAX * internal::exp( de ) ) ){
					if( si == s+1 ){

						rc = rc ? rc : grb::set( states[ n_replicas - 1 ], s0 );
						rc = rc ? rc : grb::setElement(energies, msg[ 0 ].e, n_replicas - 1 );
					}else if( si ==  s+2 ){
						rc = rc ? rc : grb::set( states[ 0 ], s1 );
						rc = rc ? rc : grb::setElement(energies, msg[ 1 ].e, 0 );
					}
				}
			}
			return rc;
		}


		/*
		 * Estimate a solution to a given optimization problem. The solution is found
		 * using Simulated Annealing-Replica Exchange (also known as Parallel Tempering).
		 *
		 * The state will be optimized to minimize the value of an energy function $U(x)$,
		 * where $x$ is the state vector. Energies will be changed when changing the
		 * states, so that each energy is the actual energy of the relative state.
		 *
		 * The parameter sweep is a user-defined function that changes a given state
		 * (possibly randomly) and returns the variation of energy made from its
		 * changes of the state. It should take three parameters: a state vector, the
		 * inverse temperature (a scalar) and sweep_data.
		 *
		 * @param[in]     sweep      	The sweeping function.
		 * 								Should return the energy variation relative to the changes that it
		 * 								made on the state.
		 * @param[in]     sweep_data    Additional data to be passed to the sweep function.
		 * @param[in,out] states        On input: initial states.
		 *                              On output: optimized states.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     	Inverse temperature of each state.
		 * @param[in,out] best_state	The state with the minimum energy found by the algorithm.
		 * @param[in,out] best_energy	The minimum value of an energy found.
		 * @param[in]     n_sweeps      Number of Simulated Annealing iterations.
		 * @param[in]     use_pt		Whether to use Parallel Tampering or not.
		 *
		 * @tparam backend		The backend used for the single objects
		 * @tparam StateType	The state variable type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 * @tparam SweepDataType	Type of data to be passed on to the sweep function
		 * (e.g. a tuple of references to temporary vectors).
		 * @tparam SweepFuncType    The type of the function.
		 * The default value suggests the signature that the function should have.
		 *
		 */
		template<
			Backend backend,
			typename StateType, // type of state, possibly 0/1
			typename EnergyType,
			typename TempType,
			typename SweepDataType, // type of data to be passed through to the sweep function
			typename SweepFuncType = std::function< 
					EnergyType(
						 grb::Vector< StateType, backend >&,
						 const TempType&,
						 SweepDataType&
				 	)
				>
			>
		grb::RC simulated_annealing_RE(
				const SweepFuncType &sweep,
				SweepDataType& sweep_data,
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				grb::Vector< TempType, backend > &betas,
				grb::Vector< StateType, backend >  &best_state,
				EnergyType &best_energy,
				const size_t &n_sweeps,
				const bool &use_pt = false
				){

			const size_t s = spmd<>::pid();
			const size_t n_replicas = states.size();
			const size_t n = grb::size(states[0]);
			(void) n;
			(void) s;

			assert( n_replicas > 0 );
			assert( n_replicas == grb::size( betas ) );

			for(size_t i = 0; i < n_replicas ; ++i ){
				assert( n == grb::size( states[ i ] ) );
			}

			grb::RC rc = grb::SUCCESS;


#ifndef NDEBUG
			if( grb::spmd<>::pid() == 0 ) {
				std::cerr << "DEBUG: Called  simulated_annealing_RE with parameters: "
						  << "\n\t n = " << n
						  << "\n\t n_replicas = " << n_replicas
						  << "\n\t n_sweeps = " << n_sweeps
						  << "\n\t use_pt = " << use_pt
						  << std::endl;
			}
#endif

			best_energy = std::numeric_limits< EnergyType >::max();
			assert( grb::size(best_state) >= n );

			for( size_t i_sweep = 0 ; rc == grb::SUCCESS && i_sweep < n_sweeps ; ++i_sweep ){
				for( size_t j = 0 ; j < n_replicas ; ++j ){
					
					energies[j] += sweep( states[j], betas[j], sweep_data );
					grb::wait();
				
					// update_best state and energy
					if( energies[j] < best_energy ){
						best_energy = energies[j];
						best_state = states[j];
					}
				} // n_replicas
				if( rc == SUCCESS && use_pt ){
					// do a Parallel Tempering move
					rc = pt< backend >( states, energies, betas );
				}
#ifndef NDEBUG
				if( s == 0 ) {
					std::cerr << "Energy at iteration " << i_sweep << " = " << energies[ 0 ] << std::endl;
				}
#endif
			} // n_sweeps

#ifndef NDEBUG
			if( rc != grb::SUCCESS ){
				std::cerr << "ERROR at line " <<  __LINE__ << " in "
					      << __FILE__ << ": " << grb::toString( rc ) << "\n";
			}
#endif
			if( rc == SUCCESS ){
				rc = rc ? rc : grb::collectives<>::allreduce(
						best_energy, grb::operators::min< EnergyType >() );
			}
			
			return rc;
		}

		/*
		 * Create a set of independent masks.
		 *
		 * Uses a graph coloring algorithm.
		 * Adapted from Alg. 2 of `Graph Coloring on the GPU, M. Osama, M. Truong, C. Yang, A. Buluc, J.D. Owens`.
		 *
		 * @param[out] masks            The vector of constructed masks.
		 * @param[in]     A             The square (symmetric) A matrix. It should not contain any explicit zeros!
		 * @param[in]     frontier      Temporary (dense) vector used in the function
		 * @param[in]     w      Temporary (dense) vector used in the function
		 * @param[in]     seed      seed for randomization
		 *
		 * @tparam descr	grb::Descriptor for matrix operations. Should probably not be used
		 * @tparam backend	The backend of GraphBLAS to be used
		 * @tparam MaskType	The state variable type.
		 * @tparam AType	The matrix values' type.
		 *
		 */
		template<
			grb::Descriptor descr = grb::descriptors::no_operation,
			Backend backend,
			typename MaskType,
			typename AType,
			typename RSI, typename CSI, typename NZI
		>
		grb::RC matrix_partition(
				std::vector< grb::Vector< MaskType, backend > > &masks,
				const grb::Matrix< AType, backend, RSI, CSI, NZI > &A,
				grb::Vector< AType, backend > &frontier,
				grb::Vector< AType, backend > &w,
				const int seed = 42
				) {
			masks.clear();
			grb::RC rc = grb::SUCCESS;
			const size_t n = grb::nrows( A );
			const size_t s = spmd<>::pid();
			assert( n == grb::ncols( A ) ); // A needs to be square
			(void) s;

			grb::resize( frontier, n );
			grb::resize( w, n );

    		std::minstd_rand rng ( seed );
			std::uniform_real_distribution< AType > rand ( 0.1, 2.0 );

			for( size_t i = 0 ; i < n ; ++i ){
				rc = rc ? rc : grb::setElement( w, rand( rng ), i );
			}

			const grb::Semiring<
			grb::operators::max< AType >, grb::operators::right_assign< AType >,
			grb::identities::negative_infinity, grb::identities::zero
			> maxTimesRing;
			const grb::Monoid< grb::operators::add< AType >, grb::identities::zero > addMonoid;
			const  grb::operators::greater_than< AType > gtOp;
			const grb::Monoid< grb::operators::right_assign< AType >, grb::identities::zero > right_assign;
 
			for( size_t i = 0; rc == grb::SUCCESS && i < n ; ++i ) {
				// find max of neighbors
				rc = rc ? rc : grb::set< descr >( frontier, static_cast< AType >( 0 ) );
				rc = rc ? rc : grb::mxv< descr | grb::descriptors::dense >( frontier, A, w, maxTimesRing );
				rc = rc ? rc : grb::foldl< descr | grb::descriptors::dense >( frontier, w, gtOp );

				// is there any new node?
				AType succ = static_cast< AType >( 0 );
				rc = rc ? rc : grb::foldl< descr >( succ, frontier, addMonoid );
				if( succ <= 0 ){
					break;
				}

				// add new mask
				masks.emplace_back( grb::Vector< bool, backend >( n ) );
				auto &new_mask = masks.at(i);
				rc = rc ? rc : grb::resize( new_mask, n );
				rc = rc ? rc : grb::set< descr >( new_mask, frontier, static_cast< MaskType >(true) );

				// do not consider the weights of used nodes
				rc = rc ? rc : grb::foldl< descr >( w, new_mask,
						static_cast< AType >( 0 ), right_assign );
			}
			assert( rc == grb::SUCCESS );

#ifndef NDEBUG
			if( rc != grb::SUCCESS) {
				std::cerr << "Error in matrix_partition: " << rc << " " << grb::toString(rc) << std::endl;

			}
			size_t cnt = 0;
			if( s == 0 ) {
				std::cerr << "Final masks: \n";
			}
			for(const auto&mask : masks ){
				for( const auto &x : mask ){
					if( x.second ){
						if( s == 0 ) {
							std::cerr << x.first << ", ";
						}
						cnt ++;
					}
				}
				if( s == 0 ) {
					std::cerr << std::endl;
				}
			}
			if( s == 0 ){
				assert( cnt == n );
			}
#endif
			return rc;
		}
	
		/*
		 * Estimate a solution to a given Ising problem. The solution is found
		 * using the Simulated Annealing-Replica Exchange function above.
		 *
		 * The function minimized is $U(x) = x^T(\frac{1}{2}Jx + h)$, where $J$ is the supplied
		 * couplings matrix and $h$ is the local_fields vector. The solution is searched
		 * in the space of vectors $x$ with entries $0$ or $1$.
		 *
		 * states should be a vector of already initialized and filled dense grb::Vector.
		 *
		 *  TODO: expand and complete documentation
		 *
		 * Warning: This function allocates $O(n)$ memory for temporary vectors.
		 *
		 * @param[in,out] states        On input: initial (dense) states.
		 *                              On output: optimized (dense) states.
		 * @param[in]     couplings     The square (symmetric) couplings matrix.
		 *                              The diagonal has to be zero!
		 * @param[in]     local_fields  The vector of local fields.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     	Inverse temperature of each state.
		 * @param[in]     n_sweeps      Number of Simulated Annealing iterations.
		 * @param[in]     use_pt		Whether to use Parallel Tampering or not.
		 *
		 * @tparam StateType	The state variable type.
		 * @tparam QType		The matrix values' type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 * @tparam SweepDataType	Type of data to be passed on to the sweep function
		 * (e.g. a tuple of references to temporary vectors).
		 *
		 */
		template<
			Backend backend,
			grb::Descriptor descr = grb::descriptors::no_operation,
			bool empty_local_fields = false,
			typename StateType, // type of state, possibly 0/1
			typename QType, // type of coupling matrix values
			typename EnergyType,
			typename TempType,
			typename RSI, typename CSI, typename NZI,
			class Ring = Semiring<
				grb::operators::add< QType >, grb::operators::mul< QType >,
				grb::identities::zero, grb::identities::one
			>
			>
		grb::RC simulated_annealing_RE_Ising(
				const grb::Matrix< QType, backend, RSI, CSI, NZI > &couplings,
				const grb::Vector< QType, backend> &local_fields,
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				grb::Vector< TempType, backend > &betas,
				grb::Vector< StateType, backend > &best_state,
				EnergyType &best_energy,
				const size_t &n_sweeps,
				const bool &use_pt = false,
				const int seed = 42,
				const Ring &ring = Ring()
				){
			const size_t n = grb::size( states[0] );
			const size_t n_replicas = grb::size(betas);
			const size_t s 		= spmd<>::pid();
			(void) s;
			grb::RC rc = grb::SUCCESS;

			assert( grb::size(states[0]) == n );
			assert( grb::nnz(states[0]) == n ); // state is dense
			assert( states.size() == n_replicas );

			EnergyType energy;
			grb::Vector< EnergyType, backend > tmp_calc_energy ( n );

			const auto get_energy = [&couplings, &local_fields, &tmp_calc_energy, &ring](
					EnergyType &energy, const grb::Vector< StateType > &state
					){
				const size_t n = grb::size( local_fields );
				(void) n;
				assert( n == grb::size( state ) );
				assert( n == grb::ncols( couplings ) );
				assert( n == grb::nrows( couplings ) );
				grb::RC rc = grb::SUCCESS;
				grb::set( tmp_calc_energy, static_cast<EnergyType>( 0.0 ) );
				rc = rc ? rc : grb::mxv< descr | grb::descriptors::dense >( tmp_calc_energy, couplings, state, ring );
				rc = rc ? rc : grb::foldl< descr | grb::descriptors::dense >( tmp_calc_energy, static_cast< EnergyType >( 0.5 ),
						ring.getMultiplicativeMonoid() );
				if( !empty_local_fields) {
					rc = rc ? rc : grb::foldl< descr | grb::descriptors::dense >( tmp_calc_energy, local_fields, ring.getAdditiveMonoid() );
				}
				rc = rc ? rc : grb::dot< descr | grb::descriptors::dense >( energy, tmp_calc_energy, state, ring );
				return rc;
			};

			// it is reasonable to allow the energies to be allocated and evaluated by this function
			if( grb::nnz(energies) == 0 ){
				grb::resize( energies, n_replicas );

				for(size_t i = 0 ; i < n_replicas ; ++i){
					energy = static_cast< EnergyType >( 0.0 );
					rc = rc ? rc : get_energy( energy, states[i] );
					grb::setElement( energies, energy, i );
				}
			}

			grb::Vector< QType, backend > h ( n );
			grb::Vector< QType, backend > rand ( n );
			grb::Vector< StateType, backend > delta ( n );
			grb::Vector< EnergyType, backend > dn ( n );
			grb::Vector< bool, backend > accept ( n );
			std::srand( static_cast<unsigned>( seed ) );
    		std::minstd_rand rng ( seed ); // minstd_rand or std::mt19937

			grb::resize( h, n );
			grb::resize( rand, n );
			grb::resize( delta, n );
			grb::resize( dn, n );
			grb::resize( accept, n );

			std::vector< grb::Vector< bool, backend > > masks ;
			rc = rc ? rc : matrix_partition< descr >( masks, couplings, h, rand, seed );
			grb::clear(h);
			constexpr auto dense_descr = descr | grb::descriptors::dense;

			auto sweep_data = std::tie(
					(const typeof(couplings)&) couplings,
					(const typeof(local_fields)&) local_fields,
					(const typeof(masks)&) masks,
					h,
					rand,
					delta,
					dn,
					accept,
					rng,
					(const typeof(ring)&) ring
					);

#ifdef NDEBUG
            const auto ising_sweep = [](
#else
            const auto ising_sweep = [&get_energy](
#endif
				 grb::Vector< StateType, backend > &state,
				 const TempType &beta,
				 typeof(sweep_data) &data
			  ){
				const size_t s 		= spmd<>::pid();
				(void) s;

				const auto &couplings = std::get<0>(data);
				const auto &local_fields = std::get<1>(data);
				const auto &masks = std::get<2>(data);
				auto &h = std::get<3>(data);
				auto &rand = std::get<4>(data);
				auto &delta = std::get<5>(data);
				auto &dn = std::get<6>(data);
				auto &accept = std::get<7>(data);
				auto &rng = std::get<8>(data);
				const auto &ring = std::get<9>(data);

				const size_t n = grb::size( state );
				EnergyType delta_energy = static_cast< EnergyType >(0.0);
				grb::RC rc = grb::SUCCESS;
				(void) n;

				if( !empty_local_fields) {
					rc = rc ? rc : grb::set< descr >( h, local_fields );
				}else {
					rc = rc ? rc : grb::set< descr >( h, static_cast< QType >( 0.0 ) );
				}
				rc = rc ? rc : grb::mxv< dense_descr >( h, couplings, state , ring );

				std::exponential_distribution< EnergyType > rand_gen ( beta );
				for( size_t i = 0 ; i < n; ++i ){
					const auto rnd = -rand_gen( rng );
					grb::setElement( rand, rnd, i );
				}

				const grb::operators::leq< EnergyType > leq_operator;
#ifndef NDEBUG
				const grb::Vector< StateType > old_state = state;
#endif
				rc = rc ? rc : grb::wait();
				for(const auto &mask : masks ){
					// dn = (2*state_slice - 1) * h_slice
					rc = rc ? rc : grb::set< descr >( dn, mask, state );
					rc = rc ? rc : grb::foldl< descr >( dn, static_cast< EnergyType >( 2 ), ring.getMultiplicativeMonoid()  );
					rc = rc ? rc : grb::foldl< descr >( dn, static_cast< EnergyType >( -1 ), ring.getAdditiveMonoid() );
					rc = rc ? rc : grb::foldl< descr >( dn, h, ring.getMultiplicativeMonoid() );

					// Choose which changes to accept
					// ( dn >= 0 ) | ( rand/beta < dn )
					rc = rc ? rc : grb::foldl< descr >( dn, rand, leq_operator );
					rc = rc ? rc : grb::set< descr >( accept, dn, mask );

					// new_state = np.where(accept, 1 - old, old)
					rc = rc ? rc : grb::foldl< descr >( state, accept, static_cast< StateType >( -1 ), ring.getMultiplicativeMonoid() );
					rc = rc ? rc : grb::foldl< descr >( state, accept, static_cast< StateType >( 1 ), ring.getAdditiveMonoid() );
					
					// delta = new - old ==> delta[accept] = 2*new_state[accept]-1
					rc = rc ? rc : grb::set< descr >( delta, accept, state );
					rc = rc ? rc : grb::foldl< descr >( delta, accept, static_cast< StateType >( 2 ), ring.getMultiplicativeMonoid() );
					rc = rc ? rc : grb::foldl< descr >( delta, accept, static_cast< StateType >( -1 ), ring.getAdditiveMonoid() );
					
					// Update delta_energy -= dot(dn, accept)
					rc = rc ? rc : grb::dot< descr >( delta_energy, delta, h, ring );

					// update h
					rc = rc ? rc : grb::mxv< descr >( h, couplings, delta, ring );
				}
				rc = rc ? rc : grb::wait();

#ifndef NDEBUG
				if( rc != grb::SUCCESS ){
					std::cerr << "\n\t Error in some GraphBLAS function of ising_sweep " << rc << " : " << grb::toString( rc ) << std::endl;
					abort();
				}
				assert( rc == grb::SUCCESS );
				const auto new_state = state;
				rc = rc ? rc : grb::wait();

				EnergyType e1 = static_cast< EnergyType >( 0.0 ),
						   e2 = static_cast< EnergyType >( 0.0 );
				get_energy(e1, old_state);
				get_energy(e2, new_state);
				const auto real_delta = e2 - e1;
				if( s == 0 ){
					std::cerr << "\n\t Delta_energy: " << delta_energy;
					std::cerr << "\n\t Real delta: " << real_delta;
					std::cerr << "\n\t Discrepancy: " << real_delta - delta_energy;
					std::cerr << std::endl;
				}

				assert( ISCLOSE(real_delta, delta_energy ) );
#endif
				return delta_energy;
			};

			return simulated_annealing_RE(
					ising_sweep, sweep_data, states, energies, betas, best_state, best_energy, n_sweeps, use_pt
					);
		}

		/*
		 * Estimate a solution to a given QUBO problem. The solution is found
		 * using the Simulated Annealing-Replica Exchange function above.
		 *
		 * The function optimized is $U(x) = \frac{1}{2}x^TQx$, with the constraint that $x$ is a
		 * 0/1 vector.
		 *
		 *  TODO: expand and complete documentation
		 *
		 * Warning: This function allocates O(n*n_replicas) memory for temporary vectors.
		 *
		 * @param[in,out] states        On input: initial (dense) states.
		 *                              On output: optimized (dense) states.
		 * @param[in]     Q             The square (symmetric) Q matrix.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     	Inverse temperature of each state.
		 * @param[in]     n_replicas    Number of replicas to run concurrently.
		 * @param[in]     n_sweeps      Number of Simulated Annealing iterations.
		 * @param[in]     use_pt		Whether to use Parallel Tampering or not.
		 *
		 * @tparam StateType	The state variable type.
		 * @tparam QType		The matrix values' type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 *
		 */
		template<
			Backend backend,
			grb::Descriptor descr = grb::descriptors::no_operation,
			typename StateType, // type of state, possibly 0/1
			typename QType, // type of coupling matrix values
			typename EnergyType,
			typename TempType,
			typename RSI, typename CSI, typename NZI,
				class Ring = Semiring<
					grb::operators::add< QType >, grb::operators::mul< QType >,
					grb::identities::zero, grb::identities::one
				>
			>
		grb::RC simulated_annealing_RE_QUBO(
				const grb::Matrix< QType, backend, RSI, CSI, NZI > &Q,
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				grb::Vector< TempType, backend > &betas,
				grb::Vector< StateType, backend > &best_state,
				EnergyType &best_energy,
				const size_t &n_sweeps,
				const bool &use_pt = false,
				const int seed = 42,
				const Ring &ring = Ring()
				){
			grb::Vector< QType > empty_local_fields ( 0 );

			return simulated_annealing_RE_Ising< backend, descr, true >(
					Q, empty_local_fields, states, energies, betas, best_state, best_energy, n_sweeps, use_pt, seed, ring
					);
		}
	} // namespace algorithms
} // end namespace grb
#undef ISCLOSE

#endif // end _H_GRB_ALGORITHMS_SA-RE


