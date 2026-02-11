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

#ifdef TIMING
#include <iomanip>
#include <chrono>
#endif

#ifndef NDEBUG
#include <iostream>
#endif

#include <graphblas.hpp>

#define ISCLOSE(a,b) (std::abs((b)-(a))/std::abs(a) < 1e-4) || (std::abs((b)-(a)) < 1e-4)

namespace grb {
	namespace algorithms {

		/*
		 * Do a Parallel Tempering pass.
		 * This means exchanging states at low temperature with states at higher temperature.
		 *
		 * TODO: Complete this documentation.
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
				const grb::Vector< TempType, backend > &betas,
				const int seed = 42
				){

			const size_t n_replicas = states.size();
			// const size_t s 		= spmd<>::pid();
			// const size_t nprocs = spmd<>::nprocs();
			grb::RC rc = grb::SUCCESS;
			std::minstd_rand rng ( seed );
			std::exponential_distribution< EnergyType > rand ( 1.0 );

			for( size_t i = n_replicas - 1 ; i > 0 ; --i ){
				const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);

				if( -rand( rng ) < de ){
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
				const grb::Vector< TempType, backend > &betas,
				const int seed = 42
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
					EnergyType r;
				};
			// TODO: should these two be static? Probably.
			grb::Vector< StateType, backend > s0 ( n );
			grb::Vector< StateType, backend > s1 ( n );
			grb::set( s0, static_cast< StateType >( 0 ) );
			grb::set( s1, static_cast< StateType >( 0 ) );

			struct data msg[ 2 ];
			rc = rc ? rc : grb::resize( s0, n );
			rc = rc ? rc : grb::resize( s1, n );
			if( rc != grb::SUCCESS ) return rc;

			std::minstd_rand rng;
			std::exponential_distribution< EnergyType > rand ( 1.0 );

			rng.seed( seed + s );
			const EnergyType myrand = -rand( rng );

			for( size_t si = nprocs ; rc == grb::SUCCESS && si > 0; --si ){
				if( si == s + 1 ){
					for( size_t i = n_replicas - 1 ; i > 0 ; --i ){
						const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);

						if( -rand( rng ) < de ){
							std::swap( states[i], states[i-1] );
							std::swap( energies[i], energies[i-1] );
						}
					}
				}

				if( si == 1 ) continue;
				if( si == s + 1 ){
					grb::set( s1, states[0] );
					msg[ 1 ].e = energies[ 0 ];
					msg[ 1 ].b = betas[0];
					msg[ 1 ].r = myrand;
				}else if( si == s + 2 ){
					grb::set( s0, states[ n_replicas - 1 ] );
					msg[ 0 ].e = energies[ n_replicas - 1 ];
					msg[ 0 ].b = betas[ n_replicas - 1 ];
					msg[ 0 ].r = myrand;
				}

#ifdef _GRB_WITH_LPF
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ], si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ], si-1 );
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

				if( rc == grb::SUCCESS && ( msg[ 1 ].r < de ) ){
#ifdef _GRB_WITH_LPF
					rc = rc ? rc : grb::internal::broadcast( s1, si-1 );
					rc = rc ? rc : grb::internal::broadcast( s0, si-2 );
					assert( grb::nnz(s0) == n ); // state has to be dense!
					assert( grb::nnz(s1) == n ); // state has to be dense!
#else
					assert( false ); // this should never run
#endif
					if( si == s + 1 ){
						rc = rc ? rc : grb::set( states[ 0 ], s0 );
						rc = rc ? rc : grb::setElement( energies, msg[ 0 ].e, 0 );
					}else if( si ==  s + 2 ){
						rc = rc ? rc : grb::set( states[ n_replicas - 1 ], s1 );
						rc = rc ? rc : grb::setElement( energies, msg[ 1 ].e, n_replicas - 1 );
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
				const EnergyType &goal = 0,
				const bool &use_pt = false,
				const size_t &seed = 42
				){

			const size_t s = spmd<>::pid();
			const size_t nprocs = spmd<>::nprocs();
			const size_t n_replicas = states.size();
			const size_t n = grb::size(states[0]);
			(void) n;
			(void) nprocs;
			(void) s;

			grb::RC rc = grb::SUCCESS;

#ifndef NDEBUG
			assert( n_replicas > 0 );
			assert( n_replicas == grb::size( betas ) );

			for(size_t i = 0; i < n_replicas ; ++i ){
				assert( n == grb::size( states[ i ] ) );
			}
			if( s == 0 ) {
				std::cerr << "DEBUG: Called  simulated_annealing_RE with parameters: "
						  << "\n\t n = " << n
						  << "\n\t n_replicas = " << n_replicas
						  << "\n\t n_sweeps = " << n_sweeps
						  << "\n\t goal = " << goal
						  << "\n\t use_pt = " << use_pt
						  << "\n\t seed = " << seed
						  << std::endl;
			}
			assert( grb::size(best_state) == n );
#endif

			best_energy = std::numeric_limits< EnergyType >::max();

#ifdef TIMING
			auto start = std::chrono::high_resolution_clock::now();
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			if( s == 0 ){
				std::cerr << "Starting with sweep..." << "\n";
			}
#endif

			for( size_t i_sweep = 0 ; rc == grb::SUCCESS && i_sweep < n_sweeps ; ++i_sweep ){
#ifdef TIMING
				start = std::chrono::high_resolution_clock::now();
#endif

				for( size_t j = 0 ; j < n_replicas ; ++j ){

					energies[j] += sweep( states[j], betas[j], sweep_data );
					rc = rc ? rc : grb::wait< backend >(); // should be done with nonblocking backend, I guess
				
					// update_best state and energy
					if( energies[j] < best_energy ){
						best_energy = energies[j];
						best_state = states[j];
					}
				} // n_replicas

#ifdef TIMING
				end = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				if( s == 0 ){
					std::cerr << "Sweeps took " << (duration.count() / 1000.0) << " ms.\t";
				}
				start = std::chrono::high_resolution_clock::now();
#endif

				if( rc == SUCCESS && use_pt ){
					// do a Parallel Tempering move
					rc = pt( states, energies, betas, seed + i_sweep );
				}
#ifdef TIMING
				end = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				if(s == 0){
					std::cerr << "PT took " << (duration.count() / 1000.0) << " ms." << "\n";
				}
#endif


#ifndef NDEBUG
				if( s == 0 ) {
					std::cerr << "Energy at iteration " << i_sweep << " = " << best_energy << std::endl;
				}
#endif
				if( goal < -1 ){
					// TODO: find a better way than this, to avoid a sync at each iteration
					rc = rc ? rc : grb::collectives<>::allreduce(
							best_energy, grb::operators::min< EnergyType >() );
					if( best_energy <= goal ) i_sweep = n_sweeps;
				}
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
				// TODO: update best state to match best energy
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
			// assert( grb::is_symmetric( A ) );
			(void) s;

			grb::resize( frontier, n );
			grb::resize( w, n );

			std::minstd_rand rng ( seed );

			// random shuffle w
			for( size_t i = 0 ; i < n ; ++i ){
				rc = rc ? rc : grb::setElement( w, i+1, i );
			}
			for( size_t i = 0 ; i < n ; ++i ){
				std::uniform_int_distribution< size_t > rand ( i, n-1 );
				const auto j = rand( rng );
				const auto a = w[i];
				const auto b = w[j];
				rc = rc ? rc : grb::setElement( w, b, i );
				rc = rc ? rc : grb::setElement( w, a, j );
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
						cnt++;
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
		 * @param[in]     seed			Seed to use for internal randomization (must be the same for all processees);
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
				const EnergyType &goal = 0,
				const bool &use_pt = false,
				const int seed = 42,
				const Ring &ring = Ring()
				){
			const size_t n = grb::size( states[0] );
			const size_t n_replicas = grb::size(betas);
			const size_t s 		= spmd<>::pid();
			(void) s;
			grb::RC rc = grb::SUCCESS;

#ifndef NDEBUG
			assert( grb::nnz(states[0]) == n ); // state is dense
			assert( states.size() == n_replicas );
			// assert( grb::is_symmetric( couplings ) );
			for( const auto &state : states ){
				for( size_t i = 0; i < n; ++i ){
					assert( (state[i] == static_cast< StateType >( 0 )) ||
							(state[i] == static_cast< StateType >( 1 )) );
				}
			}

			assert( empty_local_fields || ( grb::size( local_fields ) == n ) );
			assert( empty_local_fields || ( grb::nnz(local_fields) == n ) );
#endif

#ifdef TIMING
			if( s == 0 ){
				std::cerr << "Starting simulated_annealing_RE_ising" << "\n";
			}
			auto start = std::chrono::high_resolution_clock::now();
#endif
			EnergyType energy;
			grb::Vector< EnergyType, backend > tmp_calc_energy ( n );

			const auto get_energy = [&couplings, &local_fields, &tmp_calc_energy, &ring, &n](
					EnergyType &energy, const grb::Vector< StateType, backend > &state
					){
				assert( n == grb::size( state ) );
				assert( n == grb::ncols( couplings ) );
				assert( n == grb::nrows( couplings ) );
				grb::RC rc = grb::SUCCESS;
				constexpr auto dense_descr = descr | grb::descriptors::dense;

				assert( empty_local_fields || grb::size( local_fields ) == grb::size( state ) );
				assert( grb::ncols( couplings ) == grb::size( state ) );
				assert( grb::nrows( couplings ) == grb::size( state ) );

				grb::set( tmp_calc_energy, static_cast<EnergyType>( 0.0 ) );
				rc = rc ? rc : grb::mxv< dense_descr >( tmp_calc_energy, couplings, state, ring );
				rc = rc ? rc : grb::foldl< dense_descr >( tmp_calc_energy, static_cast< EnergyType >( 0.5 ),
						ring.getMultiplicativeMonoid() );
				if( !empty_local_fields) {
					rc = rc ? rc : grb::foldl< dense_descr >( tmp_calc_energy, local_fields, ring.getAdditiveMonoid() );
				}
				rc = rc ? rc : grb::dot< dense_descr >( energy, tmp_calc_energy, state, ring );
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
			grb::Vector< QType, backend > delta ( n );
			grb::Vector< QType, backend > dn ( n );
			grb::Vector< bool, backend > accept ( n );
			std::minstd_rand rng ( seed ); // minstd_rand or std::mt19937

			rc = rc ? rc : grb::resize( h, n );
			rc = rc ? rc : grb::resize( rand, n );
			rc = rc ? rc : grb::resize( delta, n );
			rc = rc ? rc : grb::resize( dn, n );
			rc = rc ? rc : grb::resize( accept, n );
#ifdef TIMING
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			if(s == 0){
				std::cerr << std::fixed << std::setprecision(3);
				std::cerr << "Setup took " << (duration.count() / 1000.0) << " ms.\t";
			}
			start = std::chrono::high_resolution_clock::now();
#endif

			std::vector< grb::Vector< bool, backend > > masks ;
			rc = rc ? rc : matrix_partition< descr >( masks, couplings, h, rand, seed );
#ifdef TIMING
				end = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				if( s == 0 ){
					std::cerr << "Calculating masks took " << (duration.count() / 1000.0) << " ms.\t";
				}
				start = std::chrono::high_resolution_clock::now();
#endif

			rc = rc ? rc : grb::clear(h);
			constexpr auto dense_descr = descr | grb::descriptors::dense;

			auto sweep_data = std::tie(
					(const decltype(couplings)&) couplings,
					(const decltype(local_fields)&) local_fields,
					(const decltype(masks)&) masks,
					h,
					rand,
					delta,
					dn,
					accept,
					rng,
					(const decltype(ring)&) ring
					);

#ifdef NDEBUG
            const auto ising_sweep = [](
#else
            const auto ising_sweep = [&get_energy](
#endif
				 grb::Vector< StateType, backend > &state,
				 const TempType &beta,
				 decltype(sweep_data) &data
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

				assert( grb::nnz(state) == n ); // state has to be dense!

				if( !empty_local_fields) {
					rc = rc ? rc : grb::set< descr >( h, local_fields );
				}else {
					rc = rc ? rc : grb::set< descr >( h, static_cast< QType >( 0.0 ) );
				}
				rc = rc ? rc : grb::mxv< dense_descr >( h, couplings, state , ring );

				std::exponential_distribution< EnergyType > rand_gen ( beta );
				for( size_t i = 0 ; i < n; ++i ){
					const auto rnd = -rand_gen( rng );
					rc = rc ? rc : grb::setElement( rand, rnd, i );
				}

				const grb::operators::leq< EnergyType > leq_operator;
				const grb::operators::right_assign< EnergyType > right_assign_op;
				const grb::operators::not_equal< EnergyType > neq_operator;
#ifndef NDEBUG
				const grb::Vector< StateType, backend > old_state = state;
#endif
				rc = rc ? rc : grb::wait< backend >();
				for(const auto &mask : masks ){
					// dn = (2*state_slice - 1) * h_slice
					rc = rc ? rc : grb::set< descr >( dn, mask, state );
					rc = rc ? rc : grb::foldl< descr | grb::descriptors::invert_mask >( dn, state, static_cast< QType >( -1 ), right_assign_op );
					rc = rc ? rc : grb::foldl< descr >( dn, h, ring.getMultiplicativeMonoid() );
					assert( grb::nnz( dn ) == grb::nnz( mask ) );
#ifndef NDEBUG
					for( const auto x : dn ){
						assert( mask[x.first] == 1 );
						assert( ( (state[x.first] == 1) && ( x.second  == h[x.first]) ) ||
								( (state[x.first] == 0) && ( x.second  == -h[x.first]) ) );
					}
					const auto dn0 = dn;
#endif

					// Choose which changes to accept
					// ( dn >= 0 ) | ( rand/beta < dn )
					rc = rc ? rc : grb::foldl< descr >( dn, rand, leq_operator );
					rc = rc ? rc : grb::set< descr >( accept, dn, mask );
					assert( grb::nnz( accept ) <= grb::nnz( mask ) );
#ifndef NDEBUG
					size_t cnt = 0;
					for( const auto x : dn0 ){
						const size_t i = x.first;
						assert( mask[x.first] == 1 );
						// assert( x.second );
						if( x.second >= rand[i] ){
							assert( dn[i] == 1 );
							assert( accept[i] == 1 );
							cnt++;
						}else{
							assert( dn[i] == 0 );
						}
					}
					assert( grb::nnz( accept ) == cnt );
#endif

					// new_state = np.where(accept, 1 - old, old)
					rc = rc ? rc : grb::foldl< descr >( state, accept, static_cast< StateType >( 1 ), neq_operator );
					
					// delta = new - old ==> delta[accept] = 2*new_state[accept]-1
					rc = rc ? rc : grb::set< descr >( delta, accept, state );
					rc = rc ? rc : grb::foldl< descr | grb::descriptors::invert_mask >( delta, delta, static_cast< QType >( -1 ), right_assign_op );
					
					// Update delta_energy -= dot(dn, accept)
					rc = rc ? rc : grb::dot< descr >( delta_energy, delta, h, ring );

					// update h
					rc = rc ? rc : grb::mxv< descr >( h, couplings, delta, ring );
				}
				rc = rc ? rc : grb::wait< backend >();

#ifndef NDEBUG
				if( rc != grb::SUCCESS ){
					std::cerr << "\n\t Error in some GraphBLAS function of ising_sweep " << rc << " : " << grb::toString( rc ) << std::endl;
					abort();
				}
				assert( rc == grb::SUCCESS );
				const auto new_state = state;
				rc = rc ? rc : grb::wait< backend >();

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
#ifdef TIMING
				end = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
				if( s == 0 ){
					std::cerr << "Final setup took " << (duration.count() / 1000.0) << " ms." << "\n";
				}
#endif

			return simulated_annealing_RE(
					ising_sweep, sweep_data, states, energies, betas, best_state, best_energy, n_sweeps, goal, use_pt, seed
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
				const EnergyType &goal = 0,
				const bool &use_pt = false,
				const int seed = 42,
				const Ring &ring = Ring()
				){
			grb::Vector< QType > empty_local_fields ( 0 );

			return simulated_annealing_RE_Ising< backend, descr, true >(
					Q, empty_local_fields, states, energies, betas, best_state, best_energy, n_sweeps, goal, use_pt, seed, ring
					);
		}
	} // namespace algorithms
} // end namespace grb
#undef ISCLOSE

#endif // end _H_GRB_ALGORITHMS_SA-RE


