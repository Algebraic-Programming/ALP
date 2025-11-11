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
#include <algorithm>
#include <cstdlib>
#include <cmath>

#ifndef NDEBUG
#include <iostream>
#endif

#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/*
		 * Do a Parallel Tempering pass.
		 * This means exchanging states at low temperature with states at higher temperature.
		 * To make the code simpler, this will be done by exchanging the temperatures instead.
		 *
		 * @param[in] states        On input: initial states.
		 * @param[in] energies      The initial energy of each state.
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

				if( de >= 0 || std::rand() < RAND_MAX * exp( de ) ){
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
			struct data {
					grb::Vector< StateType, backend > s;
					EnergyType e;
					TempType b;
					int r;
				};
			struct data msg[ 2 ];
			grb::resize( msg[0].s, n );
			grb::resize( msg[1].s, n );
			int rand = std::rand();

			for( size_t si = nprocs ; rc == grb::SUCCESS && si > 0; --si ){
				if( si == s+1 ){
					for( size_t i = n_replicas - 1 ; i > 0 ; --i ){
						const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);

						if( de >= 0 || std::rand() < RAND_MAX * exp( de ) ){
							std::swap( states[i], states[i-1] );
							std::swap( energies[i], energies[i-1] );
						}
					}
					grb::set( msg[1].s, states[0] );
					msg[ 1 ].e = energies[ 0 ];
					msg[ 1 ].b = betas[0];
					// msg[ 1 ].r = rand;
				}else if( si == s+2 ){
					grb::set( msg[0].s, states[ n_replicas - 1 ] );
					msg[ 0 ].e = energies[ n_replicas - 1 ];
					msg[ 0 ].b = betas[ n_replicas - 1 ];
					msg[ 0 ].r = rand;
				}
				if( si == 1 ) continue;

				// std::cerr << "Calling broadcasts" << std::endl;
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].s, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].e, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].b, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 0 ].r, si-2 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ].s, si-1 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ].e, si-1 );
				rc = rc ? rc : grb::collectives<>::broadcast( msg[ 1 ].b, si-1 );

#ifndef NDEBUG
	
				if( rc != grb::SUCCESS ){
					std::cerr << "\n\t Error in a collective broadcast " << rc << " : " << grb::toString( rc ) << std::endl;
				}
				assert( rc == grb::SUCCESS );
#endif

				const EnergyType de = ( msg[ 1 ].e - msg[ 0 ].e ) * ( msg[ 1 ].b - msg[ 0 ].b );

				if( rc == grb::SUCCESS && ( de >= 0 || msg[ 0 ].r < RAND_MAX * exp( de ) ) ){
					if( si == s+2 ){
						states[ 0 ] = msg[ 0 ].s;
						energies[ 0 ] = msg[ 0 ].e;
						// betas[ 0 ] = msg[ 0 ].b;
					}else if( si ==  s+1 ){
						states[ n_replicas-1 ] = msg[ 1 ].s;
						energies[ n_replicas-1 ] = msg[ 1 ].e;
						// betas[ n_replicas-1 ] = msg[ 1 ].b;
					}
				}
			}

			return rc;
		}


		/*
		 * Estimate a solution to a given optimization problem. The solution is found
		 * using Simulated Annealing-Replica Exchange (also known as Parallel Tempering).
		 *
		 * The state will be optimized to minimize the value of the energy $U(x)$,
		 * where $x$ is the binary state vector, and $couplings$ is the coupling matrix.
		 * Energies will be changed when changing the states, so that each energy is
		 * the actual energy of the relative state.
		 * The parameter sweep is a function that (randomly) changes a given state and
		 * returns the variation of energy made from its changes of the state.
		 *
		 * @param[in]     sweep      	The sweeping function.
		 * 								Should return the energy variation relative to the changes that it made on the state.
		 * @param[in,out] states        On input: initial states.
		 *                              On output: optimized states.
		 * @param[in]     couplings     The square (symmetric) couplings matrix.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     	Inverse temperature of each state.
		 * @param[in,out] temp_states   Inverse temperature of each state.
		 * @param[in,out] temp_energies Inverse temperature of each state.
		 * @param[in]     n_replicas    Number of replicas to run concurrently.
		 * @param[in]     n_sweeps      Number of Simulated Annealing iterations.
		 * @param[in]     use_pt		Whether to use Parallel Tampering or not.
		 *
		 * @tparam StateType	The state variable type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 * @tparam SweepDataType	Type of data to be passed on to the sweep function (e.g. a tuple of references to temporary vectors).
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
				std::vector< grb::Vector< StateType, backend > >  &temp_states,
				grb::Vector< EnergyType, backend > &temp_energies,
				const size_t &n_sweeps,
				const bool &use_pt = false
				){

			const size_t s = spmd<>::pid();
			const size_t n_replicas = states.size();
			const size_t n = grb::size(states[0]);

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

			temp_energies = energies;
			temp_states =  states;

			for( size_t i_sweep = 0 ; rc == grb::SUCCESS && i_sweep < n_sweeps ; ++i_sweep ){
				for( size_t j = 0 ; j < n_replicas ; ++j ){
					
					energies[j] += sweep( states[j], betas[j], sweep_data );
					grb::wait();
				
					// update_best state and energy
					if( energies[j] < temp_energies[j] ){
						temp_energies[j] = energies[j];
						temp_states[j] = states[j];
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
			// grb::collectives<>::reduce(); ?
			if( rc == SUCCESS ){
				states = temp_states;
				energies = temp_energies;
			}
			
			return rc;
		}

		/*
		 * Estimate a solution to a given Ising problem. The solution is found
		 * using the Simulated Annealing-Replica Exchange function above.
		 *
		 *  TODO: expand and complete documentation
		 *
		 * This function allocates O(n*n_replicas) memory for temporary vectors.
		 *
		 * @param[in,out] states        On input: initial states.
		 *                              On output: optimized states.
		 * @param[in]     couplings     The square (symmetric) couplings matrix.
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
		 * @tparam SweepDataType	Type of data to be passed on to the sweep function (e.g. a tuple of references to temporary vectors).
		 *
		 */
		template<
			Backend backend,
			typename StateType, // type of state, possibly 0/1
			typename QType, // type of coupling matrix values
			typename EnergyType,
			typename TempType,
			typename SweepDataType, // type of data to be passed through to the sweep function
			typename SweepFuncType = std::function< 
					EnergyType(
						 grb::Vector< StateType, backend >&,
						 const TempType&,
						 SweepDataType&
				 	)
				>,
				typename RSI, typename CSI, typename NZI
			>
		grb::RC simulated_annealing_RE_Ising(
				const grb::Matrix< QType, backend, RSI, CSI, NZI >& Q,
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				grb::Vector< TempType, backend > &betas,
				const size_t &n_sweeps,
				const bool &use_pt = false
				);

		/*
		 * Estimate a solution to a given QUBO problem. The solution is found
		 * using the Simulated Annealing-Replica Exchange function above.
		 *
		 *  TODO: expand and complete documentation
		 *
		 * @param[in,out] states        On input: initial states.
		 *                              On output: optimized states.
		 * @param[in]     couplings     The square (symmetric) couplings matrix.
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
		 * @tparam SweepDataType	Type of data to be passed on to the sweep function (e.g. a tuple of references to temporary vectors).
		 *
		 */
		template<
			Backend backend,
			typename StateType, // type of state, possibly 0/1
			typename QType, // type of coupling matrix values
			typename EnergyType,
			typename TempType,
			typename SweepDataType, // type of data to be passed through to the sweep function
			typename SweepFuncType = std::function< 
					EnergyType(
						 grb::Vector< StateType, backend >&,
						 const TempType&,
						 SweepDataType&
				 	)
				>,
				typename RSI, typename CSI, typename NZI
			>
		grb::RC simulated_annealing_RE_QUBO(
				const grb::Matrix< QType, backend, RSI, CSI, NZI > &Q,
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType, backend > &energies,
				grb::Vector< TempType, backend > &betas,
				const size_t &n_sweeps,
				const bool &use_pt = false
				);

		template< typename T >
		inline T
		exp(T x ){
			static_assert(std::is_same<T, float>::value ||
				std::is_same<T, double>::value ||
				std::is_same<T, long double>::value);
			return std::exp( x );
		}
	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SA-RE


