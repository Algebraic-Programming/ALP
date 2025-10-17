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
#include <algorithm>
#include <cstdlib>
#include <cassert>

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
		 */
		template<
			typename StateType, 
			typename EnergyType,
			typename TempType,
			Backend backend
			>
		grb::RC pt(
				const std::vector< grb::Vector< StateType, backend > > &states,
				const grb::Vector< EnergyType > &energies,
				grb::Vector< TempType > &betas
				){
			const size_t n_replicas = states.size();

			for( size_t i = 1 ; i < n_replicas ; ++i ){
        		const EnergyType de = ( energies[ i ] - energies[ i-1 ]) * (betas[ i ] - betas[ i-1 ]);
				if( de >= 0 || std::rand() < RAND_MAX*std::exp(de) ){
					std::swap( betas[i], betas[i-1] );
				}	
			}

			return grb::SUCCESS;
		}

		/*
		 * Estimate a solution to a given Quadratic Unconstrained Binary Optimization
		 * (QUBO) optimization problem. The solution is found using Simulated Annealing-
		 * Replica Exchange (also known as Parallel Tempering).
		 *
		 * The state will be optimized to minimize the value of the energy $U(x)$,
		 * where $x$ is the binary state vector, and $couplings$ is the coupling matrix.
		 * Energies will be changed when changing the states, so that each energy is
		 * the actual energy of the relative state.
		 * The parameter sweep is a function that (randomly) changes a given state and
		 * returns the variation of energy made from its changes of the state.
		 *
		 * @param[in]     sweep      	The sweeping function.
		 * 								Should return the energy variation implied from the changes that it made on the state.
		 * @param[in,out] states        On input: initial states.
		 *                              On output: optimized states.
		 * @param[in]     couplings     The square (symmetric) couplings matrix.
		 * @param[in,out] energies      The initial energy of each state.
		 * @param[in,out] betas     	Inverse temperature of each state.
		 * @param[in]     n_replicas    Number of replicas to run concurrently.
		 * @param[in]     n_sweeps      Number of Simulated Annealing iterations.
		 * @param[in]     use_pt		Whether to use Parallel Tampering or not.
		 *
		 * @tparam QType		The coupling matrix and the local fields type.
		 * @tparam StateType	The state variable type.
		 * @tparam EnergyType	The energy type.
		 * @tparam TempType		The inverse temperature type.
		 * @tparam Ring			The semiring under which to make the sweeps.
		 *
		 */
		template<
			typename QType, // type of coupling matrix values
			typename StateType, // type of state, ideally 0/1
			typename EnergyType,
			typename TempType,
			typename RSI, typename CSI, typename NZI, Backend backend,
			class Ring = Semiring<
				grb::operators::add< QType >, grb::operators::mul< QType >,
				grb::identities::zero, grb::identities::one
				>
			>
		grb::RC simulated_annealing_RE(
				const std::function< 
					EnergyType(
						 const grb::Matrix< QType, backend, RSI, CSI, NZI >&,
						 const grb::Vector< QType, backend >&,
						 grb::Vector< StateType, backend >&,
						 const TempType&,
						 const Ring&
				 	)
				> &sweep,
				std::vector< grb::Vector< StateType, backend > > &states,
				const grb::Matrix< QType, backend, RSI, CSI, NZI > &couplings,
				const grb::Vector< QType, backend > &local_fields,
				grb::Vector< EnergyType > &energies,
				grb::Vector< TempType > &betas,
				const size_t &n_sweeps = 1,
				const bool &use_pt = false,
				const Ring &ring = Ring()
				){

			size_t n_replicas = states.size();

			assert( n_replicas > 0 );
			assert( n_replicas == grb::size( betas ) );
			assert( grb::ncols( couplings ) == grb::nrows( couplings ) );
			assert( grb::size( states[0] ) == grb::nrows( couplings ) );
			assert( grb::size( states[0] ) == grb::size( local_fields ) );

			for(size_t i = 1; i < n_replicas ; ++i ){
				assert( grb::size( states[0] ) == grb::size( states[ i ] ) );
			}

			const size_t n = grb::size(states[0]);

#ifndef NDEBUG
			std::cerr << "DEBUG: Called  simulated_annealing_RE with parameters: "
				      << "\n\t n = " << n
				      << "\n\t n_replicas = " << n_replicas
				      << "\n\t n_sweeps = " << n_sweeps
				      << "\n\t use_pt = " << use_pt
				      << std::endl;
#endif

			grb::RC rc = grb::SUCCESS;

			static std::vector< grb::Vector< StateType, backend > >  best_states;
			static grb::Vector< EnergyType > best_energies ( n_replicas );
			best_energies = energies;
			best_states =  states;

			for( size_t i_sweep = 0 ; rc == grb::SUCCESS && i_sweep < n_sweeps ; ++i_sweep ){
				// randomize order of replicas
				// std::random_shuffle( states.begin(), states.end() );

				/*
				grb::eWiseApply(energies, states, betas, 
						[&](auto state, auto beta){
						return sweep( couplings, local_fields, state, beta, ring )
						}
						);
						*/
				for( size_t j = 0 ; rc == grb::SUCCESS && j < n_replicas ; ++j ){
					
					energies[j] += sweep( couplings, local_fields, states[j], betas[j], ring );
				
					// update_best state and energy
					if( energies[j] < best_energies[j] ){
						best_energies[j] = energies[j];
						best_states[j] = states[j];
					}
				} // n_replicas

				if( rc == SUCCESS && use_pt ){ // Parallel Tempering move
					rc = pt( states, energies, betas );
				}
#ifndef NDEBUG
				std::cerr << "Energy at iteration " << i_sweep << " = " << energies[ 0 ] << std::endl;
#endif
			} // n_sweeps

#ifndef NDEBUG
			if( rc != grb::SUCCESS ){
				std::cerr << "ERROR at line " <<  __LINE__ << " in "
					      << __FILE__ << ": " << grb::toString( rc ) << "\n";
			}
#endif
			if( rc == SUCCESS ){
				// copy assignment throws an error. We'll do move-assignment I guess
				states = std::move(best_states);
				energies = std::move(best_energies);
			}

			return rc;
		}

	} // namespace algorithms

} // end namespace grb

#endif // end _H_GRB_ALGORITHMS_SA-RE


