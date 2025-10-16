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
#include <assert.h>

#ifndef NDEBUG
#include <iostream>
#endif


#include <graphblas.hpp>

namespace grb {

	namespace algorithms {

		/*
		 * Parallel Tempering
		 *
		 *
		 */
		template<
			typename StateType, 
			typename EnergyType,
			typename TempType,
			Backend backend
			>
		grb::RC pt(
				std::vector< grb::Vector< StateType, backend > > &states,
				grb::Vector< EnergyType > &energies,
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
		 * The state will be optimized to minimize the expression:
		 * $x^TQx$, where $x$ is the binary state vector, and $Q$ is the coupling matrix.
		 *
		 * @param[in,out] x              On input: an initial state.
		 *                               On output: the optimized state
		 * @param[in]     Q              The (square, symmetric) couplings matrix.
		 * @param[in]     te             Probabilities of flipping each bit at each
		 *                               iteration (values between 0 and 1)
		 * @param[in]     n_replicas     Number of replicas to run concurrently.
		 * @param[in]     n_sweeps       Number of iterations.
		 * @param[in]     seed 			 Seed to use in the generation of random bit flips.
		 *
		 * @tparam QType         The input/output vector nonzero type
		 * @tparam QType         The input/output vector nonzero type
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
				const grb::Matrix< QType, backend, RSI, CSI, NZI > &Q,
				const grb::Vector< QType, backend > &local_fields,
				grb::Vector< EnergyType > &energies,
				grb::Vector< TempType > &betas,
				const size_t &n_sweeps = 1,
				const bool &use_pt = false,
				const Ring &ring = Ring()
				){

			const size_t n_replicas = states.size();

			assert( n_replicas > 0 );
			assert( n_replicas == grb::size( betas ) );
			assert( grb::ncols( Q ) == grb::nrows( Q ) );
			assert( grb::size( states[0] ) == grb::nrows( Q ) );
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

			static std::vector< grb::Vector< StateType, backend > >  best_states = states;
			auto best_energies = energies;

			for( size_t i_sweep = 0 ; rc == grb::SUCCESS && i_sweep < n_sweeps ; ++i_sweep ){
				// randomize order of replicas
				std::random_shuffle( states.begin(), states.end() );

				for( size_t j = 0 ; rc == grb::SUCCESS && j < n_replicas ; ++j ){
					
					energies[j] += sweep( Q, local_fields, states[j], betas[j], ring );
				
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


