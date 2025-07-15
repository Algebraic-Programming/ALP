
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
 * Implements the Ising machine SB
 *
 * @author Denis Jelovina
 */

#ifndef _H_GRB_ALGORITHMS_ISING_MACHINE_SB
#define _H_GRB_ALGORITHMS_ISING_MACHINE_SB

#include <graphblas.hpp>          // master ALP/GraphBLAS header
#include <algorithm>              // std::min/max
#include <cmath>                  // std::abs, std::sqrt
#include <iostream>
#include <vector>

#include <graphblas/algorithms/norm.hpp>


namespace grb {
	namespace algorithms {



		// // Custom unary operator for sign extraction
		// struct signum {
		// 	constexpr IOType operator()( const IOType x ) const noexcept {
		// 		return ( x > 0 ) - ( x < 0 );
		// 	}
		// };

		// Custom unary operator for hard clipping to [-1,1]
		template< typename IOType >
		struct clip11 {
			constexpr IOType operator()( const IOType x ) const noexcept {
				return std::min<IOType>( 1.0, std::max<IOType>( -1.0, x ) );
			}
		};

		// // Unary predicate to build a structural mask |x|>1
		// struct abs_gt1 {
		// 	constexpr bool operator()( const IOType x ) const noexcept {
		// 		return std::abs( x ) > 1.0;
		// 	}
		// };

		/*-------------------------------------------------------------*
		*  bSB — core optimisation routine                             *
		*-------------------------------------------------------------*/
			template<
				Descriptor descr = descriptors::no_operation,
				typename IsingHType,
				typename IOType,
				typename RSI, typename NZI, Backend backend,
				class Ring = Semiring<
					grb::operators::add< IOType >, grb::operators::mul< IOType >,
					grb::identities::zero, grb::identities::one
				>,
				class Minus = operators::subtract< IOType >,
				class Divide = operators::divide< IOType >
			>
			grb::RC bSB(
				std::vector< IOType > & energies, // output length num_iters
				grb::Vector< IOType, backend > & x_comp,      // in/out, size N
				grb::Vector< IOType, backend > & y_comp,      // in/out, size N
				const grb::Matrix< IsingHType, backend, RSI, RSI, NZI > &          J,           // NxN, symmetric
				const grb::Vector< IOType, backend > &          h,           // size N
				const IOType         p_init,
				const IOType         p_end,
				const std::size_t    num_iters,
				const IOType         dt,
				// default semiring, minus, divide
				const Ring &ring = Ring(),
				const Minus &minus = Minus(),
				const Divide &divide = Divide(),
				const std::function< IOType( IOType ) > &sqrtX =
					std_sqrt< IOType, IOType >
			) {

			constexpr const Descriptor descr_dense = descr | descriptors::dense;

			const std::size_t N = grb::nrows(J);

			assert( grb::ncols(J) == N );
			assert( grb::nrows(h) == N );
			assert( grb::ncols(h) == 1 );
			assert( grb::nrows(x_comp) == N );
			assert( grb::ncols(x_comp) == 1 );
			assert( grb::nrows(y_comp) == N );
			assert( grb::ncols(y_comp) == 1 );
			// TODO: check that J is symmetric once properly implemented
			//assert( grb::is_symmetric(J) );

			if ( num_iters < 1 ) {
				return grb::SUCCESS;
			}

			grb::RC rc = grb::SUCCESS;
			/* ---- pre-compute ---- */
			IOType sumJ2 = ring.template getZero< IOType >();
			rc = rc ? rc : grb::eWiseLambda( [&J, &sumJ2]( const size_t i, const size_t j, IOType& v ) {
				(void) i;
				(void) j;
				// rewrite this to use graphblas language
				sumJ2 += v*v;
			}, J );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error in eWiseLambda for sumJ2: " << rc << '\n';
				return rc;
			}
#ifdef DEBUG
			// for debugging purposes, print sumJ2
			std::cout << "sumJ2: " << sumJ2 << '\n';
#endif

			// rewrite this to use graphblas language
			rc = rc ? rc : grb::foldl( sumJ2, static_cast<IOType>( N - 1 ), divide );
			IOType xi = 0.5;
			sumJ2 = sqrtX( sumJ2 );
			rc = rc ? rc : grb::foldl( xi, sumJ2, divide );
#ifdef DEBUG
			// for debugging purposes, print xi
			std::cout << "xi: " << xi << '\n';
#endif

			/* ---- iteration variables ---- */
			IOType ps  = p_init;
			const IOType dps = ( p_end - p_init ) / static_cast<IOType>( num_iters - 1 );
			// assert len of energies == N
			assert( energies.size() == num_iters );

			// TODO: move these to the aggument list of bSB
			// Vec Jx( N ), temp( N ), mask( N );  // workspace vectors
			grb::Vector< IOType, backend > Jx( N ), temp( N );
			grb::Vector< bool, backend > mask( N );

			for ( std::size_t iter = 0; iter < num_iters; ++iter ) {

			    /* y_comp += ((-1+ps)*x_comp + xi*(Jx + h)) * dt */

			    // Jx ← J * x_comp
				rc = rc ? rc : grb::mxv< descr_dense >( Jx, J, x_comp, ring );
				assert( rc == grb::SUCCESS );

			    // temp ← Jx + h
			    rc = rc ? rc : grb::eWiseApply< descr_dense >(
			        temp, Jx, h, ring.getAdditiveMonoid()
			    );
				assert( rc == grb::SUCCESS );

			    // temp ← xi * temp
			    rc = rc ? rc : grb::foldl< descr_dense >( 
					temp, xi, ring.getMultiplicativeMonoid() 
				);
				assert( rc == grb::SUCCESS );

			    // temp ← temp + (-1+ps) * x_comp
			    const IOType scale = -1.0 + ps;
				rc = rc ? rc : grb::eWiseMul< descr_dense >( temp, scale, x_comp, ring );
				assert( rc == grb::SUCCESS );

			    // y_comp += dt * temp
			    rc = rc ? rc : grb::eWiseMul< descr_dense >( y_comp, dt, temp, ring );
				assert( rc == grb::SUCCESS );

			    /* x_comp += dt * y_comp */
			    rc = rc ? rc : grb::eWiseMul< descr_dense >( x_comp, dt, y_comp, ring );
				assert( rc == grb::SUCCESS );

			    /* y_comp[ |x|>1 ] = 0 */
				// mask = np.abs(x_comp) > 1
			    rc = rc ? rc : grb::eWiseLambda< descr_dense >( [&mask, &x_comp]( const size_t i ) {
					(void) i;
					// rewrite this to use graphblas language
					mask[i] = std::abs(x_comp[i]) > 1;
					}, mask 
				);
				assert( rc == grb::SUCCESS );

				// y_comp[ mask ] = 0
			    rc = rc ? rc : grb::eWiseLambda< descr_dense >( [&mask, &y_comp]( const size_t i ) {
					(void) i;
					// rewrite this to use graphblas language
					if(mask[i]) {
						y_comp[i] = 0;
					}
					}, y_comp 
				);
				assert( rc == grb::SUCCESS );

			    /* x_comp = clip( x_comp ) */
				rc = rc ? rc : grb::eWiseLambda< descr_dense >( [&x_comp]( const size_t i ) {
					(void) i;
					x_comp[i] = clip11<IOType>()( x_comp[i] );	}, x_comp 
				);
				assert( rc == grb::SUCCESS );

			//     /* Energy evaluation */
			//     Vec sol( N );
			//     GRB_TRY( apply( sol, x_comp, signum() ) );

			//     // temp ← J * sol
			//     GRB_TRY( mxv( temp, J, sol ) );

			//     // e = -0.5 * sol^T * temp  –  h^T * sol
			//     IOType dot1 = 0.0, dot2 = 0.0;
			//     GRB_TRY( foldl( dot1, sol, temp, operators::add< IOType >(),
			//                     operators::mul< IOType >() ) );
			//     GRB_TRY( foldl( dot2, h, sol, operators::add< IOType >(),
			//                     operators::mul< IOType >() ) );

					energies[ iter ] = iter; // just for testing purposes 
			//     energies[ iter ] = -0.5 * dot1 - dot2;

			    ps += dps;
			}

			return SUCCESS;
		}

	} // algorithms namespace

} // grb namespace

#endif // end _H_GRB_ALGORITHMS_ISING_MACHINE_SB

