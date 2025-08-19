
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

		template< typename IOType, Backend backend >
		void vector_print( grb::Vector< IOType, backend > & x_comp, const std::string & vector_name ) {
			grb::PinnedVector< IOType > pinnedVector;
			pinnedVector = grb::PinnedVector< IOType >( x_comp, grb::SEQUENTIAL );
			std::cout << "First 10 nonzeroes of " << vector_name << " = [ ";
			for( size_t k = 0; k < pinnedVector.nonzeroes() && k < 10; ++k ) {
				const IOType & nonzeroValue = pinnedVector.getNonzeroValue( k );
				std::cout << nonzeroValue;
				if(k!=pinnedVector.nonzeroes()-1) std::cout << ", ";
			}
			std::cout << "]" << std::endl;
		}


		// // Custom unary operator for sign extraction
		// struct signum {
		// 	constexpr IOType operator()( const IOType x ) const noexcept {
		// 		return ( x > 0 ) - ( x < 0 );
		// 	}
		// };
		template< typename IType, typename ReturnType >
		inline ReturnType sign(IType x) {
			return (x > 0) - (x < 0);
		}

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
		template< Descriptor descr = descriptors::no_operation,
			typename IsingHType,
			typename IOType,
			typename RSI,
			typename NZI,
			Backend backend,
			class Ring = Semiring< 
				grb::operators::add< IOType >, 
				grb::operators::mul< IOType >, 
				grb::identities::zero, 
				grb::identities::one
			>,
			class Minus = operators::subtract< IOType >,
			class Divide = operators::divide< IOType >,
			class RingIType = Semiring< 
				grb::operators::add< IsingHType >, 
				grb::operators::mul< IsingHType >, 
				grb::identities::zero, 
				grb::identities::one
			>
		>
		grb::RC bSB( std::vector< IOType > & energies,                   // output length num_iters
			grb::Vector< IOType, backend > & x_comp,                     // in/out, size N
			grb::Vector< IOType, backend > & y_comp,                     // in/out, size N
			const grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J, // NxN, symmetric
			// TODO: make h const
			grb::Vector< IsingHType, backend > & h,                    // size N
			const IOType p_init,
			const IOType p_end,
			const std::size_t num_iters,
			const IOType dt,
			// workspace
			grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J2,
			grb::Vector< IOType, backend > & Jx,
    		grb::Vector< IOType, backend > & temp,
			grb::Vector< IsingHType, backend > & temp_int,
			grb::Vector< bool, backend > & mask,
			grb::Vector< IsingHType, backend > & sol,
			// default semiring, minus, divide
			const Ring & ring = Ring(),
			const Minus & minus = Minus(),
			const Divide & divide = Divide(),
			const IOType zero = 0,
			const RingIType & ringIType = RingIType(),
			const IsingHType zero_itype = 0,
			const std::function< IOType( IOType ) > & sqrtX = std_sqrt< IOType, IOType > ) {
			(void)minus; // suppress unused parameter warning

			constexpr const Descriptor descr_dense = descr | descriptors::dense;

			const std::size_t N = grb::nrows(J);

			assert( grb::ncols(J) == N );
			assert( grb::size(h) == N );
			assert( grb::size(x_comp) == N );
			assert( grb::size(y_comp) == N );

			assert( grb::capacity(J) == grb::capacity(J2) );
			assert( grb::ncols(J) == grb::ncols(J2) );
			assert( grb::nrows(J) == grb::nrows(J2) );
			// TODO: check that J is symmetric once properly implemented
			//assert( grb::is_symmetric(J) );

			grb::set( sol, zero_itype );
			grb::set( mask, ring.template getZero< bool >() );

			// print pinned vector x_comp
			// for debugging purposes, print x_comp
#ifdef DEBUG_IMSB
			vector_print( x_comp, "x_comp" );
			vector_print( y_comp, "y_comp" );
			vector_print( h, "h" );
#endif

			// assert that energies is of length num_iters
			assert( energies.size() == num_iters );

			if ( num_iters < 1 ) {
				return grb::SUCCESS;
			}

			grb::RC rc = grb::SUCCESS;

			/* ---- pre-compute ---- */
			rc = rc ? rc : grb::set( J2, J );
			assert( rc == grb::SUCCESS );
			rc = rc ? rc : grb::eWiseLambda( [&ring]( const size_t i, const size_t j, IsingHType& v ) {
				(void) i;
				(void) j;
				apply( v, v, v, ring.getMultiplicativeOperator() );
			}, J2 );
			assert( rc == grb::SUCCESS );

			IsingHType sumJ2 = zero_itype;
			rc = rc ? rc : grb::foldl( sumJ2, J2, ringIType.getAdditiveMonoid() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error in eWiseLambda for sumJ2: " << rc << '\n';
				return rc;
			}
#ifdef DEBUG_IMSB
			// for debugging purposes, print sumJ2
			std::cout << "sumJ2: " << sumJ2 << '\n';
#endif

			rc = rc ? rc : grb::foldl< descr_dense >( sumJ2, static_cast<IOType>( N - 1 ), divide );
			IOType xi = 0.5;
			IOType sqrt_sumJ2 = zero_itype;
			sqrt_sumJ2 = sqrtX( static_cast<IOType>( sumJ2 ) );
			rc = rc ? rc : grb::foldl< descr_dense >( xi, sqrt_sumJ2, divide );
#ifdef DEBUG_IMSB
			// for debugging purposes, print xi
			std::cout << "xi: " << xi << '\n';
#endif

			/* ---- iteration variables ---- */
			IOType ps  = p_init;
			const IOType dps = ( p_end - p_init ) / static_cast<IOType>( num_iters - 1 );
			// assert len of energies == N
			assert( energies.size() == num_iters );

			for ( std::size_t iter = 0; iter < num_iters; ++iter ) {

			    /* y_comp += ((-1+ps)*x_comp + xi*(Jx + h)) * dt */

			    // Jx <- J * x_comp
				grb::set( Jx, zero );
				rc = rc ? rc : grb::mxv< descr_dense >( Jx, J, x_comp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( Jx, "Jx" );
#endif

			    // temp <- Jx + h
				grb::set( temp, zero );
			    rc = rc ? rc : grb::eWiseApply(
			        temp, Jx, h, ring.getAdditiveMonoid()
			    );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp, "temp = Jx + h" );
#endif

			    // temp <- xi * temp
			    rc = rc ? rc : grb::foldl< descr_dense >( 
					temp, xi, ring.getMultiplicativeMonoid() 
				);
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp, "xi * temp" );
#endif
			    // temp <- temp + (-1+ps) * x_comp
			    const IOType scale = -1.0 + ps;
				rc = rc ? rc : grb::eWiseMul< descr_dense >( temp, scale, x_comp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "scale: " << scale << '\n';
				vector_print( temp, "temp + (-1+ps) * x_comp" );
#endif

			    // y_comp += dt * temp
			    rc = rc ? rc : grb::eWiseMul< descr_dense >( y_comp, dt, temp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( y_comp, "y_comp (a)" );
#endif

			    /* x_comp += dt * y_comp */
			    rc = rc ? rc : grb::eWiseMul< descr_dense >( x_comp, dt, y_comp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( x_comp, "x_comp" );
#endif

			    /* y_comp[ |x|>1 ] = 0 */
				// mask = np.abs(x_comp) > 1
			    rc = rc ? rc : grb::eWiseLambda< descr_dense >( [&mask, &x_comp]( const size_t i ) {
					(void) i;
					// rewrite this to use graphblas language
					mask[i] = std::abs(x_comp[i]) > 1;
					}, mask, x_comp
				);
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( mask, "mask" );
#endif
				rc = rc ? rc : grb::foldl< descr_dense >(  
					y_comp, mask, zero, 
					grb::operators::right_assign<IOType>()
				);
#ifdef DEBUG_IMSB
				vector_print( y_comp, "y_comp (b)" );
#endif
				assert( rc == grb::SUCCESS );

			    /* x_comp = clip( x_comp ) */
				rc = rc ? rc : foldl< descr_dense >( x_comp, static_cast<IOType>(-1), grb::operators::max < IOType >() );
				rc = rc ? rc : foldl< descr_dense >( x_comp, static_cast<IOType>(1), grb::operators::min < IOType >() );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "i =  " << iter << "\n ";
				vector_print( x_comp, "x_comp_alp " );
				vector_print( y_comp, "y_comp_alp" );
#endif

			    /* Energy evaluation */
				// sol[i] = sign(x_comp[i]); which in graphblas is:
				rc = rc ? rc : grb::eWiseLambda< descr_dense >( 
					[&sol,&x_comp]( const size_t i ) {
						(void) i;
						sol[i] = sign<IOType, IsingHType>(x_comp[i]);
					}, 
					sol, x_comp
				);
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( sol, "sol" );
#endif
			    // temp <- J * sol
				rc = rc ? rc : grb::set( temp_int, zero_itype );
				rc = rc ? rc : grb::mxv< descr_dense >( temp_int, J, sol, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp_int, "temp = J * sol" );
#endif
			    // e = -0.5 * sol.dot(temp)   –  h.dot(sol)
			    IsingHType dot1 = 0;
				IOType dot2 = 0;
				rc = rc ? rc : grb::dot< descr_dense >( dot1, sol, temp_int, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "dot1: " << dot1 << '\n';
#endif

				rc = rc ? rc : grb::dot< descr_dense >( dot2, h, sol, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "dot2: " << dot2 << '\n';
#endif

				IOType e = -0.5 * dot1 - dot2;
#ifdef DEBUG_IMSB
				std::cout << "e: " << e << '\n';
#endif
				energies[ iter ] = e;
			    ps += dps;
			}

			return SUCCESS;
		}

	} // algorithms namespace

} // grb namespace

#endif // end _H_GRB_ALGORITHMS_ISING_MACHINE_SB

