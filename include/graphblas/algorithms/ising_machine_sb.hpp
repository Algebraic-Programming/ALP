
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
#include <vector>

#include <graphblas/algorithms/norm.hpp>

namespace grb {
	namespace algorithms {

		template< typename IOType, Backend backend >
		void vector_print( const grb::Vector< IOType, backend > & v, const std::string & vector_name ) {
			grb::PinnedVector< IOType > pinnedVector;
			pinnedVector = grb::PinnedVector< IOType >( v, grb::SEQUENTIAL );
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

		/*
		 * This function minimizes -xJx/2-h
		 */
		template< Descriptor descr = descriptors::no_operation,
			bool DISCRETIZEJX, // if true, the method is dSB
			typename IsingHType,
			typename IOType,
			typename solType,
			typename RSI,
			typename NZI,
			Backend backend,
			class Ring = Semiring<
				grb::operators::add< IOType >,
				grb::operators::mul< IOType >,
				grb::identities::zero,
				grb::identities::one
			>,
			class Divide = operators::divide< IOType >,
			class RingIType = Semiring<
				grb::operators::add< IsingHType >,
				grb::operators::mul< IsingHType >,
				grb::identities::zero,
				grb::identities::one
			>
		>
		grb::RC SB( std::vector< IOType > & energies,                   // output length num_iters
			grb::Vector< IOType, backend > & x_comp,                     // in/out, size N
			grb::Vector< IOType, backend > & y_comp,                     // in/out, size N
			const grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J, // NxN, symmetric
			const grb::Vector< IsingHType, backend > & h,                    // size N
			const IOType p_init,
			const IOType p_end,
			const std::size_t num_iters,
			const IOType dt,
			// workspace
			grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J2,
    		grb::Vector< IOType, backend > & temp,
			grb::Vector< IsingHType, backend > & temp_int,
			grb::Vector< solType, backend > & sol,
			size_t & iterations,
			// Parameters present in
			// Goto et al., “High-performance combinatorial optimization based on classical mechanics"
			const IOType a0 = 1,
			// default semiring, divide
			const Ring & ring = Ring(),
			const Divide & divide = Divide(),
			const IOType zero = 0,
			const RingIType & ringIType = RingIType(),
			const IsingHType zero_itype = 0,
			const std::function< IOType( IOType ) > & sqrtX = std_sqrt< IOType, IOType > ) {

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

			grb::set( sol, static_cast<solType>(0) );

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
				grb::apply( v, v, v, ring.getMultiplicativeOperator() );
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
			IOType c0 = 0.5;
			IOType sqrt_sumJ2 = zero_itype;
			sqrt_sumJ2 = sqrtX( static_cast<IOType>( sumJ2 ) );
			rc = rc ? rc : grb::foldl< descr_dense >( c0, sqrt_sumJ2, divide );
#ifdef DEBUG_IMSB
			// for debugging purposes, print c0
			std::cout << "c0: " << c0 << '\n';
#endif
			if( DISCRETIZEJX ){
				// sol[i] = sign(x_comp[i]); which in graphblas is:
				rc = rc ? rc : grb::eWiseLambda< descr_dense >(
					[&sol,&x_comp]( const size_t i ) {
						(void) i;
						sol[i] = sign<IOType, IsingHType>(x_comp[i]);
					},
					sol, x_comp
				);
			}

			/* ---- iteration variables ---- */
			IOType ps  = p_init;
			const IOType dps = ( p_end - p_init ) / static_cast<IOType>( num_iters - 1 );
			// assert len of energies == N
			assert( energies.size() == num_iters );

			for ( iterations = 0; iterations < num_iters; ++iterations ) {

			    /* y_comp += (dt*(-a0+ps)*x_comp) + (dt*c0*(Jx + h)) */

			    // Jx <- J * x_comp
				rc = rc ? rc : grb::set( temp, zero );
				if( DISCRETIZEJX ){
					// this makes the method dSB !
					rc = rc ? rc : grb::mxv< descr_dense >( temp, J, sol, ring );
				}else{
					rc = rc ? rc : grb::mxv< descr_dense >( temp, J, x_comp, ring );
				}

				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp, "Jx" );
#endif
			    // y_comp <- y_comp + dt * c0 * temp
			    rc = rc ? rc : grb::eWiseMul< descr_dense >(
					y_comp, dt * c0, temp, ring
				);
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp, "c0 * temp" );
#endif
			    // y_comp <- y_comp + dt * (-a0+ps) * x_comp
			    const IOType scale = -a0 + ps;
				rc = rc ? rc : grb::eWiseMul< descr_dense >( y_comp, dt * scale, x_comp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "scale: " << scale << '\n';
				vector_print( y_comp, "y_comp + (-a0+ps) * x_comp" );
#endif

				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( y_comp, "y_comp (a)" );
#endif
				// ----------------------------------------------------

			    /* x_comp += a0 * dt * y_comp */
			    rc = rc ? rc : grb::eWiseMul< descr_dense >( x_comp, a0 * dt, y_comp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( x_comp, "x_comp" );
#endif

			    /* y_comp[ |x|>1 ] = 0 */
			    rc = rc ? rc : grb::eWiseLambda< descr_dense >( [&y_comp, &x_comp]( const size_t i ) {
					(void) i;
					// TODO: rewrite this to use graphblas language
					y_comp[i] = (std::abs(x_comp[i]) > 1) ? 0 : y_comp[i];
					}, y_comp, x_comp
				);
				assert( rc == grb::SUCCESS );
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
						sol[i] = sign<IOType, solType>( x_comp[i] );
					},
					sol, x_comp
				);

				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( sol, "sol" );
#endif
			    // temp <- J * sol
				rc = rc ? rc : grb::set( temp, zero );
				rc = rc ? rc : grb::mxv< descr_dense >( temp, J, sol, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				vector_print( temp_int, "temp = J * sol" );
#endif
			    // e = -0.5 * sol.dot(temp)   –  h.dot(sol)
			    IsingHType dot1 = 0;
				IOType dot2 = 0;
				rc = rc ? rc : grb::dot< descr_dense >( dot1, sol, temp, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "dot1: " << dot1 << '\n';
#endif

				rc = rc ? rc : grb::dot< descr_dense >( dot2, h, sol, ring );
				assert( rc == grb::SUCCESS );
#ifdef DEBUG_IMSB
				std::cout << "dot2: " << dot2 << '\n';
#endif

				const IOType e = -0.5 * dot1 - dot2;
#ifdef DEBUG_IMSB
				std::cout << "e: " << e << '\n';
#endif
				energies[ iterations ] = e;
			    ps += dps;
			}

			return SUCCESS;
		}

		/*
		 * Wrapper for SB that specializes the function to the discrete Simulated Bifurcation algorithm
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IsingHType,
			typename IOType,
			typename solType,
			typename RSI,
			typename NZI,
			Backend backend,
			class Ring = Semiring<
				grb::operators::add< IOType >,
				grb::operators::mul< IOType >,
				grb::identities::zero,
				grb::identities::one
			>,
			class Divide = operators::divide< IOType >,
			class RingIType = Semiring<
				grb::operators::add< IsingHType >,
				grb::operators::mul< IsingHType >,
				grb::identities::zero,
				grb::identities::one
			>
		>
		inline constexpr grb::RC dSB( std::vector< IOType > & energies,                   // output length num_iters
			grb::Vector< IOType, backend > & x_comp,                     // in/out, size N
			grb::Vector< IOType, backend > & y_comp,                     // in/out, size N
			const grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J, // NxN, symmetric
			const grb::Vector< IsingHType, backend > & h,                    // size N
			const IOType p_init,
			const IOType p_end,
			const std::size_t num_iters,
			const IOType dt,
			// workspace
			grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J2,
			grb::Vector< IOType, backend > & temp,
			grb::Vector< IsingHType, backend > & temp_int,
			grb::Vector< solType, backend > & sol,
			size_t & iterations,
			const IOType a0 = 1,
			// default semiring, divide
			const Ring & ring = Ring(),
			const Divide & divide = Divide(),
			const IOType zero = 0,
			const RingIType & ringIType = RingIType(),
			const IsingHType zero_itype = 0,
			const std::function< IOType( IOType ) > & sqrtX = std_sqrt< IOType, IOType > ) {
				return SB< descr, true >( energies, x_comp, y_comp, J, h, p_init, p_end, num_iters, dt,
						J2, temp, temp_int, sol, iterations,
						a0, ring, divide, zero, ringIType, zero_itype, sqrtX);
			}

		/*
		 * Wrapper for SB that specializes the function to the ballistic Simulated Bifurcation algorithm
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename IsingHType,
			typename IOType,
			typename solType,
			typename RSI,
			typename NZI,
			Backend backend,
			class Ring = Semiring<
				grb::operators::add< IOType >,
				grb::operators::mul< IOType >,
				grb::identities::zero,
				grb::identities::one
			>,
			class Divide = operators::divide< IOType >,
			class RingIType = Semiring<
				grb::operators::add< IsingHType >,
				grb::operators::mul< IsingHType >,
				grb::identities::zero,
				grb::identities::one
			>
		>
		inline constexpr grb::RC bSB( std::vector< IOType > & energies,
			grb::Vector< IOType, backend > & x_comp,
			grb::Vector< IOType, backend > & y_comp,
			const grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J,
			const grb::Vector< IsingHType, backend > & h,
			const IOType p_init,
			const IOType p_end,
			const std::size_t num_iters,
			const IOType dt,
			// workspace
			grb::Matrix< IsingHType, backend, RSI, RSI, NZI > & J2,
			grb::Vector< IOType, backend > & temp,
			grb::Vector< IsingHType, backend > & temp_int,
			grb::Vector< solType, backend > & sol,
			size_t & iterations,
			const IOType a0 = 1,
			// default semiring, divide
			const Ring & ring = Ring(),
			const Divide & divide = Divide(),
			const IOType zero = 0,
			const RingIType & ringIType = RingIType(),
			const IsingHType zero_itype = 0,
			const std::function< IOType( IOType ) > & sqrtX = std_sqrt< IOType, IOType > ) {
				return SB< descr, false >( energies, x_comp, y_comp, J, h, p_init, p_end, num_iters, dt,
						J2, temp, temp_int, sol, iterations,
						a0, ring, divide, zero, ringIType, zero_itype, sqrtX);
			}


	} // algorithms namespace

} // grb namespace

#endif // end _H_GRB_ALGORITHMS_ISING_MACHINE_SB

