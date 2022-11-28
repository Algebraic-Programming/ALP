
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @file red_black_gauss_seidel.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Contains the routines to perform a forward-backward pass of a Red-Black Gauss-Seidel smoother.
 */

#ifndef _H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL
#define _H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL

#include <cassert>

#include <graphblas.hpp>

#include "multigrid_data.hpp"

namespace grb {
	namespace algorithms {

		/**
		 * Data structures to run the RBGS smoother on a single level of the multi-grid.
		 */
		template< typename IOType > struct SmootherData {

			grb::Vector< IOType > A_diagonal; ///< vector with the diagonal of #A
			grb::Vector< IOType > smoother_temp; ///< for smoother's intermediate results
			std::vector< grb::Vector< bool > > color_masks; ///< for color masks

			/**
			 * Construct a new SmootherData object from the level size.
			 */
			SmootherData( size_t sys_size ) :
				A_diagonal( sys_size ),
				smoother_temp( sys_size ) {}

			// for safety, disable copy semantics
			SmootherData( const SmootherData & o ) = delete;

			SmootherData & operator=( const SmootherData & ) = delete;

			grb::RC init_vectors( IOType zero ) {
				return grb::set( smoother_temp, zero );
			}
		};

		namespace internal {

			/**
			 * Runs a single step of Red-Black Gauss-Seidel for a specific color.
			 *
			 * @tparam descr descriptor for static information
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[in] A the system matrix
			 * @param[in] A_diagonal a vector storing the diagonal elements of \p A
			 * @param[in] r the residual
			 * @param[in,out] x the initial solution to start from, and where the smoothed solution is stored to
			 * @param[out] smoother_temp a vector for temporary values
			 * @param[in] color_mask the mask of colors to filter the rows to smooth
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *  unsuccessful operation otherwise
			 */
			template<
				Descriptor descr,
				typename IOType,
				typename NonzeroType,
				class Ring
			> grb::RC rbgs_single_step(
				const grb::Matrix< NonzeroType > & A,
				const grb::Vector< IOType > & A_diagonal,
				const grb::Vector< IOType > & r,
				grb::Vector< IOType > & x,
				grb::Vector< IOType > & smoother_temp,
				const grb::Vector< bool > & color_mask,
				const Ring & ring
			) {
				RC ret = SUCCESS;

				// smoother_temp[color_mask] = A[color_mask] * x[color_mask]
				// use the structural descriptors, assuming ONLY the values of the current color are set
				// note that if this assumption does not hold, also the following eWiseLambda() is wrong
				ret = ret ? ret : grb::mxv< grb::descriptors::safe_overlap | grb::descriptors::structural >(
					smoother_temp, color_mask, A, x, ring );
				assert( ret == SUCCESS );

				// TODO internal issue #201
				// Replace below with masked calls:
				// x[mask] = r[mask] - smoother_temp[mask] + x[mask] .* diagonal[mask]
				// x[mask] = x[maks] ./ diagonal[mask]
				ret = ret ? ret :
					grb::eWiseLambda(
						[ &x, &r, &smoother_temp, &color_mask, &A_diagonal ]( const size_t i ) {
							// if the mask was properly initialized, the check on the mask value is unnecessary;
							// if( color_mask[ i ] ) {
							IOType d = A_diagonal[ i ];
							IOType v = r[ i ] - smoother_temp[ i ] + x[ i ] * d;
							x[ i ] = v / d;
							// }
						},
						color_mask, x, r, smoother_temp, A_diagonal );
				assert( ret == SUCCESS );
				return ret;
			}

			/**
			 * Runs a single forward and backward pass of Red-Black Gauss-Seidel smoothing
			 * on the system stored in \p data.
			 *
			 * This routine performs a forward and a backward step of Red-Black Gauss-Seidel for each color
			 * stored in \p data.color_masks. Colors stored inside this container
			 * <b>are assumed to be mutually exclusive and to cover all rows of the solution vector<\b>,
			 * and no check is performed to ensure these assumptions hold. Hence, it is up to user logic
			 * to pass correct coloring information. Otherwise, \b no guarantees hold on the result.
			 *
			 * @tparam descr descriptor for static information
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[in,out] data structure with the data of a single grid level
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template<
				Descriptor descr,
				typename IOType,
				typename NonzeroType,
				class Ring
			> grb::RC red_black_gauss_seidel(
				MultiGridData< IOType, NonzeroType > &data,
				SmootherData< IOType > &smoothing_info,
				const Ring & ring
			) {
				RC ret = SUCCESS;
				// zero the temp output just once, assuming proper masking avoids
				// interference among different colors
				ret = ret ? ret : grb::set< descr >( smoothing_info.smoother_temp,
					ring. template getZero< IOType >() );

				// forward step
				using cit_t = typename std::vector< grb::Vector< bool > >::const_iterator;
				cit_t end = smoothing_info.color_masks.cend();
				for( cit_t it = smoothing_info.color_masks.cbegin(); it != end && ret == SUCCESS; ++it ) {
					ret = rbgs_single_step< descr >( data.A, smoothing_info.A_diagonal, data.r,
						data.z, smoothing_info.smoother_temp, *it, ring );
				}
				ret = ret ? ret : grb::set< descr >( smoothing_info.smoother_temp,
					ring. template getZero< IOType >() );

				// backward step
				using crit_t = typename std::vector< grb::Vector< bool > >::const_reverse_iterator;
				crit_t rend = smoothing_info.color_masks.crend();
				for( crit_t rit = smoothing_info.color_masks.crbegin(); rit != rend && ret == SUCCESS; ++rit ) {
					ret = rbgs_single_step< descr >( data.A, smoothing_info.A_diagonal, data.r,
						data.z, smoothing_info.smoother_temp, *rit, ring );
				}
				return ret;
			}

		} // namespace internal

		/**
		 * Runner object for the RBGS smoother, with multiple methods for each type of smoothing step:
		 * pre-, post- and non-recursive, as invoked during a full run of a multi-grid V-cycle.
		 *
		 * It stores the information to smooth each level of the grid, to be initalized separately.
		 *
		 * @tparam IOType type of result and intermediate vectors used during computation
		 * @tparam NonzeroType type of matrix values
		 * @tparam Ring the ring of algebraic operators
		 * @tparam descr descriptors with statically-known data for computation and containers
		 */
		template <
			typename IOType,
			typename NonzeroType,
			class Ring,
			Descriptor descr = descriptors::no_operation
		> struct RedBlackGSSmootherRunner {

			size_t presmoother_steps; ///< number of pre-smoother steps
			size_t postsmoother_steps;  ///< number of post-smoother steps
			size_t non_recursive_smooth_steps;  ///< number of smoother steps for the last grid level
			std::vector< std::unique_ptr< SmootherData< IOType > > > levels;  ///< for each grid level,
				///< the smoothing data (finest first)
			Ring ring;  ///< the algebraic ring

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring operator with default values" );

			using SmootherInputType = MultiGridData< IOType, NonzeroType >;

			inline grb::RC pre_smooth( SmootherInputType& data ) {
				return __run_smoother( data, presmoother_steps );
			}

			inline grb::RC post_smooth( SmootherInputType& data ) {
				return __run_smoother( data, postsmoother_steps );
			}

			inline grb::RC nonrecursive_smooth( SmootherInputType& data ) {
				return __run_smoother( data, non_recursive_smooth_steps );
			}

			/**
			 * Runs \p smoother_steps iteration of the Red-Black Gauss-Seidel smoother,
			 * with inputs and outputs stored inside \p data.
			 *
			 * This is an internal method called by all user-facing methods, because this specific
			 * smoother performs all smoothing steps the same way.
			 */
			grb::RC __run_smoother(
				SmootherInputType &data,
				const size_t smoother_steps
			) {
				RC ret = SUCCESS;

				SmootherData< IOType > &smoothing_info = *( levels.at( data.level ).get() );

				for( size_t i = 0; i < smoother_steps && ret == SUCCESS; i++ ) {
					ret = ret ? ret : internal::red_black_gauss_seidel< descr >(
						data, smoothing_info, ring );
					assert( ret == SUCCESS );
				}
				return ret;
			}
		};

	}     // namespace algorithms
} // namespace grb

#endif // H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL
