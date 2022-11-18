
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
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
 * @brief Contains the routines to perform a forward-backward pass of a Red-Black Gauss-Seidel smoother.
 * @date 2021-04-30
 */

#ifndef _H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL
#define _H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL

#include <cassert>

#include <graphblas.hpp>

#include "multigrid_data.hpp"

namespace grb {
	namespace algorithms {

		template< typename IOType > struct smoother_data {

			grb::Vector< IOType > A_diagonal;               ///< vector with the diagonal of #A
			grb::Vector< IOType > smoother_temp;            ///< for smoother's intermediate results
			std::vector< grb::Vector< bool > > color_masks; ///< for color masks

			smoother_data( size_t sys_size ) :
				A_diagonal( sys_size ),
				smoother_temp( sys_size ) { }

			// for safety, disable copy semantics
			smoother_data( const smoother_data & o ) = delete;

			smoother_data & operator=( const smoother_data & ) = delete;

			grb::RC zero_temp_vectors() {
				return grb::set( smoother_temp, 0 );
			}
		};

		namespace internal {

			/**
			 * @brief Runs a single step of Red-Black Gauss-Seidel for a specific color.
			 *
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
			 *                          unsuccessful operation otherwise
			 */
			template<
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
				RC ret { SUCCESS };
				ret = ret ? ret : grb::set( smoother_temp, 0 );

				// acc_temp[mask] = A[mask] * x[mask]
				ret = ret ? ret : grb::mxv< grb::descriptors::safe_overlap >( smoother_temp, color_mask, A, x, ring );
				assert( ret == SUCCESS );

				// TODO internal issue #201
				// Replace below with masked calls:
				// x[mask] = r[mask] - smoother_temp[mask] + x[mask] .* diagonal[mask]
				// x[mask] = x[maks] ./ diagonal[mask]
				ret = ret ? ret :
                            grb::eWiseLambda(
								[ &x, &r, &smoother_temp, &color_mask, &A_diagonal ]( const size_t i ) {
									// if the mask was properly initialized, the check on the mask value is unnecessary;
					                // nonetheless, it is left not to violate the semantics of RBGS in case also the false values
					                // had been initialized (in which case the check is fundamental); if only true values were initialized,
					                // we expect CPU branch prediction to neutralize the branch cost
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
			 * @brief Runs a single forward and backward pass of Red-Black Gauss-Seidel smoothing on the system stored in \p data.
			 *
			 * This routine performs a forward and a backward step of Red-Black Gauss-Seidel for each color stored in \p data.color_masks.
			 * Color stored inside this container <b>are assumed to be mutually exclusive and to cover all rows of the solution vector<\b>,
			 * and no check is performed to ensure these assumptions hold. Hence, it is up to user logic to generate and pass correct
			 * coloring information. Otherwise, \b no guarantees hold on the result.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param data \ref system_data data structure with relevant inpus and outputs: system matrix, initial solution,
			 *             residual, system matrix colors, temporary vectors
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			template<
				typename IOType,
				typename NonzeroType,
				class Ring
			> grb::RC red_black_gauss_seidel(
				multigrid_data< IOType, NonzeroType > &data,
				smoother_data< IOType > &smoothing_info,
				const Ring & ring
			) {
				RC ret { SUCCESS };
				// forward step
				std::vector< grb::Vector< bool > >::const_iterator end { smoothing_info.color_masks.cend() };
				for( std::vector< grb::Vector< bool > >::const_iterator it {
					smoothing_info.color_masks.cbegin() }; it != end && ret == SUCCESS; ++it ) {
					ret = rbgs_single_step( data.A, smoothing_info.A_diagonal, data.r, data.z,
						smoothing_info.smoother_temp, *it, ring );
				}
				// backward step
				std::vector< grb::Vector< bool > >::const_reverse_iterator rend { smoothing_info.color_masks.crend() };
				for( std::vector< grb::Vector< bool > >::const_reverse_iterator rit {
					smoothing_info.color_masks.crbegin() }; rit != rend && ret == SUCCESS; ++rit ) {
					ret = rbgs_single_step( data.A, smoothing_info.A_diagonal, data.r, data.z,
						smoothing_info.smoother_temp, *rit, ring );
				}
				return ret;
			}

		} // namespace internal

		template <
			typename IOType,
			typename NonzeroType,
			class Ring
		> struct red_black_smoother_runner {
			size_t presmoother_steps ;
			size_t postsmoother_steps;
			size_t non_recursive_smooth_steps;
			std::vector< std::unique_ptr< smoother_data< IOType > > > levels;
			Ring ring;

			static_assert( std::is_default_constructible< Ring >::value,
				"cannot construct the Ring operator with default values" );

			using SmootherInputType = multigrid_data< IOType, NonzeroType >;

			inline grb::RC pre_smooth(
				SmootherInputType& data
			) {
				return run_smoother( data, presmoother_steps );
			}

			inline grb::RC post_smooth(
				SmootherInputType& data
			) {
				return run_smoother( data, postsmoother_steps );
			}

			inline grb::RC nonrecursive_smooth(
				SmootherInputType& data
			) {
				return run_smoother( data, non_recursive_smooth_steps );
			}

			/**
			 * @brief Runs \p smoother_steps iteration of the Red-Black Gauss-Seidel smoother, with inputs and outputs stored
			 * inside \p data.
			 *
			 * @tparam IOType type of result and intermediate vectors used during computation
			 * @tparam NonzeroType type of matrix values
			 * @tparam Ring the ring of algebraic operators zero-values
			 *
			 * @param[in,out] data \ref system_data data structure with relevant inpus and outputs: system matrix, initial solution,
			 *                     residual, system matrix colors, temporary vectors
			 * @param[in] smoother_steps how many smoothing steps to run
			 * @param[in] ring the ring to perform the operations on
			 * @return grb::RC::SUCCESS if the algorithm could correctly terminate, the error code of the first
			 *                          unsuccessful operation otherwise
			 */
			grb::RC run_smoother(
				SmootherInputType &data,
				const size_t smoother_steps
			) {
				RC ret { SUCCESS };

				smoother_data< IOType > &smoothing_info = *( levels.at( data.level ).get() );

				for( size_t i { 0 }; i < smoother_steps && ret == SUCCESS; i++ ) {
					ret = ret ? ret : internal::red_black_gauss_seidel( data, smoothing_info, ring );
					assert( ret == SUCCESS );
				}
				return ret;
			}
		};

	}     // namespace algorithms
} // namespace grb

#endif // H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL
