
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


namespace grb {
	namespace algorithms {
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
			template< typename IOType, typename NonzeroType, class Ring >
			grb::RC __rbgs_single_step( const grb::Matrix< NonzeroType > & A,
				const grb::Vector< IOType > & A_diagonal,
				const grb::Vector< IOType > & r,
				grb::Vector< IOType > & x,
				grb::Vector< IOType > & smoother_temp,
				const grb::Vector< bool > & color_mask,
				const Ring & ring ) {
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
									if( color_mask[ i ] ) {
										IOType d = A_diagonal[ i ];
										IOType v = r[ i ] - smoother_temp[ i ] + x[ i ] * d;
										x[ i ] = v / d;
									}
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
			template< typename IOType, typename NonzeroType, class Ring >
			grb::RC red_black_gauss_seidel( system_data< IOType, NonzeroType > & data, const Ring & ring ) {
				RC ret { SUCCESS };
				// forward step
				std::vector< grb::Vector< bool > >::const_iterator end { data.color_masks.cend() };
				for( std::vector< grb::Vector< bool > >::const_iterator it { data.color_masks.cbegin() }; it != end && ret == SUCCESS; ++it ) {
					ret = ret ? ret : __rbgs_single_step( data.A, data.A_diagonal, data.r, data.z, data.smoother_temp, *it, ring );
				}
				// backward step
				std::vector< grb::Vector< bool > >::const_reverse_iterator rend { data.color_masks.crend() };
				for( std::vector< grb::Vector< bool > >::const_reverse_iterator rit { data.color_masks.crbegin() }; rit != rend && ret == SUCCESS; ++rit ) {
					ret = ret ? ret : __rbgs_single_step( data.A, data.A_diagonal, data.r, data.z, data.smoother_temp, *rit, ring );
				}
				return ret;
			}

		} // namespace internal
	}     // namespace algorithms
} // namespace grb

#endif // H_GRB_ALGORITHMS_RED_BLACK_GAUSS_SEIDEL
