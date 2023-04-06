
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
 * @file
 *
 * Implements k-means. The state of the algorithms defined within are
 * \em experimental.
 *
 * @author Verner Vlacic
 */

#ifndef _H_GRB_KMEANS
#define _H_GRB_KMEANS

#include <chrono>
#include <random>

#include <assert.h>

#include <graphblas.hpp>


namespace grb {

	namespace algorithms {

		/**
		 * a simple implementation of the k++ initialisation algorithm for kmeans
		 *
		 * @param[in,out] K k by m matrix containing the current k means as row vectors
		 * @param[in]     X m by n matrix containing the n points to be classified as
		 *                  column vectors
		 * @param[in] dist_op Coordinatewise distance operator, squared difference by
		 *                    default
		 *
		 * \todo more efficient implementation using Walker's alias method
		 *
		 * \todo expand documentation
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType = double,
			class Operator = operators::square_diff< IOType, IOType, IOType >
		>
		RC kpp_initialisation(
			Matrix< IOType > &K,
			const Matrix< IOType > &X,
			const Operator &dist_op = Operator()
		) {
			// declare monoids and semirings
			Monoid< grb::operators::add< IOType >, grb::identities::zero > add_monoid;
			Monoid<
				grb::operators::min< IOType >,
				grb::identities::infinity
			> min_monoid;
			Semiring<
				grb::operators::add< IOType >,
				grb::operators::right_assign_if< bool, IOType, IOType >,
				grb::identities::zero, grb::identities::logical_true
			> pattern_sum;

			// runtime sanity checks: the row dimension of X should match the column
			// dimension of K
			if( ncols( K ) != nrows( X ) ) {
				return MISMATCH;
			}

			// running error code
			RC ret = SUCCESS;

			// get problem dimensions
			const size_t n = ncols( X );
			const size_t m = nrows( X );
			const size_t k = nrows( K );

			// declare vector of indices of columns of X selected as the initial
			// centroids
			Vector< size_t > selected_indices( k );

			// declare column selection vector
			Vector< bool > col_select( n );

			// declare selected point
			Vector< IOType > selected( m );

			// declare vector of distances from the selected point
			Vector< IOType > selected_distances( n );

			// declare vector of minimum distances to all points selected so far
			Vector< IOType > min_distances( n );
			ret = ret ? ret : grb::set( min_distances,
				grb::identities::infinity< IOType >::value()
			);

			// generate first centroid by selecting a column of X uniformly at random

			size_t i;
			{
				const size_t seed_uniform =
					std::chrono::system_clock::now().time_since_epoch().count();
				std::default_random_engine random_generator( seed_uniform );
				std::uniform_int_distribution< size_t > uniform( 0, n - 1 );
				i = uniform( random_generator );
			}

			for( size_t l = 0; ret == SUCCESS && l < k; ++l ) {

				ret = grb::clear( col_select );
				ret = ret ? ret : grb::clear( selected );
				ret = ret ? ret : grb::clear( selected_distances );

				ret = ret ? ret : grb::setElement( selected_indices, i, l );

				ret = ret ? ret : grb::setElement( col_select, true, i );

				ret = ret ? ret : grb::vxm< grb::descriptors::transpose_matrix >(
					selected, col_select, X, pattern_sum );

				ret = ret ? ret : grb::vxm( selected_distances, selected, X, add_monoid,
					dist_op );

				ret = ret ? ret : grb::foldl( min_distances, selected_distances,
					min_monoid );

				// TODO the remaining part of the loop should be replaced with the alias
				//      algorithm

				IOType range = add_monoid.template getIdentity< IOType >();
				ret = ret ? ret : grb::foldl( range, min_distances, add_monoid );

				double sample = -1;
				if( ret == SUCCESS ) {
					{
						const size_t seed =
							std::chrono::system_clock::now().time_since_epoch().count();
						std::default_random_engine generator( seed );
						std::uniform_real_distribution< double > uniform( 0, 1 );
						sample = uniform( generator );
					}
					ret = grb::collectives<>::broadcast( sample, 0 );
				}
				assert( sample >= 0 );

				// The following is not standard ALP/GraphBLAS and does not work for P>1
				//    (TODO internal issue #320)
				if( ret == SUCCESS ) {
					assert( grb::spmd<>::nprocs() == 1 );
					IOType * const raw = internal::getRaw( selected_distances );
					IOType running_sum = 0;
					i = 0;
					do {
						running_sum += static_cast< double >( raw[ i ] ) / range;
					} while( running_sum < sample && ++i < n );
					i = ( i == n ) ? n - 1 : i;
				}
			}

			// create the matrix K by selecting the columns of X indexed by
			// selected_indices

			// declare pattern matrix
			Matrix< void > M( k, n );
			ret = ret ? ret : grb::resize( M, n );

			if( ret == SUCCESS ) {
				auto converter = grb::utils::makeVectorToMatrixConverter< void, size_t >(
					selected_indices, [](const size_t &ind, const size_t &val ) {
						return std::make_pair( ind, val );
					}
				);
				ret = grb::buildMatrixUnique( M, converter.begin(), converter.end(),
					PARALLEL );
			}

			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X,
				pattern_sum, RESIZE );
			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X,
				pattern_sum );

			if( ret != SUCCESS ) {
				std::cout << "\tkpp finished with unexpected return code!" << std::endl;
			}

			return ret;
		}

		/**
		 * The kmeans iteration given an initialisation
		 *
		 * @param[in,out] K k by m matrix containing the current k means as row
		 *                  vectors
		 * @param[in] clusters_and_distances Vector containing the class and distance
		 *                                   to centroid for each point
		 * @param[in] X m by n matrix containing the n points to be classified as
		 *              column vectors
		 * @param[in] max_iter Maximum number of iterations
		 * @param[in] dist_op Coordinatewise distance operator, squared difference by
		 *                    default
		 *
		 * \internal
		 * \todo expand documentation
		 * \endeinternal
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType = double,
			class Operator = operators::square_diff< IOType, IOType, IOType >
		>
		RC kmeans_iteration(
			Matrix< IOType > &K,
			Vector< std::pair< size_t, IOType > > &clusters_and_distances,
			const Matrix< IOType > &X,
			const size_t max_iter = 1000,
			const Operator &dist_op = Operator()
		) {
			// declare monoids and semirings

			typedef std::pair< size_t, IOType > indexIOType;

			Monoid< grb::operators::add< IOType >, grb::identities::zero > add_monoid;

			Monoid<
				grb::operators::argmin< size_t, IOType >,
				grb::identities::infinity
			> argmin_monoid;

			Monoid<
				grb::operators::logical_and< bool >,
				grb::identities::logical_true
			> comparison_monoid;

			Semiring<
				grb::operators::add< IOType >,
				grb::operators::right_assign_if< bool, IOType, IOType >,
				grb::identities::zero, grb::identities::logical_true
			> pattern_sum;

			Semiring<
				grb::operators::add< size_t >,
				grb::operators::right_assign_if< size_t, size_t, size_t >,
				grb::identities::zero, grb::identities::logical_true
			> pattern_count;

			// runtime sanity checks: the row dimension of X should match the column
			// dimension of K
			if( ncols( K ) != nrows( X ) ) {
				return MISMATCH;
			}
			if( size( clusters_and_distances ) != ncols( X ) ) {
				return MISMATCH;
			}

			// running error code
			RC ret = SUCCESS;

			// get problem dimensions
			const size_t n = ncols( X );
			const size_t m = nrows( X );
			const size_t k = nrows( K );

			// declare distance matrix
			Matrix< IOType > Dist( k, n );

			// declare and initialise labels vector and ones vectors
			Vector< size_t > labels( k );
			Vector< bool > n_ones( n ), m_ones( m );

			ret = ret ? ret : grb::set< grb::descriptors::use_index >( labels, 0 );
			ret = ret ? ret : grb::set( n_ones, true );
			ret = ret ? ret : grb::set( m_ones, true );

			// declare pattern matrix
			Matrix< void > M( k, n );
			ret = ret ? ret : grb::resize( M, n );

			// declare the sizes vector
			Vector< size_t > sizes( k );

			// declare auxiliary vectors and matrices
			Matrix< IOType > K_aux( k, m );
			Matrix< size_t > V_aux( k, m );

			// control variables
			size_t iter = 0;
			Vector< indexIOType > clusters_and_distances_prev( n );
			bool converged;

			do {
				(void) ++iter;

				ret = ret ? ret : grb::set( clusters_and_distances_prev,
					clusters_and_distances );

				ret = ret ? ret : mxm( Dist, K, X, add_monoid, dist_op, RESIZE );
				ret = ret ? ret : mxm( Dist, K, X, add_monoid, dist_op );

				ret = ret ? ret : vxm( clusters_and_distances, labels, Dist, argmin_monoid,
					operators::zip< size_t, IOType >() );

				auto converter = grb::utils::makeVectorToMatrixConverter<
					void, indexIOType
				> (
					clusters_and_distances,
					[]( const size_t &ind, const indexIOType &pair ) {
						return std::make_pair( pair.first, ind );
					}
				);

				ret = ret ? ret : grb::buildMatrixUnique( M, converter.begin(),
					converter.end(), PARALLEL );

				ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K_aux, M, X,
					pattern_sum, RESIZE );
				ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K_aux, M, X,
					pattern_sum );

				ret = ret ? ret : grb::mxv( sizes, M, n_ones, pattern_count );

				ret = ret ? ret : grb::outer( V_aux, sizes, m_ones,
					operators::left_assign_if< IOType, bool, IOType >(), RESIZE );
				ret = ret ? ret : grb::outer( V_aux, sizes, m_ones,
					operators::left_assign_if< IOType, bool, IOType >() );

				ret = ret ? ret : eWiseApply( K, V_aux, K_aux,
					operators::divide_reverse< size_t, IOType, IOType >(), RESIZE );
				ret = ret ? ret : eWiseApply( K, V_aux, K_aux,
					operators::divide_reverse< size_t, IOType, IOType >() );

				converged = true;
				ret = ret ? ret : grb::dot(
					converged,
					clusters_and_distances_prev, clusters_and_distances,
					comparison_monoid,
					grb::operators::equal_first< indexIOType, indexIOType, bool >()
				);

			} while( ret == SUCCESS && !converged && iter < max_iter );

			if( iter == max_iter ) {
				std::cout << "\tkmeans reached maximum number of iterations!" << std::endl;
				return FAILED;
			}
			if( converged ) {
				std::cout << "\tkmeans converged successfully after " << iter
					<< " iterations." << std::endl;
				return SUCCESS;
			}

			std::cout << "\tkmeans finished with unexpected return code!" << std::endl;
			return ret;
		}

	} // namespace algorithms

} // namespace grb

#endif // end _H_GRB_KMEANS

