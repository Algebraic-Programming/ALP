
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

/*
 * @author Verner Vlacic
 */

#ifndef _H_GRB_KMEANS
#define _H_GRB_KMEANS

#include <chrono>
#include <random>

#include <assert.h>

#include <graphblas.hpp>

using namespace grb;


namespace grb {

	namespace algorithms {


		/**
		 * a simple implementation of the k++ initialisation algorithm for kmeans
		 * 
		 * @param[in,out] K k by m matrix containing the current k means as row vectors
		 * @param[in]     X m by n matrix containing the n points to be classified as
		 *                  column vectors
		 * @param[in]    op coordinatewise distance operator, squared difference by
		 *                  default
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
			Matrix< IOType > &K,                // kxm matrix containing the current k means as row vectors
			const Matrix< IOType > &X,          // mxn matrix containing the n points to be classified as column vectors
			const Operator dist_op = Operator() // coordinatewise distance operator, squared difference by default
		) {
			// declare monoids and semirings

			Monoid<
				grb::operators::add< IOType >,
				grb::identities::zero
			> add_monoid;

			Monoid<
				grb::operators::min< IOType >,
				grb::identities::infinity
			> min_monoid;

			// TODO check output type casting for right_assign_if in internalops 
			Semiring< 
				grb::operators::add< IOType >,
				grb::operators::right_assign_if< bool, IOType, IOType >,
				grb::identities::zero,
				grb::identities::logical_true
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
#ifndef DETERMINISTIC
			std::default_random_engine random_generator( seed_uniform );
#else
			std::default_random_engine random_generator( 1234 );
#endif
				std::uniform_int_distribution< size_t > uniform( 0, n - 1 );
				i = uniform( random_generator );
			}

			for( size_t l = 0; ret == SUCCESS && l < k; ++l ) {

				ret = grb::clear( col_select );
				ret = ret ? ret : grb::clear( selected );
				ret = ret ? ret : grb::clear( selected_distances );

				ret = ret ? ret : grb::setElement( selected_indices, i, l );

				ret = ret ? ret : grb::setElement( col_select, true, i );

				ret = ret ? ret : grb::vxm< grb::descriptors::transpose_matrix >( selected, col_select, X, pattern_sum );

				ret = ret ? ret : grb::vxm( selected_distances, selected, X, add_monoid, dist_op );

				ret = ret ? ret : grb::foldl( min_distances, selected_distances, min_monoid );

				//TODO the remaining part of the loop should be replaced with the alias algorithm

				IOType range = add_monoid.template getIdentity< IOType >();
				ret = ret ? ret : grb::foldl( range, min_distances, add_monoid );

				double sample = -1;
				if( ret == SUCCESS ) {
					const size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
#ifndef DETERMINISTIC
			
					std::default_random_engine generator( seed_uniform );
#else
					std::default_random_engine generator( 1234 );
#endif					
					std::uniform_real_distribution< double > uniform( 0, 1 );
					sample = uniform( generator );
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

			// create the matrix K by selecting the columns of X indexed by selected_indices

			// declare pattern matrix
			Matrix< void > M( k, n );
			ret = ret ? ret : grb::resize( M, n );

			if( ret == SUCCESS ) {
				auto converter = grb::utils::makeVectorToMatrixConverter< void, size_t >(
					selected_indices, [](const size_t &ind, const size_t &val ) {
						return std::make_pair( ind, val );
					}
				);
				ret = grb::buildMatrixUnique( M, converter.begin(), converter.end(), PARALLEL );
			}

			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X, pattern_sum, RESIZE );
			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X, pattern_sum );

			if ( ret != SUCCESS ) {
				std::cout << "\tkpp_initialization finished with unexpected return code!" << std::endl;
			}

			return ret;
		}

		/**
		 * an implementation of the orthogonal initialisation algorithm for kmeans
		 *
		 * TODO add documentation
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType = double,
			class EuclideanSpace = Semiring< 
				grb::operators::add< IOType >,
				grb::operators::mul< IOType, IOType, IOType >,
				grb::identities::zero,
				grb::identities::one
			>	
		>
		RC korth_initialisation(
			Matrix< IOType > &K,                // kxm matrix containing the current k means as row vectors
			const Matrix< IOType > &X,          // mxn matrix containing the n points to be classified as column vectors
			const EuclideanSpace euc_sp = EuclideanSpace() // coordinatewise distance operator, squared difference by default
		) {
			// declare algebraic structures

			const auto &euc_sp_mul = euc_sp.getMultiplicativeOperator();

			Monoid<
				grb::operators::max< IOType >,
				grb::identities::negative_infinity
			> max_monoid;

			Monoid<
				grb::operators::argmin< size_t, IOType >,
				grb::identities::infinity
			> argmin_monoid;

			//TODO check output type casting for right_assign_if in internalops 
			Semiring< 
				grb::operators::add< IOType >,
				grb::operators::right_assign_if< bool, IOType, IOType >,
				grb::identities::zero,
				grb::identities::logical_true
			> pattern_sum;

			// runtime sanity checks: the row dimension of X should match the column dimension of K
			if( ncols( K ) != nrows( X ) ) {
				return MISMATCH;
			}

			// running error code
			RC ret = SUCCESS;

			// get problem dimensions
			const size_t n = ncols( X );
			const size_t m = nrows( X );
			const size_t k = nrows( K );

			// vector of norms of the columns of X
			Vector< IOType > colnorms( n );

			// vector of ones/trues
			Vector< bool > n_ones( n ), m_ones( m ), k_ones( k );
			ret = ret ? ret : grb::set( n_ones, true );
			// std::cout << ret << "No 1" << std::endl;
			ret = ret ? ret : grb::set( m_ones, true );
			ret = ret ? ret : grb::set( k_ones, true );

			// X with normalised columns
			Matrix< IOType > X_norm( m, n );

			// colnorms outer product with m_ones
			Matrix< IOType > colnorms_outer_m_ones( m, n );

			// declare vector of labels from 0 to n-1
			Vector< size_t > labels( n );
			ret = ret ? ret : grb::set< grb::descriptors::use_index >( labels, 0 );

			// declare vector of indices of columns of X selected as the initial centroids
			Vector< size_t > selected_indices( k );

			// declare column selection vector
			Vector< bool > col_select( n );

			// declare selected point
			Vector< IOType > selected( m );

			// declare vector of distances from the selected point
			Vector< IOType > selected_innerprods( n );

			// declare vector of maximum innerprods to all points selected so far
			Vector< IOType > max_innerprods( n );
			ret = ret ? ret : grb::set( max_innerprods, 0 );

			// declare the index-value pair of the final selected index
			std::pair< size_t, IOType > selected_index;


			//COMPUTATION

			// compute column norms
			ret = ret ? ret : grb::resize( X_norm, grb::nnz( X ) );

			ret = ret ? ret : grb::set( X_norm, X );

			ret = ret ? ret : grb::eWiseLambda( [ &X_norm, &euc_sp_mul ]( const size_t i, const size_t j, double &v ){
				grb::apply( v, v, v, euc_sp_mul );
				(void) i;
				(void) j;
			}, X_norm );

			// < grb::descriptors::transpose_matrix > TODO is it correct this is disabled in the below?
			ret = ret ? ret : grb::vxm( colnorms, m_ones, X_norm, pattern_sum );

			ret = ret ? ret : grb::eWiseLambda( [&colnorms]( const size_t i ){
				colnorms[i] = std::sqrt( colnorms[i] );
			}, colnorms );

			// compute outer product of column norms with m_ones
			ret = ret ? ret : grb::outer(
				colnorms_outer_m_ones, m_ones, colnorms,
				operators::left_assign_if< IOType, bool, IOType >(), SYMBOLIC
			);	
			ret = ret ? ret : grb::outer(
				colnorms_outer_m_ones, m_ones, colnorms,
				operators::left_assign_if< IOType, bool, IOType >()
			);

			// divide columns of X by norms to get X_norm
			ret = ret ? ret : grb::clear( X_norm );

			ret = ret ? ret : grb::eWiseApply(
				X_norm, colnorms_outer_m_ones, X,
				operators::divide_reverse< IOType, IOType, IOType >()
			);

			// generate first centroid by selecting a column of X uniformly at random

			size_t seed_uniform = std::chrono::system_clock::now().time_since_epoch().count();
#ifndef DETERMINISTIC
			std::default_random_engine random_generator( seed_uniform );
#else
			std::default_random_engine random_generator( 1234 );
#endif
			std::uniform_int_distribution< size_t > uniform( 0, n-1 );

			size_t i = uniform( random_generator );

			for ( size_t l = 0; l < k; ++l ) {

				ret = ret ? ret : grb::clear( col_select );

				ret = ret ? ret : grb::clear( selected );

				ret = ret ? ret : grb::clear( selected_innerprods );

				// add last selected index i to selected_indices 
				ret = ret ? ret : grb::setElement( selected_indices, i, l );

				// extract column i from X_norm
				ret = ret ? ret : grb::setElement( col_select, true, i );

				ret = ret ? ret : grb::vxm< grb::descriptors::transpose_matrix >( selected, col_select, X_norm, pattern_sum );

				// compute inner products of column i with other columns of X_norm
				ret = ret ? ret : grb::vxm( selected_innerprods, selected, X_norm, euc_sp );

				ret = ret ? ret : grb::eWiseLambda( [&selected_innerprods]( const size_t i ){
					selected_innerprods[ i ] = std::abs( selected_innerprods[ i ] );
				}, selected_innerprods );

				// update maximum inner products of all points to the already selected ones
				ret = ret ? ret : grb::foldl( max_innerprods, selected_innerprods, max_monoid );

				// find the minimum entry of max_innerprods and select the next index
				ret = ret ? ret : grb::dot(
					selected_index, labels, max_innerprods, argmin_monoid,
					operators::zip< size_t, IOType >()
				);

				i = selected_index.first;
			}

			// create the matrix K by selecting the columns of X indexed by selected_indices

			// declare pattern matrix
			Matrix< void > M( k, n );
			ret = ret ? ret : grb::resize( M, n );

			auto converter = grb::utils::makeVectorToMatrixConverter< void, size_t >(
				selected_indices,
				[]( const size_t &ind, const size_t &val ) {
					return std::make_pair( ind, val );
				} );

			ret = ret ? ret : grb::buildMatrixUnique( M, converter.begin(), converter.end(), PARALLEL );
			
			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X, pattern_sum, SYMBOLIC );
			ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K, M, X, pattern_sum );
			
			if ( ret != SUCCESS ) {
				std::cout << "\tkorth_initialization finished with unexpected return code!" << std::endl;
			}

			return ret;
		}

		/**
		 * The kmeans iteration given an initialisation
		 *
		 * TODO add documentation
		 */
		template<
			Descriptor descr = descriptors::no_operation,
			typename IOType = double,
			class Operator = operators::square_diff< IOType, IOType, IOType >
		>
		RC kmeans_iteration(
			Matrix< IOType > &K,                                           // kxm matrix containing the current k means as row vectors
			Vector< std::pair< size_t, IOType > > &clusters_and_distances, // vector containing the class and distance to centroid for each point
			const Matrix< IOType > &X,                                     // mxn matrix containing the n points to be classified as column vectors
			const size_t max_iter = 1000,                                  // maximum number of iterations
			const Operator dist_op = Operator()                            // coordinatewise distance operator, squared difference by default
		) {
			// declare monoids and semirings

			typedef std::pair< size_t, IOType > indexIOType;

			Monoid<
				grb::operators::add< IOType >,
				grb::identities::zero
			> add_monoid;

			Monoid<
				grb::operators::argmin< size_t, IOType >,
				grb::identities::infinity
			> argmin_monoid;

			Monoid< 
				grb::operators::logical_and< bool >,
				grb::identities::logical_true
			> comparison_monoid;

			//TODO check output type casting for right_assign_if in internalops 
			Semiring< 
				grb::operators::add< IOType >,
				grb::operators::right_assign_if< bool, IOType, IOType >,
				grb::identities::zero,
				grb::identities::logical_true
			> pattern_sum;

			Semiring< 
				grb::operators::add< size_t >,
				grb::operators::right_assign_if< size_t, size_t, size_t >,
				grb::identities::zero,
				grb::identities::logical_true
			> pattern_count;

			// runtime sanity checks: the row dimension of X should match the column dimension of K
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
				++iter;

				ret = ret ? ret : grb::set( clusters_and_distances_prev, clusters_and_distances );

				ret = ret ? ret : mxm( Dist, K, X, add_monoid, dist_op, RESIZE );
				ret = ret ? ret : mxm( Dist, K, X, add_monoid, dist_op );

				ret = ret ? ret : vxm(
					clusters_and_distances, labels, Dist, argmin_monoid,
					operators::zip< size_t, IOType >()
				);

				auto converter = grb::utils::makeVectorToMatrixConverter< void, indexIOType >(
				clusters_and_distances,
				[]( const size_t &ind, const indexIOType &pair ) {
					return std::make_pair( pair.first, ind );
				} );

				ret = ret ? ret : grb::buildMatrixUnique( M, converter.begin(), converter.end(), PARALLEL );

				ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K_aux, M, X, pattern_sum, RESIZE );
				ret = ret ? ret : grb::mxm< descriptors::transpose_right >( K_aux, M, X, pattern_sum );

				ret = ret ? ret : grb::mxv( sizes, M, n_ones, pattern_count );

				ret = ret ? ret : grb::outer( V_aux, sizes, m_ones, operators::left_assign_if< IOType, bool, IOType >(), RESIZE );
				ret = ret ? ret : grb::outer( V_aux, sizes, m_ones, operators::left_assign_if< IOType, bool, IOType >() );

				ret = ret ? ret : eWiseApply( K, V_aux, K_aux, operators::divide_reverse< size_t, IOType, IOType >(), RESIZE );
				ret = ret ? ret : eWiseApply( K, V_aux, K_aux, operators::divide_reverse< size_t, IOType, IOType >() );

				converged = true;
				ret = ret ? ret : grb::dot(
					converged,
					clusters_and_distances_prev, clusters_and_distances,
					comparison_monoid,
					grb::operators::equal_first< indexIOType, indexIOType, bool >()
				);

			} while ( ret == SUCCESS && !converged && iter < max_iter );

			if ( iter == max_iter ) {
				std::cout << "\tkmeans reached maximum number of iterations!" << std::endl;
				return FAILED;
			}
			if ( converged ) {
				std::cout << "\tkmeans converged successfully after " << iter << " iterations." << std::endl;
				return SUCCESS;
			}

			std::cout << "\tkmeans finished with unexpected return code!" << std::endl;
			return ret;
		}

	} // end ``algorithms'' namespace

} // end ``grb'' namespace

#endif // end _H_GRB_KMEANS

