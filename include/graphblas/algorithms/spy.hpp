
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
 * Implements a simple matrix spy algorithm.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_ALGORITHMS_SPY
#define _H_GRB_ALGORITHMS_SPY

#include <type_traits>
#include <vector>

#include <graphblas.hpp>


namespace grb {

	namespace algorithms {

		namespace internal {

			/**
			 * \internal This is the main implementation of the spy algorithm. It assumes
			 * a void or boolean input matrix \a in. All other inputs require a
			 * translation step in order to cope with possible explicit zeroes in the
			 * input.
			 */
			template< bool normalize, typename IOType, typename InputType >
			RC spy_from_bool_or_void_input(
				grb::Matrix< IOType > &out, const grb::Matrix< InputType > &in,
				const size_t m, const size_t n,
				const size_t small_m, const size_t small_n
			) {
				static_assert( std::is_same< InputType, bool >::value ||
					std::is_same< InputType, void >::value,
					"Error in call to internal::spy_from_bool_or_void_input"
				);

				// Q must be n by small_n
				grb::Matrix< unsigned char > Q( n, small_n );
				grb::RC ret = grb::resize( Q, n );
				// TODO FIXME use repeating + auto-incrementing iterators
				std::vector< size_t > I, J;
				std::vector< unsigned char > V;
				const double n_sample = static_cast< double >(n) /
					static_cast< double >(small_n);
				for( size_t i = 0; i < n; ++i ) {
					I.push_back( i );
					J.push_back( static_cast< double >(i) / n_sample );
					V.push_back( 1 );
				}
				ret = grb::buildMatrixUnique(
					Q, &(I[0]), &(J[0]), &(V[0]), n,
					grb::SEQUENTIAL
				);

				// P must be small_m by m
				grb::Matrix< unsigned char > P( small_m, m );
				if( ret == SUCCESS ) {
					ret = grb::resize( P, m );
					// TODO FIXME use repeating + auto-incrementing iterators
					std::vector< size_t > I, J;
					std::vector< unsigned char > V;
					const double m_sample = static_cast< double >(m) /
						static_cast< double >(small_m);
					for( size_t i = 0; i < m; ++i ) {
						I.push_back( static_cast< double >(i) / m_sample );
						J.push_back( i );
						V.push_back( 1 );
					}
					ret = grb::buildMatrixUnique(
						P, &(I[0]), &(J[0]), &(V[0]), m,
						grb::SEQUENTIAL
					);
				}

				// tmp must be m by small_n OR small_m by n
				if( ret == SUCCESS && m - small_m > n - small_n ) {
					grb::Semiring<
						grb::operators::add< size_t >,
						grb::operators::left_assign_if< size_t, bool, size_t >,
						grb::identities::zero,
						grb::identities::logical_true
					> leftAssignAndAdd;
					grb::Matrix< size_t > tmp( small_m, n );
					ret = ret ? ret : grb::mxm( tmp, P, in, leftAssignAndAdd, RESIZE );
					ret = ret ? ret : grb::mxm( tmp, P, in, leftAssignAndAdd, EXECUTE );
					ret = ret ? ret : grb::mxm( out, tmp, Q, leftAssignAndAdd, RESIZE );
					ret = ret ? ret : grb::mxm( out, tmp, Q, leftAssignAndAdd, EXECUTE );
				} else {
					grb::Semiring<
						grb::operators::add< size_t >,
						grb::operators::right_assign_if< bool, size_t, size_t >,
						grb::identities::zero,
						grb::identities::logical_true
					> rightAssignAndAdd;
					grb::Matrix< size_t > tmp( m, small_n );
					ret = ret ? ret : grb::mxm( tmp, in, Q, rightAssignAndAdd, RESIZE );
					ret = ret ? ret : grb::mxm( tmp, in, Q, rightAssignAndAdd, EXECUTE );
					ret = ret ? ret : grb::mxm( out, P, tmp, rightAssignAndAdd, RESIZE );
					ret = ret ? ret : grb::mxm( out, P, tmp, rightAssignAndAdd, EXECUTE );
				}

				if( ret == SUCCESS && normalize ) {
					ret = grb::eWiseLambda( [] (const size_t, const size_t, IOType &v ) {
						assert( v > 0 );
						v = static_cast< IOType >( 1 ) / v;
					}, out );
				}

				return ret;
			}

		}

		/**
		 * Given an input matrix and a smaller output matrix, map nonzeroes from the
		 * input matrix into the smaller one and count the number of nonzeroes that
		 * are mapped from the bigger matrix into the smaller.
		 *
		 * @tparam normalize If set to true, will not compute a number of mapped
		 *                   nonzeroes, but its inverse instead (one divided by the
		 *                   count). The default value for this template parameter is
		 *                   <tt>false</tt>.
		 *
		 * @param[out] out  The smaller output matrix.
		 * @param[in]  in   The larger input matrix.
		 *
		 * @returns SUCCESS If the computation completes successfully.
		 * @returns ILLEGAL If \a out has a number of rows or columns larger than that
		 *                  of \a in.
		 *
		 * \warning Explicit zeroes (that when cast from \a InputType to \a bool read
		 *          <tt>false</tt>) \em will be counted as a nonzero by this
		 *          algorithm.
		 *
		 * \note To not count explicit zeroes, pre-process the input matrix \a in, for
		 *       example, as follows: <tt>grb::set( tmp, in, true );</tt>, with \a tmp
		 *       a Boolean or pattern matrix of the same size as \a in.
		 *
		 * \parblock
		 * \par Performance semantics
		 *
		 * \warning This algorithm does NOT request workspace buffers since due to the
		 *          use of level-3 primitives it will have to allocate anyway-- as
		 *          such, this algorithm does not have clear performance semantics and
		 *          should be used with care.
		 *
		 * For performance semantics regarding work, inter-process data movement,
		 * intra-process data movement, synchronisations, and memory use, please see
		 * the specification of the ALP primitives this function relies on. These
		 * performance semantics, with the exception of getters such as #grb::nnz, are
		 * specific to the backend selected during compilation.
		 *
		 * \todo Introduce a template argument <tt>once<tt> that, if set to true,
		 *       signals that \a out is guaranteed to have sufficient capacity to
		 *       complete the computation without having to resize the capacity of
		 *       \a out. The default value for this parameter would be <tt>false</tt>,
		 *       meaning that the algorithm will perform both inspect and execute
		 *       steps; typically, these are equivalent to a symbolic phase which
		 *       resizes output containers, followed by a numeric phase which computes
		 *       the outputs.
		 *       This seems like a useful feature, since \a out may be small and could
		 *       in fact reasonably be resized to have a full capacity (if small
		 *       enough, compared to \in). However, note also that the spy algorithm
		 *       requires workspaces that are larger in size, which a) would have to
		 *       be passed in, and b) will most likely remain sparse, or at least and
		 *       unlike \a out cannot be reasonably be demanded to have full capacity.
		 *       This implies that most likely some symbolic steps need to remain,
		 *       which may only be skipped if the given workspaces happen to have a
		 *       capacity of #grb::nnz( \a in ).
		 *
		 * \endparblock
		 */
		template<
			bool normalize = false,
			typename IOType, typename InputType
		>
		RC spy( grb::Matrix< IOType > &out, const grb::Matrix< InputType > &in ) {
			static_assert( !normalize || std::is_floating_point< IOType >::value,
				"When requesting a normalised spy plot, the data type must be "
				"floating-point"
			);

			const size_t m = grb::nrows( in );
			const size_t n = grb::ncols( in );
			const size_t small_m = grb::nrows( out );
			const size_t small_n = grb::ncols( out );

			// runtime checks and shortcuts
			if( small_m > m ) { return ILLEGAL; }
			if( small_n > n ) { return ILLEGAL; }
			if( small_m == m && small_n == n ) {
				return grb::set< grb::descriptors::structural >( out, in, 1 );
			}

			grb::RC ret = grb::clear( out );

			grb::Matrix< bool > tmp( m, n );
			ret = ret ? ret : grb::resize( tmp, grb::nnz( in ) );
			ret = ret ? ret : grb::set< grb::descriptors::structural >( tmp, in, true );
			ret = ret ? ret : grb::algorithms::internal::template
				spy_from_bool_or_void_input< normalize >(
					out, tmp, m, n, small_m, small_n
				);

			return ret;
		}

		/** Specialisation for boolean input matrices \a in. See grb::algorithms::spy. */
		template< bool normalize = false, typename IOType >
		RC spy( grb::Matrix< IOType > &out, const grb::Matrix< bool > &in ) {
			static_assert( !normalize || std::is_floating_point< IOType >::value,
				"When requesting a normalised spy plot, the data type must be "
				"floating-point" );

			const size_t m = grb::nrows( in );
			const size_t n = grb::ncols( in );
			const size_t small_m = grb::nrows( out );
			const size_t small_n = grb::ncols( out );

			// runtime checks and shortcuts
			if( small_m > m ) { return ILLEGAL; }
			if( small_n > n ) { return ILLEGAL; }
			if( small_m == m && small_n == n ) {
				return grb::set< grb::descriptors::structural >( out, in, 1 );
			}

			grb::RC ret = grb::clear( out );

			ret = ret ? ret : grb::algorithms::internal::template
				spy_from_bool_or_void_input< normalize >(
					out, in, m, n, small_m, small_n
				);

			return ret;
		}

		/** Specialisation for void input matrices \a in. See grb::algorithms::spy. */
		template< bool normalize = false, typename IOType >
		RC spy( grb::Matrix< IOType > &out, const grb::Matrix< void > &in ) {
			static_assert( !normalize || std::is_floating_point< IOType >::value,
				"When requesting a normalised spy plot, the data type must be "
				"floating-point"
			);

			const size_t m = grb::nrows( in );
			const size_t n = grb::ncols( in );
			const size_t small_m = grb::nrows( out );
			const size_t small_n = grb::ncols( out );

			// runtime checks and shortcuts
			if( small_m > m ) { return ILLEGAL; }
			if( small_n > n ) { return ILLEGAL; }
			if( small_m == m && small_n == n ) {
				return grb::set< grb::descriptors::structural >( out, in, 1 );
			}

			grb::RC ret = grb::clear( out );

			ret = ret ? ret : grb::algorithms::internal::template
				spy_from_bool_or_void_input< normalize >(
					out, in, m, n, small_m, small_n
				);

			return ret;
		}

	} // end namespace ``grb::algorithms''

} // end namespace ``grb''

#endif // _H_GRB_ALGORITHMS_SPY

