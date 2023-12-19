
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @file matrix_factory.hpp
 *
 * Implements the following matrix factory methods:
 *
 * @author Benjamin Lozes
 * @date 7th of August, 2023
 */

#ifndef _H_GRB_MATRIX_FACTORY
#define _H_GRB_MATRIX_FACTORY

#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

#include <graphblas/utils/iterators/adapter.hpp>
#include <graphblas/utils/iterators/regular.hpp>

#include <graphblas.hpp>


/**
 * A namespace for factories of ALP/GraphBLAS matrices.
 *
 * Calling a factory function, in the case of an ALP process with multiple user
 * processes is an \em collective call. ALP guarantees that if a call fails at
 * one user process, the call also fails at all other user processes.
 *
 * Just as with constructing any ALP container, the use of factory functions,
 * when an error is encountered, will result in C++ exceptions being thrown.
 *
 * The following matrix factory methods are supported:
 *  -# #diag,
 *  -# #empty,
 *  -# #eye,
 *  -# #identity,
 *  -# #full,
 *  -# #dense,
 *  -# #ones,
 *  -# #zeros, and
 *  -# #random.
 *
 * For each factory method, there is a variant for producing pattern as well as
 * non-pattern matrices. All these functions are implemented on top of core ALP
 * primitives.
 *
 * \par Performance semantics
 * \parblock
 * While the implementation of all factory methods guarantee their construction
 * within \f$ \Theta(1) \f$ work-space, the involved work and data movement are
 * \f$ \mathcal{O}(n+m) \f$ instead, where \f$ n \f$ is the maximum matrix
 * dimension and \f$ m \f$ the number of nonzeroes in the produced matrix.
 * The work and data movement furthermore do \em not scale with the number of
 * user processes-- the factory methods are implemented using sequential I/O
 * semantics.
 *
 * \note Providing these matrix factory methods via sequential I/O is needed
 *       since otherwise we may not implement this functionality on the ALP
 *       algorithm level, and must instead either 1) implement these as core ALP
 *       primitives, or 2) provide efficient introspection to the distribution
 *       of \em arbitrary backends. The former entails significant work, while
 *       the latter does not yet exist.
 *
 * \note If scalable factory methods are required, please submit a feature
 *       request and/or contact the maintainers.
 *
 * Most matrix factory methods are shared-memory parallel. Those which are
 * \em not shared-memory parallel are all variants of #random.
 *
 * @see grb::IOMode for a more in-depth description of sequential (versus
 *                  parallel) I/O semantics.
 * \endparblock
 *
 * \par A note on aliases
 * Some methods are aliases of one another -- different from core ALP
 * primitives, the goal for this factory class, just as with any ALP algorithm,
 * is to provide the end-user with productive tools; the goal is not to provide
 * some minimal set of primitives that allow the widest range of algorithms to
 * efficiently be implemented on top of them. Given this difference in end-goal,
 * maintaining multiple aliases are hence encouraged, as long as they indeed
 * increase productivity.
 */
namespace grb::factory {

	/**
	 * Builds an empty matrix, without any values.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The requested number of rows of the returned matrix.
	 * @param[in] n The requested number of columns of the returned matrix.
	 *
	 * @returns An empty matrix of the requested type.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > empty(
		const size_t m, const size_t n
	) {
		return Matrix< D, implementation, RIT, CIT, NIT >( m, n, 0 );
	}

	namespace internal {

		/**
		 * Given a matrix size as well as a diagonal offset, computes the length
		 * (in number of elements) of that diagonal.
		 *
		 * @param[in] nrows The number of rows of the matrix.
		 * @param[in] ncols The number of columns of the matrix.
		 * @param[in] k     The diagonal offset (may be a negative integer).
		 *
		 * @returns The diagonal length, in number of elements.
		 */
		size_t compute_diag_length(
			const size_t nrows, const size_t ncols,
			const long k
		) {
			constexpr const long zero = static_cast< long >( 0 );
			const auto k_abs = static_cast< size_t >(
				(k < zero) ? -k : k );
			return (k_abs >= nrows || k_abs >= ncols)
				? 0
				: std::min(
					std::min( nrows, ncols ),
					std::min( ncols - k_abs, nrows - k_abs )
				);
		}

		/**
		 * Generic implementation for creating an identity matrix.
		 *
		 * @tparam D       The type of a non-zero element.
		 * @tparam RIT     The type used for row indices.
		 * @tparam CIT     The type used for column indices.
		 * @tparam NIT     The type used for non-zero indices.
		 * @tparam backend The selected backend.
		 *
		 * @param[in] nrows The number of rows of the matrix.
		 * @param[in] ncols The number of columns of the matrix.
		 * @param[in] k     The diagonal offset (may be a negative integer).
		 *
		 * @param[in] V_iter Start iterator to a collection of values to be put on the
		 *                   diagonal.
		 * @param[in] V_end  Iterator to the same collection as \a V_iter, at its end
		 *                   position.
		 *
		 * \warning Assumes \a V_iter contains enough elements (but potentially more)
		 *          to fill up the requested diagonal.
		 *
		 * \note This is usually achieved implicitly by \a V_iter being a repeater
		 *       iterator of sufficient size, e.g., of full matrix size.
		 */
		template<
			typename D, typename RIT, typename CIT, typename NIT,
			Backend backend,
			class IteratorV,
			typename std::enable_if< !std::is_void< D >::value, int >::type = 0
		>
		Matrix< D, backend, RIT, CIT, NIT > createIdentity_generic(
			const size_t nrows, const size_t ncols,
			const long k,
			const IteratorV V_iter, const IteratorV V_end
		) {
			// some useful scalars
			constexpr const long s_zero = static_cast< long >( 0 );
			constexpr const size_t u_zero = static_cast< size_t >( 0 );
			const size_t diag_length = compute_diag_length( nrows, ncols, k );
			assert( static_cast< size_t >(std::distance( V_iter, V_end )) >=
				diag_length );
#ifdef NDEBUG
			(void) V_end;
#endif

			// declare matrix-to-be-returned
			Matrix< D, backend, RIT, CIT, NIT > matrix(
				nrows, ncols, diag_length );

			// get row- and column-wise starting positions
			const RIT k_i_incr = static_cast< RIT >(
				(k < s_zero) ? std::abs( k ) : u_zero );
			const CIT k_j_incr = static_cast< CIT >(
				(k < s_zero) ? u_zero : std::abs( k ) );

			// translate it to a range so we can get iterators
			grb::utils::containers::Range< RIT > I( k_i_incr, diag_length + k_i_incr );
			grb::utils::containers::Range< CIT > J( k_j_incr, diag_length + k_j_incr );

			// construct the matrix from the given iterators
			const RC rc = buildMatrixUnique(
				matrix, I.begin(), J.begin(), V_iter, diag_length, SEQUENTIAL
			);

			if( rc != SUCCESS ) {
				throw std::runtime_error(
					"Error: createIdentity_generic failed: rc = " + grb::toString( rc )
				);
			}
			return matrix;
		}

		/**
		 * Generic implementation for creating an identity matrix.
		 *
		 * @tparam D       The type of a non-zero element.
		 * @tparam RIT     The type used for row indices.
		 * @tparam CIT     The type used for column indices.
		 * @tparam NIT     The type used for non-zero indices.
		 * @tparam backend The selected backend.
		 *
		 * @param[in] nrows The number of rows of the matrix.
		 * @param[in] ncols The number of columns of the matrix.
		 * @param[in] k     The diagonal offset (may be a negative integer).
		 *
		 * This is the variant for void matrices.
		 */
		template<
			typename D, typename RIT, typename CIT, typename NIT,
			Backend backend,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		>
		Matrix< void, backend, RIT, CIT, NIT > createIdentity_generic(
			const size_t nrows, const size_t ncols,
			const long k
		) {
			// pattern matrix variant of the above
			constexpr const long s_zero = static_cast< long >( 0 );
			constexpr const size_t u_zero = static_cast< size_t >( 0 );
			const size_t diag_length = compute_diag_length( nrows, ncols, k );
			Matrix< void, backend, RIT, CIT, NIT > matrix(
				nrows, ncols, diag_length
			);
			const RIT k_i_incr = static_cast< RIT >(
				(k < s_zero) ? std::abs( k ) : u_zero );
			const CIT k_j_incr = static_cast< CIT >(
				(k < s_zero) ? u_zero : std::abs( k ) );
			grb::utils::containers::Range< RIT > I( k_i_incr, diag_length + k_i_incr );
			grb::utils::containers::Range< CIT > J( k_j_incr, diag_length + k_j_incr );
			const RC rc = buildMatrixUnique(
				matrix, I.begin(), J.begin(), diag_length, SEQUENTIAL
			);
			if( rc != SUCCESS ) {
				throw std::runtime_error(
					"Error: createIdentity_generic<void> failed: rc = " + grb::toString( rc )
				);
			}
			return matrix;
		}

	} // namespace internal

	/**
	 * Builds a diagonal matrix with the given values.
	 *
	 * This function takes as input an iterator-pair over any STL-style container
	 * of diagonal values. This employs sequential I/O semantics as described in
	 * #grb::IOMode.
	 *
	 * \warning Therefore, performance-critical code should not depend on this
	 *          factory method-- building the requested matrix inherently does not
	 *          scale with the number of user processes due to the sequential input
	 *          it entails.
	 *
	 * The output matrix will contain \f$ q = \min\{ m, n \} \f$ non-zeroes or less
	 * if \a k is not zero.
	 *
	 * @tparam D             The type of a non-zero element.
	 * @tparam RIT           The type used for row indices.
	 * @tparam CIT           The type used for column indices.
	 * @tparam NIT           The type used for non-zero indices.
	 * @tparam backend       The selected backend.
	 * @tparam ValueIterator The type of the iterator used to provide the values.
	 *
	 * @param[in] m      The number of rows of the matrix.
	 * @param[in] n      The number of columns of the matrix.
	 * @param[in] values The iterator over the diagonal values.
	 * @param[in] valEnd Iterator in end-position matching \a values.
	 * @param[in] k      The diagonal offset (default = 0). A positive value
	 *                   indicates an offset above the main diagonal, while a
	 *                   negative value indicates an offset below the main
	 *                   diagonal.
	 *
	 * The distance between \a values and \a valEnd must be at least \f$ q \f$, or
	 * less if \a k is not zero.
	 *
	 * @returns The requested diagonal matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		class ValueIterator,
		typename std::enable_if< !std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > diag(
		const size_t m, const size_t n,
		const ValueIterator values, const ValueIterator valEnd,
		const long k = static_cast< long >( 0 )
	) {
		// static sanity checks
		static_assert(
			std::is_convertible<
				typename std::iterator_traits< ValueIterator >::value_type,
				D
			>::value,
			"grb::factory::diag called with value types that are not convertible to the "
			"matrix nonzero type."
		);

		// check trivial dispatch
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( n, n );
		}

		return internal::createIdentity_generic<
				D, RIT, CIT, NIT, backend, ValueIterator
			>( m, n, k, values, valEnd );
	}

	/**
	 * Builds an identity matrix.
	 *
	 * Output matrix will contain \f$ \min\{ m, n \} \f$ non-zero elements or less
	 * if \a k is not zero.
	 *
	 * @tparam D       The type of non-zero elements.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m               The number of rows of the matrix.
	 * @param[in] n               The number of columns of the matrix.
	 * @param[in] identity_value  The value of each non-zero element (default = 1).
	 * @param[in] k               The diagonal offset (default = 0). A positive
	 *                            value indicates an offset above the main
	 *                            diagonal, while a negative value indicates an
	 *                            offset below the main diagonal.
	 *
	 * @returns The requested identity matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = config::default_backend,
		typename std::enable_if< !std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > eye(
		const size_t m, size_t n,
		const D identity_value = static_cast< D >( 1 ),
		const long k = static_cast< long >( 0 )
	) {
		// check for possible trivial dispatch
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		// we are in the non-pattern case, so we need a repeated iterator for the
		// values-- determine worst-case length (cheaper than computing the actual
		// diagonal length)
		const size_t wcl = std::max( m, n );
		const grb::utils::containers::ConstantVector< D > V( identity_value, wcl );

		// call generic implementation
		return internal::createIdentity_generic<
				D, RIT, CIT, NIT, backend 
			>( m, n, k, V.cbegin(), V.cend() );
	}

	/**
	 * Builds an identity pattern matrix.
	 *
	 * Output matrix will contain \f$ \min\{ m, n \} \f$ non-zero elements or less
	 * if \a k is not zero.
	 *
	 * @note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D       The type of a non-zero element (void).
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 * @param[in] k The diagonal offset (default = 0). A positive value indicates
	 *              an offset above the main diagonal, while a negative value
	 *              indicates an offset below the main diagonal.
	 *
	 * @returns The requested identity matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, backend, RIT, CIT, NIT > eye(
		const size_t m, const size_t n, long k = static_cast< long >(0)
	) {
		// check trivial case
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		// dispatch to generic function
		return internal::createIdentity_generic<
				void, RIT, CIT, NIT, backend
			>( m, n, k );
	}

	/**
	 * Builds an identity matrix.
	 *
	 * \note This is an alias for factory::eye( n, n ). It differs only in that
	 *       this function produces square matrices.
	 *
	 * See #factory::eye for detailed documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] n              The number of rows/columns of the matrix.
	 * @param[in] identity_value The value of each non-zero element (default = 1).
	 *
	 * @returns The requested identity matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > identity(
		const size_t n, const D identity_value = static_cast< D >( 1 )
	) {
		return eye< D, RIT, CIT, NIT, backend >(
			n, n, identity_value
		);
	}

	/**
	 * Builds an identity pattern matrix.
	 *
	 * This is the pattern matrix variant -- see the non-patterm variant for
	 * complete documentation.
	 *
	 * @tparam D       The type of a non-zero element (void).
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] n The number of rows/columns of the matrix.
	 *
	 * @returns The requested identity matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, backend, RIT, CIT, NIT > identity( const size_t n ) {
		return eye< void, RIT, CIT, NIT, backend >( n, n );
	}

	/**
	 * Builds a dense matrix filled with a given value.
	 *
	 * \note ALP/GraphBLAS does not have efficient support for dense matrices--
	 *       the dense data will be stored in a sparse format, which will not
	 *       lead to efficient computations. See ALP/Dense for efficient dense
	 *       linear algebra support.
	 *
	 * Output matrix will contain \f$ mn \f$ non-zero elements.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m     The number of rows of the matrix.
	 * @param[in] n     The number of columns of the matrix.
	 * @param[in] value The value of each non-zero element.
	 *
	 * @returns The requested full matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< !std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > full(
		const size_t m, const size_t n, const D value
	) {
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		const size_t nz = m * n;
		if( nz / m != n ) {
			throw std::runtime_error( "Requested dense matrix overflows in number of "
				"nonzeroes." );
		}

		Matrix< D, backend, RIT, CIT, NIT > matrix( m, n, nz );

		// Initialise rows indices container with a range from 0 to nrows,
		// each value repeated ncols times.
		grb::utils::containers::Range< RIT > I( 0, m, 1, n );
		// Initialise columns values container with a range from 0 to ncols
		// repeated nrows times. There are two ways of doing this:
		//  1) using ChainedIterators, or
		//  2) using the iterator::Adapter.
		// We select here way #2, and disable way #1:
#if 0
		grb::utils::containers::ChainedIteratorsVector<
				typename grb::utils::containers::Range< CIT >::const_iterator
			> J( m );
		for( size_t i = 0; i < m; ++i ) {
			J.emplace_back( grb::utils::containers::Range< CIT >( 0, n ) );
		}
#endif
		grb::utils::containers::Range< size_t > J_raw( 0, nz );
		const auto entryInd2colInd = [&m, &n] (const CIT k) -> CIT {
				return k / m;
			};
		auto J_begin = grb::utils::iterators::make_adapter_iterator(
			J_raw.cbegin(), J_raw.cend(), entryInd2colInd );
		const auto J_end = grb::utils::iterators::make_adapter_iterator(
			J_raw.cend(), J_raw.cend(), entryInd2colInd );
		// Initialise values container with the given value.
		grb::utils::containers::ConstantVector< D > V( value, nz );
#ifndef NDEBUG
		const size_t Isz = std::distance( I.begin(), I.end() );
		const size_t Jsz = std::distance( J_begin, J_end );
		const size_t Vsz = std::distance( V.begin(), V.end() );
		assert( Isz == Jsz );
		assert( Isz == Vsz );
#endif

		const RC rc = buildMatrixUnique(
			matrix,
			I.begin(), I.end(), J_begin, J_end, V.begin(), V.end(),
			SEQUENTIAL
		);

		if( rc != SUCCESS ) {
			throw std::runtime_error(
				"Error: factory::full<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Build a dense pattern matrix.
	 *
	 * \note ALP/GraphBLAS does not have efficient support for dense matrices--
	 *       the dense data will be stored in a sparse format, which will not
	 *       lead to efficient computations. See ALP/Dense for efficient dense
	 *       linear algebra support.
	 *
	 * Output matrix will contain \f$ mn \f$ non-zero elements.
	 *
	 * \note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D       The type of a non-zero element (void).
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m     The number of rows of the matrix.
	 * @param[in] n     The number of columns of the matrix.
	 *
	 * @returns The requested full matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, backend, RIT, CIT, NIT > full(
		const size_t m, const size_t n
	) {
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		const size_t nz = m * n;
		if( nz / m != n ) {
			throw std::runtime_error( "Requested dense matrix overflows in number of "
				"nonzeroes." );
		}

		Matrix< void, backend, RIT, CIT, NIT > matrix( m, n, nz );

		// Initialise rows indices container with a range from 0 to nrows,
		// each value repeated ncols times.
		grb::utils::containers::Range< RIT > I( 0, m, 1, n );
		// Initialise columns values container with a range from 0 to ncols
		// repeated nrows times. As mentioned above, there are two ways to provide
		// iterators, we disable the first way (via ChainedIterators) and enable the
		// iterators::adaptor way:
#if 0
		grb::utils::containers::ChainedIteratorsVector<
				typename grb::utils::containers::Range< CIT >::const_iterator
			> J( m );
		for( size_t i = 0; i < m; ++i ) {
			J.emplace_back( grb::utils::containers::Range< CIT >( 0, n ) );
		}
#endif
		grb::utils::containers::Range< size_t > J_raw( 0, nz );
		const auto nonzeroInd2colInd = [&m] (const size_t k) -> size_t {
				return k / m;
			};
		auto J_begin = grb::utils::iterators::make_adapter_iterator(
			J_raw.cbegin(), J_raw.cend(), nonzeroInd2colInd );
		auto J_end = grb::utils::iterators::make_adapter_iterator(
			J_raw.cend(), J_raw.cend(), nonzeroInd2colInd );
		assert( std::distance( I.begin(), I.end() ) ==
			std::distance( J_begin, J_end ) );
		const RC rc = buildMatrixUnique(
			matrix, I.begin(), I.end(), J_begin, J_end, SEQUENTIAL
		);

		if( rc != SUCCESS ) {
			throw std::runtime_error(
				"Error: factory::full<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Builds a dense matrix filled with a given value.
	 *
	 * \note This is an alias for #grb::factory::full -- see that function for
	 *       complete documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m     The number of rows of the matrix.
	 * @param[in] n     The number of columns of the matrix.
	 * @param[in] value The value of each non-zero element.
	 *
	 * @returns The requested dense matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > dense(
		const size_t m, const size_t n, const D value
	) {
		return full< D, RIT, CIT, NIT, backend >( m, n, value );
	}

	/**
	 * Builds a dense pattern matrix.
	 *
	 * \note This is an alias for #grb::factory::full -- see that function for
	 *       complete documentation.
	 *
	 * @tparam D       The type of a non-zero element (void).
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 *
	 * @returns The requested dense matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, backend, RIT, CIT, NIT > dense(
		const size_t m, const size_t n
	) {
		return full< void, RIT, CIT, NIT, backend >( m, n );
	}

	/**
	 * Builds a matrix filled with ones.
	 *
	 * \note This is an alias for factory::full( m, n, 1 ).
	 *
	 * @see #grb::factory::full for complete documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 *
	 * @returns Returns a dense matrix with entries one.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< !std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, backend, RIT, CIT, NIT > ones(
		const size_t m, const size_t n
	) {
		return full< D, RIT, CIT, NIT, backend >( m, n, static_cast< D >( 1 ) );
	}

	/**
	 * Builds a dense pattern matrix filled with ones.
	 *
	 * \note This is an alias for factory::full( m, n, 1 ). Since a pattern matrix
	 *       is requested, however, the numerical value 1 is meaningless.
	 *
	 * @see #grb::factory::full for complete documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 *
	 * @returns Returns a dense pattern matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, backend, RIT, CIT, NIT > ones(
		const size_t m, const size_t n
	) {
		return full< D, RIT, CIT, NIT, backend >( m, n );
	}

	/**
	 * Builds a matrix filled with zeros.
	 *
	 * \note This is an alias for factory::full( m, n, 0 )
	 *
	 * @see #grb::factory::full for complete documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 *
	 * @returns A dense matrix of zeroes.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend
	>
	Matrix< D, backend, RIT, CIT, NIT > zeros(
		const size_t m, const size_t n,
		typename std::enable_if< !std::is_void< D >::value, int >::type = 0
	) {
		return full< D, RIT, CIT, NIT, backend >( m, n, static_cast< D >( 1 ) );
	}

	/**
	 * Builds a matrix filled with zeros.
	 *
	 * \note This is an alias for factory::full( m, n, 0 ). Since a pattern matrix
	 *       is requested, however, the numerical value 0 is meaningless.
	 *
	 * @see #grb::factory::full for complete documentation.
	 *
	 * @tparam D       The type of a non-zero element.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m The number of rows of the matrix.
	 * @param[in] n The number of columns of the matrix.
	 *
	 * @returns A dense pattern matrix.
	 */

	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend
	>
	Matrix< void, backend, RIT, CIT, NIT > zeros(
		const size_t m, const size_t n,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		return full< D, RIT, CIT, NIT, backend >( m, n );
	}

	/**
	 * Builds a matrix filled with random values at random positions.
	 *
	 * \warning Usually, (uniform) random matrices do \em not mimic practical
	 *          graph and sparse matrix structures at all. Therefore, use this
	 *          functionality with care.
	 *
	 * More advanced models that mimic graphs and/or sparse matrices from different
	 * practical application domains may be integrated by passing different
	 * distributions to the below API.
	 *
	 * \note One might be inclined to use this functionality to implement
	 *       randomised linear algebra methods. Doing so would be inefficient since
	 *       such algorithms would require repeated sampling. A different and
	 *       significantly more suitable design, akin to that of views in
	 *       ALP/Dense, is under development.
	 *
	 * @tparam D                     The type of a non-zero values.
	 * @tparam RIT                   The type used for row indices.
	 * @tparam CIT                   The type used for column indices.
	 * @tparam NIT                   The type used for non-zero indices.
	 * @tparam RandomDeviceType      The type of the random device used to generate
	 *                               the random data.
	 * @tparam RowDistributionType   The type of the distribution used to generate
	 *                               the row indices.
	 * @tparam ColDistributionType   The type of the distribution used to generate
	 *                               the column indices.
	 * @tparam ValueDistributionType The type of the distribution used to generate
	 *                               the values.
	 * @tparam backend               The selected backend.
	 *
	 * @param[in] m        The number of rows of the matrix.
	 * @param[in] n        The number of columns of the matrix.
	 * @param[in] sparsity The sparsity factor of the matrix, 1.0 being a dense
	 *                     matrix and 0.0 being an empty matrix.
	 * @param[in] rgen     The random device used to generate the random data.
	 * @param[in] row_dist The distribution used to generate the row indices.
	 * @param[in] col_dist The distribution used to generate the column indices.
	 * @param[in] val_dist The distribution used to generate the values.
	 *
	 * @returns The requested random matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		typename RandomGeneratorType,
		typename RowDistributionType,
		typename ColDistributionType,
		typename ValueDistributionType,
		Backend backend = grb::config::default_backend
	>
	Matrix< D, backend, RIT, CIT, NIT > random(
		const size_t m, const size_t n,
		const double sparsity,
		RandomGeneratorType &rgen,
		RowDistributionType &row_dist,
		ColDistributionType &col_dist,
		ValueDistributionType &val_dist
	) {
		// FIXME guard against overflow
		const size_t nvals = m * n * std::max( 1.0, std::min( 1.0, sparsity ) );

		if( m == 0 || n == 0 || nvals == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		Matrix< D, backend, RIT, CIT, NIT > matrix( m, n, nvals );

		std::vector< RIT > I( nvals );
		std::vector< CIT > J( nvals );
		std::vector< D > V( nvals );
		for( size_t i = 0; i < nvals; ++i ) {
			I[ i ] = row_dist( rgen );
			J[ i ] = col_dist( rgen );
			V[ i ] = val_dist( rgen );
		}
		// FIXME filter out / re-sample any repeated entries

		const RC rc = buildMatrixUnique(
			matrix,
			I.begin(), I.end(), J.begin(), J.end(), V.begin(), V.end(),
			SEQUENTIAL
		);

		if( rc != SUCCESS ) {
			throw std::runtime_error(
				"Error: factory::random failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Builds a pattern matrix filled with random values at random positions.
	 *
	 * This is the pattern-specialisation of the #grb::factory::random function.
	 *
	 * \warning Usually, (uniform) random matrices do \em not mimic practical
	 *          graph and sparse matrix structures at all. Therefore, use this
	 *          functionality with care.
	 *
	 * More advanced models that mimic graphs and/or sparse matrices from different
	 * practical application domains may be integrated by passing different
	 * distributions to the below API.
	 *
	 * \note One might be inclined to use this functionality to implement
	 *       randomised linear algebra methods. Doing so would be inefficient since
	 *       such algorithms would require repeated sampling. A different and
	 *       significantly more suitable design, akin to that of views in
	 *       ALP/Dense, is under development.
	 *
	 * @tparam D                     The type of a non-zero values.
	 * @tparam RIT                   The type used for row indices.
	 * @tparam CIT                   The type used for column indices.
	 * @tparam NIT                   The type used for non-zero indices.
	 * @tparam RandomDeviceType      The type of the random device used to generate
	 *                               the random data.
	 * @tparam RowDistributionType   The type of the distribution used to generate
	 *                               the row indices.
	 * @tparam ColDistributionType   The type of the distribution used to generate
	 *                               the column indices.
	 * @tparam ValueDistributionType The type of the distribution used to generate
	 *                               the values.
	 * @tparam backend               The selected backend.
	 *
	 * @param[in] m        The number of rows of the matrix.
	 * @param[in] n        The number of columns of the matrix.
	 * @param[in] sparsity The sparsity factor of the matrix, 1.0 being a dense
	 *                     matrix and 0.0 being an empty matrix.
	 * @param[in] rgen     The random device used to generate the random data.
	 * @param[in] row_dist The distribution used to generate the row indices.
	 * @param[in] col_dist The distribution used to generate the column indices.
	 * @param[in] val_dist The distribution used to generate the values.
	 *
	 * @returns The requested random matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		typename RandomGeneratorType,
		typename RowDistributionType,
		typename ColDistributionType,
		Backend backend = grb::config::default_backend
	>
	Matrix< void, backend, RIT, CIT, NIT > random(
		const size_t m, const size_t n,
		const double sparsity,
		RandomGeneratorType &rgen,
		RowDistributionType &row_dist,
		ColDistributionType &col_dist,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		// FIXME guard against overflow
		const size_t nvals = m * n * std::max( 1.0, std::min( 1.0, sparsity ) );

		if( m == 0 || n == 0 || nvals == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		Matrix< void, backend, RIT, CIT, NIT > matrix( m, n, nvals );

		std::vector< RIT > I( nvals );
		std::vector< CIT > J( nvals );
		for( size_t i = 0; i < nvals; ++i ) {
			I[ i ] = row_dist( rgen );
			J[ i ] = col_dist( rgen );
		}
		// FIXME filter out / re-sample any repeated entries

		const RC rc = buildMatrixUnique(
			matrix, I.begin(), I.end(), J.begin(), J.end(), SEQUENTIAL
		);
		if( rc != SUCCESS ) {
			throw std::runtime_error(
				"Error: factory::random<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Builds a matrix filled with random values at random positions.
	 *
	 * Will use an \a mt19937 random generator with the given seed.
	 *
	 * The distributions used to generate the random data are
	 * uniform_real_distribution with the following ranges:
	 *  -# row indices:    [0, m - 1]
	 *  -# column indices: [0, n - 1]
	 *  -# values:         [0, 1]
	 *
	 * \warning Usually, uniform random matrices do \em not mimic practical graph
	 *          and sparse matrix structures at all. Therefore, use this
	 *          functionality with care.
	 *
	 * @tparam D       The type of a non-zero values.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m        The number of rows of the matrix.
	 * @param[in] n        The number of columns of the matrix.
	 * @param[in] sparsity The sparsity factor of the matrix, 1.0 being a dense
	 *                     matrix and 0.0 being an empty matrix.
	 * @param[in] seed     The seed used to generate the random values.
	 *
	 * @returns The requested random matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend
	>
	Matrix< D, backend, RIT, CIT, NIT > random(
		const size_t m, const size_t n,
		const double sparsity,
		const unsigned long seed = static_cast< unsigned long >( 0 )
	) {
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		std::mt19937 gen( seed );

		const std::uniform_real_distribution< RIT > rowDistribution(
			static_cast< RIT >( 0 ), static_cast< RIT >( m - 1 )
		);
		const std::uniform_real_distribution< CIT > colDistribution(
			static_cast< CIT >( 0 ), static_cast< CIT >( n - 1 )
		);
		const std::uniform_real_distribution< D > valuesDistribution(
			static_cast< D >( 0 ), static_cast< D >( 1 )
		);

		return random< D, RIT, CIT, NIT, backend >(
			nrows, ncols, sparsity,
			gen, rowDistribution, colDistribution, valuesDistribution
		);
	}

	/**
	 * Builds a pattern matrix filled with random values at random positions.
	 *
	 * Will use an \a mt19937 random generator with the given seed.
	 *
	 * The distributions used to generate the random data are
	 * uniform_real_distribution with the following ranges:
	 *  -# row indices:    [0, m - 1]
	 *  -# column indices: [0, n - 1]
	 *
	 * \warning Usually, uniform random matrices do \em not mimic practical graph
	 *          and sparse matrix structures at all. Therefore, use this
	 *          functionality with care.
	 *
	 * This is the pattern-specialisation of the #grb::factory::random function
	 * specialised for uniform random sampling using the \a mt19937 RNG.
	 *
	 * @tparam D       The type of a non-zero values.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 * @tparam backend The selected backend.
	 *
	 * @param[in] m        The number of rows of the matrix.
	 * @param[in] n        The number of columns of the matrix.
	 * @param[in] sparsity The sparsity factor of the matrix, 1.0 being a dense
	 *                     matrix and 0.0 being an empty matrix.
	 * @param[in] seed     The seed used to generate the random values.
	 *
	 * @returns The requested random matrix.
	 */
	template<
		typename D,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend backend = grb::config::default_backend
	>
	Matrix< void, backend, RIT, CIT, NIT > random(
		const size_t m, const size_t n,
		const double sparsity,
		const unsigned long seed = static_cast< unsigned long >(0),
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		if( m == 0 || n == 0 ) {
			return empty< D, RIT, CIT, NIT, backend >( m, n );
		}

		std::mt19937 gen( seed );

		const std::uniform_real_distribution< RIT > rowDistribution(
			static_cast< RIT >( 0 ), static_cast< RIT >( m - 1 )
		);
		const std::uniform_real_distribution< CIT > colDistribution(
			static_cast< CIT >( 0 ), static_cast< CIT >( n - 1 )
		);

		return random< void, RIT, CIT, NIT, backend >(
			m, n, sparsity,
			gen, rowDistribution, colDistribution
		);
	}

} // namespace grb::factory

#endif // end _H_GRB_MATRIX_FACTORY

