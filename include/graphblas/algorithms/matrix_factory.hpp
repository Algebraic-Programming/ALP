
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
 * @file
 *
 * Implements the following matrix factory methods:
 * - empty
 * - eye
 * - eye< void >
 * - identity
 * - identity< void >
 * - full
 * - full< void >
 * - dense
 * - dense< void >
 * - ones
 * - zeros
 * - random< void >
 * - random
 *
 * @author Benjamin Lozes
 * @date 7th of August, 2023
 */

#ifndef _H_GRB_MATRIX_FACTORY
#define _H_GRB_MATRIX_FACTORY

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <graphblas/utils/iterators/ChainedIterators.hpp>
#include <graphblas/utils/iterators/regular.hpp>

#include <graphblas.hpp>

namespace grb::factory {

	namespace internal {

		size_t compute_diag_length( size_t nrows, size_t ncols, long k ) {
			const size_t k_abs = static_cast< size_t >( ( k < 0L ) ? -k : k );
			return ( k_abs >= nrows || k_abs >= ncols )
				? 0
				: std::min(
					std::min( nrows, ncols ),
					std::min( ncols - k_abs, nrows - k_abs )
				);
		}

		template<
			typename D, Descriptor descr,
			typename RIT, typename CIT, typename NIT,
			Backend implementation,
			class IteratorV,
			typename std::enable_if< not std::is_void< D >::value, int >::type = 0
		>
		Matrix< D, implementation, RIT, CIT, NIT > createIdentity_generic(
			size_t nrows, size_t ncols,
			long k,
			IOMode io_mode,
			IteratorV V_iter
		) {
			const size_t diag_length = compute_diag_length( nrows, ncols, k );

			Matrix< D, implementation, RIT, CIT, NIT > matrix( nrows, ncols, diag_length );
			const RIT k_i_incr = static_cast< RIT >( ( k < 0L ) ? std::abs( k ) : 0UL );
			const CIT k_j_incr = static_cast< CIT >( ( k < 0L ) ? 0UL : std::abs( k ) );
			grb::utils::containers::Range< RIT > I( k_i_incr, diag_length + k_i_incr );
			grb::utils::containers::Range< CIT > J( k_j_incr, diag_length + k_j_incr );

			RC rc = buildMatrixUnique< descr >(
				matrix, I.begin(), J.begin(), V_iter, diag_length, io_mode
			);

			assert( rc == SUCCESS );
			if( rc != SUCCESS ) {
				// Todo: Throw an exception or just return an empty matrix?
				// Nb: We should consider the distributed case if we throw an exception.
				throw std::runtime_error(
					"Error: createIdentity_generic failed: rc = " + grb::toString( rc )
				);
			}
			return matrix;
		}

		template<
			typename D, Descriptor descr,
			typename RIT, typename CIT, typename NIT,
			Backend implementation
		>
		Matrix< void, implementation, RIT, CIT, NIT > createIdentity_generic(
			size_t nrows, size_t ncols,
			long k,
			IOMode io_mode,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		) {
			const size_t diag_length = compute_diag_length( nrows, ncols, k );
			Matrix< void, implementation, RIT, CIT, NIT > matrix(
				nrows, ncols, diag_length
			);
			grb::utils::containers::Range< RIT > I( 0, diag_length );
			grb::utils::containers::Range< CIT > J( 0, diag_length );

			RC rc = buildMatrixUnique< descr >(
				matrix, I.begin(), J.begin(), diag_length, io_mode
			);

			assert( rc == SUCCESS );
			if( rc != SUCCESS ) {
				// Todo: Throw an exception or just return an empty matrix?
				// Nb: We should consider the distributed case if we throw an exception.
				throw std::runtime_error(
					"Error: createIdentity_generic<void> failed: rc = " + grb::toString( rc )
				);
			}
			return matrix;
		}

	} // namespace internal

	/**
	 * Build an empty matrix, with no non-zero elements.
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix (unused).
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > empty(
		size_t nrows, size_t ncols, IOMode io_mode = SEQUENTIAL
	) {
		(void)io_mode;
		return Matrix< D, implementation, RIT, CIT, NIT >( nrows, ncols, 0UL );
	}

	/**
	 * Build an identity matrix. Output matrix will contain
	 * min( \a nrows, \a ncols ) non-zero elements, or less if \a k
	 * is not zero.
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows           The number of rows of the matrix.
	 * @param ncols           The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param identity_value  The value of each non-zero element (default = 1).
	 * @param k               The diagonal offset (default = 0).
	 *                        A positive value indicates an offset above the main
	 *                        diagonal, and a negative value below the main diagonal.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > eye(
		size_t nrows, size_t ncols, IOMode io_mode,
		D identity_value = static_cast< D >( 1 ),
		long k = 0L
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		const size_t diag_length = internal::compute_diag_length( nrows, ncols, k );
		const grb::utils::containers::ConstantVector< D > V( identity_value, diag_length );
		return internal::createIdentity_generic<
				D, descr, RIT, CIT, NIT, implementation
			>( nrows, ncols, k, io_mode, V.cbegin() );
	}

	/**
	 * Build an identity pattern matrix. Output matrix will contain
	 * min( \a nrows, \a ncols ) non-zero elements, or less if \a k
	 * is not zero.
	 *
	 * @note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D              The type of a non-zero element (void).
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows           The number of rows of the matrix.
	 * @param ncols           The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param k               The diagonal offset (default = 0).
	 *                        A positive value indicates an offset above the main
	 *                        diagonal, and a negative value below the main diagonal.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template< typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0 >
	Matrix< void, implementation, RIT, CIT, NIT > eye(
		size_t nrows, size_t ncols, IOMode io_mode, long k = 0L
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		return internal::createIdentity_generic<
				void, descr, RIT, CIT, NIT, implementation
			>( nrows, ncols, k, io_mode );
	}

	/**
	 * Build an identity matrix. Output matrix will contain
	 * \a n non-zero elements.
	 *
	 * @note Alias for factory::eye( n, n, io_mode )
	 *
	 * @tparam D                  The type of a non-zero element.
	 * @tparam descr              The descriptor used to build the matrix.
	 * @tparam RIT                The type used for row indices.
	 * @tparam CIT                The type used for column indices.
	 * @tparam NIT                The type used for non-zero indices.
	 * @tparam implementation     The backend implementation used to build
	 *                            the matrix (default: config::default_backend).
	 * @param[in] n               The number of rows/columns of the matrix.
	 * @param[in] io_mode         The I/O mode used to build the matrix.
	 * @param[in] identity_value  The value of each non-zero element (default = 1)
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > identity(
		size_t n, IOMode io_mode, D identity_value = static_cast< D >( 1 )
	) {
		return eye< D, descr, RIT, CIT, NIT, implementation >(
			n, n, io_mode, identity_value
		);
	}

	/**
	 * Build an identity pattern matrix. Output matrix will contain
	 * \a n non-zero elements.
	 *
	 * @note Alias for factory::eye< void, ... >( n, n, io_mode )
	 *
	 * @note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D              The type of a non-zero element (void).
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param n               The number of rows/columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param identity_value  The value of each non-zero element (default = 1)
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template< typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0 >
	Matrix< void, implementation, RIT, CIT, NIT > identity( size_t n, IOMode io_mode ) {
		return eye< void, descr, RIT, CIT, NIT, implementation >( n, n, io_mode );
	}

	/**
	 * Build an identity matrix with the given values.
	 * Output matrix will contain \a n non-zero elements.
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @tparam ValueIterator  The type of the iterator used to provide the values.
	 * @param n               The number of rows/columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param values          The iterator used to provide the values.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		class ValueIterator,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > identity(
		size_t n, IOMode io_mode,
		ValueIterator values,
		const typename std::enable_if<
			std::is_same<
				typename std::iterator_traits< ValueIterator >::value_type,
				D
			>::value,
			void
		>::type * = nullptr
	) {
		if( n == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				n, n, io_mode
			);
		}

		return internal::createIdentity_generic<
				D, descr, RIT, CIT, NIT, implementation, ValueIterator
			>( n, n, 0L, io_mode, values );
	}

	/**
	 * Build an identity matrix with the given values.
	 * Output matrix will contain \a n non-zero elements.
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @tparam ValueIterator  The type of the iterator used to provide the values.
	 * @param n               The number of rows/columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param values          The iterator used to provide the values.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > identity(
		size_t n, IOMode io_mode, const D * values
	) {
		if( n == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				n, n, io_mode
			);
		}

		return internal::createIdentity_generic<
				D, descr, RIT, CIT, NIT, implementation
			>(
				n, n, 0L, io_mode, values
			);
	}

	/**
	 * Build a dense matrix filled with a given value.
	 * Output matrix will contain nrows * ncols non-zero elements.
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param value           The value of each non-zero element.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > full(
		size_t nrows, size_t ncols, IOMode io_mode, D value
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		Matrix< D, implementation, RIT, CIT, NIT > matrix(
			nrows, ncols, nrows * ncols
		);
		// Initialise rows indices container with a range from 0 to nrows,
		// each value repeated ncols times.
		grb::utils::containers::Range< RIT > I( 0, nrows, 1, ncols );
		// Initialise columns values container with a range from 0 to ncols
		// repeated nrows times.
		grb::utils::containers::ChainedIteratorsVector<
				typename grb::utils::containers::Range< CIT >::const_iterator
			> J( nrows );
		for( size_t i = 0; i < nrows; ++i ) {
			J.push_back( grb::utils::containers::Range< CIT >( 0, ncols ) );
		}
		// Initialise values container with the given value.
		grb::utils::containers::ConstantVector< D > V( value, nrows * ncols );
		assert( std::distance( I.begin(), I.end() ) == std::distance( J.begin(), J.end() ) );
		assert( std::distance( I.begin(), I.end() ) == std::distance( V.begin(), V.end() ) );

		RC rc = buildMatrixUnique< descr >(
			matrix, I.begin(), I.end(), J.begin(), J.end(), V.begin(), V.end(), io_mode
		);

		assert( rc == SUCCESS );
		if( rc != SUCCESS ) {
			// Question for review: Throw an exception or just return an empty matrix?
			//                  Nb: We should consider the distributed case if we throw an exception.
			throw std::runtime_error(
				"Error: factory::full<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Build a dense pattern matrix.
	 * Output matrix will contain nrows * ncols non-zero elements.
	 *
	 * @note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D              The type of a non-zero element (void).
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, implementation, RIT, CIT, NIT > full(
		size_t nrows, size_t ncols, IOMode io_mode
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		Matrix< void, implementation, RIT, CIT, NIT > matrix(
			nrows, ncols, nrows * ncols
		);
		// Initialise rows indices container with a range from 0 to nrows,
		// each value repeated ncols times.
		grb::utils::containers::Range< RIT > I( 0, nrows, 1, ncols );
		// Initialise columns values container with a range from 0 to ncols
		// repeated nrows times.
		grb::utils::containers::ChainedIteratorsVector<
				typename grb::utils::containers::Range< CIT >::const_iterator
			> J( nrows );
		for( size_t i = 0; i < nrows; ++i ) {
			J.push_back( grb::utils::containers::Range< CIT >( 0, ncols ) );
		}
		assert( std::distance( I.begin(), I.end() ) == std::distance( J.begin(), J.end() ) );
		RC rc = buildMatrixUnique< descr >(
			matrix, I.begin(), I.end(), J.begin(), J.end(), io_mode
		);

		if( rc != SUCCESS ) {
			// Question for review: Throw an exception or just return an empty matrix?
			//                  Nb: We should consider the distributed case if we throw an exception.
			throw std::runtime_error(
				"Error: factory::full<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Build a dense matrix filled with a given value.
	 *
	 * @note Alias for factory::full( nrows, ncols, io_mode, value )
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param value           The value of each non-zero element.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< not std::is_void< D >::value, int >::type = 0
	>
	Matrix< D, implementation, RIT, CIT, NIT > dense(
		size_t nrows, size_t ncols, IOMode io_mode, D value
	) {
		return full< D, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode, value
		);
	}

	/**
	 * Build a dense pattern matrix.
	 *
	 * @note Alias for factory::full< void, ... >( nrows, ncols, io_mode )
	 *
	 * @note This method is specialised for pattern matrices (void non-zero type).
	 *
	 * @tparam D              The type of a non-zero element (void).
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	>
	Matrix< void, implementation, RIT, CIT, NIT > dense(
		size_t nrows, size_t ncols, IOMode io_mode
	) {
		return full< void, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode
		);
	}

	/**
	 * Build a matrix filled with ones.
	 *
	 * @note Alias for factory::full( nrows, ncols, io_mode, 1 )
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows            The number of rows of the matrix.
	 * @param ncols            The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > ones(
		size_t nrows, size_t ncols, IOMode io_mode,
		typename std::enable_if< ! std::is_void< D >::value, int >::type = 0
	) {
		return full< D, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode, static_cast< D >( 1 )
		);
	}

	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< void, implementation, RIT, CIT, NIT > ones(
		size_t nrows, size_t ncols, IOMode io_mode,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		return full< D, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode
		);
	}

	/**
	 * Build a matrix filled with zeros.
	 *
	 * @note Alias for factory::full( nrows, ncols, io_mode, 0 )
	 *
	 * @tparam D              The type of a non-zero element.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows           The number of rows of the matrix.
	 * @param ncols           The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > zeros(
		size_t nrows, size_t ncols, IOMode io_mode,
		typename std::enable_if< ! std::is_void< D >::value, int >::type = 0
	) {
		return full< D, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode, static_cast< D >( 1 )
		);
	}

	template< typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend >
	Matrix< void, implementation, RIT, CIT, NIT > zeros( size_t nrows, size_t ncols, IOMode io_mode, typename std::enable_if< std::is_void< D >::value, int >::type = 0 ) {
		return full< D, descr, RIT, CIT, NIT, implementation >( nrows, ncols, io_mode );
	}

	/**
	 * Build a matrix filled with random values at random positions.
	 *
	 * @tparam D                      The type of a non-zero values.
	 * @tparam descr                  The descriptor used to build the matrix.
	 * @tparam RIT                    The type used for row indices.
	 * @tparam CIT                    The type used for column indices.
	 * @tparam NIT                    The type used for non-zero indices.
	 * @tparam RandomDeviceType       The type of the random device used to generate
	 *                                the random data.
	 * @tparam RowDistributionType    The type of the distribution used to generate
	 *                                the row indices.
	 * @tparam ColDistributionType    The type of the distribution used to generate
	 *                                the column indices.
	 * @tparam ValueDistributionType  The type of the distribution used to generate
	 *                                the values.
	 * @tparam implementation         The backend implementation used to build
	 *                                the matrix (default: config::default_backend).
	 *
	 * @param nrows                   The number of rows of the matrix.
	 * @param ncols                   The number of columns of the matrix.
	 * @param io_mode                 The I/O mode used to build the matrix.
	 * @param sparsity		          The sparsity factor of the matrix, 1.0 being
	 *                                a dense matrix and 0.0 being an empty matrix.
	 * @param rd                      The random device used to generate the random data.
	 * @param row_dist                The distribution used to generate the row indices.
	 * @param col_dist                The distribution used to generate the column indices.
	 * @param val_dist                The distribution used to generate the values.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		typename RandomGeneratorType,
		typename RowDistributionType,
		typename ColDistributionType,
		typename ValueDistributionType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > random(
		size_t nrows, size_t ncols, IOMode io_mode,
		double sparsity, RandomGeneratorType & rgen,
		RowDistributionType & row_dist,
		ColDistributionType & col_dist,
		ValueDistributionType & val_dist
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		const size_t nvals = nrows * ncols * std::max( 1.0, std::min( 1.0, sparsity ) );
		Matrix< D, implementation, RIT, CIT, NIT > matrix( nrows, ncols, nvals );

		std::vector< RIT > I( nvals );
		std::vector< CIT > J( nvals );
		std::vector< D > V( nvals );
		for( size_t i = 0; i < nvals; ++i ) {
			I[ i ] = row_dist( rgen );
			J[ i ] = col_dist( rgen );
			V[ i ] = val_dist( rgen );
		}

		RC rc = buildMatrixUnique< descr >(
			matrix, I.begin(), I.end(), J.begin(), J.end(), V.begin(), V.end(), io_mode
		);

		assert( rc == SUCCESS );
		if( rc != SUCCESS ) {
			// Question for review: Throw an exception or just return an empty matrix?
			//                  Nb: We should consider the distributed case if we throw an exception.
			throw std::runtime_error(
				"Error: factory::random failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		typename RandomGeneratorType,
		typename RowDistributionType,
		typename ColDistributionType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< void, implementation, RIT, CIT, NIT > random(
		size_t nrows, size_t ncols, IOMode io_mode,
		double sparsity, RandomGeneratorType & rgen,
		RowDistributionType & row_dist, ColDistributionType & col_dist,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >( nrows, ncols, io_mode );
		}

		const size_t nvals = nrows * ncols * std::max( 1.0, std::min( 1.0, sparsity ) );
		Matrix< void, implementation, RIT, CIT, NIT > matrix( nrows, ncols, nvals );

		std::vector< RIT > I( nvals );
		std::vector< CIT > J( nvals );
		for( size_t i = 0; i < nvals; ++i ) {
			I[ i ] = row_dist( rgen );
			J[ i ] = col_dist( rgen );
		}

		RC rc = buildMatrixUnique< descr >(
			matrix, I.begin(), I.end(), J.begin(), J.end(), io_mode
		);

		assert( rc == SUCCESS );
		if( rc != SUCCESS ) {
			// Question for review: Throw an exception or just return an empty matrix?
			//                  Nb: We should consider the distributed case if we throw an exception.
			throw std::runtime_error(
				"Error: factory::random<void> failed: rc = " + grb::toString( rc )
			);
		}

		return matrix;
	}

	/**
	 * Build a matrix filled with random values at random positions.
	 *
	 * Will use an \a mt19937 random generator with the given seed.
	 * The distributions used to generate the random data are uniform_real_distribution
	 * with the following ranges:
	 * * row indices:    [0, nrows - 1]
	 * * column indices: [0, ncols - 1]
	 * * values:         [0, 1]
	 *
	 * @tparam D              The type of a non-zero values.
	 * @tparam descr          The descriptor used to build the matrix.
	 * @tparam RIT            The type used for row indices.
	 * @tparam CIT            The type used for column indices.
	 * @tparam NIT            The type used for non-zero indices.
	 * @tparam implementation The backend implementation used to build
	 *                        the matrix (default: config::default_backend).
	 * @param nrows           The number of rows of the matrix.
	 * @param ncols           The number of columns of the matrix.
	 * @param io_mode         The I/O mode used to build the matrix.
	 * @param sparsity		  The sparsity factor of the matrix, 1.0 being
	 *                        a dense matrix and 0.0 being an empty matrix.
	 * @param seed            The seed used to generate the random values.
	 *
	 * @return Matrix< D, implementation, RIT, CIT, NIT >
	 */
	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< D, implementation, RIT, CIT, NIT > random(
		size_t nrows, size_t ncols, IOMode io_mode,
		double sparsity, unsigned long seed = 0UL
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		std::mt19937 gen( seed );

		const std::uniform_real_distribution< RIT > rowDistribution(
			static_cast< RIT >( 0 ), static_cast< RIT >( nrows - 1 )
		);
		const std::uniform_real_distribution< CIT > colDistribution(
			static_cast< CIT >( 0 ), static_cast< CIT >( ncols - 1 )
		);
		const std::uniform_real_distribution< D > valuesDistribution(
			static_cast< D >( 0 ), static_cast< D >( 1 )
		);

		return random< D, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode,
			sparsity, gen,
			rowDistribution, colDistribution, valuesDistribution
		);
	}

	template<
		typename D,
		Descriptor descr = descriptors::no_operation,
		typename RIT = config::RowIndexType,
		typename CIT = config::ColIndexType,
		typename NIT = config::NonzeroIndexType,
		Backend implementation = grb::config::default_backend
	>
	Matrix< void, implementation, RIT, CIT, NIT >
	random(
		size_t nrows, size_t ncols, IOMode io_mode,
		double sparsity, unsigned long seed = 0UL,
		typename std::enable_if< std::is_void< D >::value, int >::type = 0
	) {
		if( nrows == 0 || ncols == 0 ) {
			return empty< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, io_mode
			);
		}

		std::mt19937 gen( seed );

		const std::uniform_real_distribution< RIT > rowDistribution(
			static_cast< RIT >( 0 ), static_cast< RIT >( nrows - 1 )
		);
		const std::uniform_real_distribution< CIT > colDistribution(
			static_cast< CIT >( 0 ), static_cast< CIT >( ncols - 1 )
		);

		return random< void, descr, RIT, CIT, NIT, implementation >(
			nrows, ncols, io_mode,
			sparsity, gen,
			rowDistribution, colDistribution
		);
	}

} // namespace grb::factory

#endif // end _H_GRB_MATRIX_FACTORY
