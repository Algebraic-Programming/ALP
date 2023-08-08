
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
 * Implements the following matrix factory methods:
 * - empty
 * - eye
 * - identity
 * - full
 * - dense
 * - ones
 * - zeros
 *
 * Some of these primitives are also specialised for pattern matrices (void non-zero type).
 *
 * @author Benjamin Lozes
 * @date 7th of August, 2023
 */

#ifndef _H_GRB_MATRIX_FACTORY
#define _H_GRB_MATRIX_FACTORY

#include <iostream>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/regular.hpp>

namespace grb {

	namespace factory {

		namespace internal
		{
			template<
				typename D,
				Descriptor descr,
				typename RIT,
				typename CIT,
				typename NIT,
				Backend implementation,
				class IteratorV,
				typename std::enable_if< not std::is_void< D >::value, int >::type = 0
			>
			Matrix< D, implementation, RIT, CIT, NIT > createIdentity_generic(
				const size_t nrows,
				const size_t ncols,
				const long k,
				IOMode io_mode,
				IteratorV V_iter,
				long V_length_limit = -1L // Negative means the limit is diag_length_k
			) {
				const size_t k_abs = static_cast<size_t>( (k < 0L) ? -k : k );
				const size_t diag_length_k = ( k_abs >= nrows || k_abs >= ncols )
					? 0
					: std::min( std::min( nrows, ncols ), std::min( ncols - k_abs, nrows - k_abs ) );

				const size_t diag_length = ( V_length_limit < 0 )
					? diag_length_k
					: std::min( diag_length_k, static_cast< size_t >( V_length_limit ) );

				Matrix< D, implementation, RIT, CIT, NIT > matrix( nrows, ncols, diag_length );
				const RIT k_i_incr = static_cast< RIT >( ( k < 0L ) ? k_abs : 0UL );
				const CIT k_j_incr = static_cast< CIT >( ( k < 0L ) ? 0UL : k_abs );
				utils::containers::Range< RIT > I( k_i_incr, 1, diag_length + k_i_incr );
				utils::containers::Range< CIT > J( k_j_incr, 1, diag_length + k_j_incr );

				RC rc = ( descr & descriptors::transpose_matrix )
					? buildMatrixUnique( matrix, I.begin(), J.begin(), V_iter, diag_length, io_mode )
					: buildMatrixUnique( matrix, J.begin(), I.begin(), V_iter, diag_length, io_mode );

				assert( rc == SUCCESS );
				if( rc != SUCCESS ) {
					// Todo: Throw an exception or just return an empty matrix?
					// Nb: We should consider the distributed case if we throw an exception.
					(void) clear( matrix );
				}
				return matrix;
			}

			template<
				typename D,
				Descriptor descr,
				typename RIT,
				typename CIT,
				typename NIT,
				Backend implementation
			>
			Matrix< void, implementation, RIT, CIT, NIT > createIdentity_generic(
				const size_t nrows,
				const size_t ncols,
				const long k,
				IOMode io_mode,
				typename std::enable_if< std::is_void< D >::value, int >::type = 0
			) {
				const size_t k_abs = static_cast<size_t>( (k < 0L) ? -k : k );
				const size_t diag_length = ( k_abs >= nrows || k_abs >= ncols )
					? 0
					: std::min( std::min( nrows, ncols ), std::min( ncols - k_abs, nrows - k_abs ) );

				Matrix< void, implementation, RIT, CIT, NIT > matrix( nrows, ncols, diag_length );
				utils::containers::Range< RIT > I( 0, 1, diag_length );
				utils::containers::Range< CIT > J( 0, 1, diag_length );

				RC rc = ( descr & descriptors::transpose_matrix )
					? buildMatrixUnique( matrix, I.begin(), J.begin(), diag_length, io_mode )
					: buildMatrixUnique( matrix, J.begin(), I.begin(), diag_length, io_mode );

				assert( rc == SUCCESS );
				if( rc != SUCCESS ) {
					// Todo: Throw an exception or just return an empty matrix?
					// Nb: We should consider the distributed case if we throw an exception.
					(void) clear( matrix );
				}
				return matrix;
			}

		} // namespace internal


		/**
		 * @brief Build an empty matrix, with no non-zero elements.
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > empty(
			const size_t nrows, const size_t ncols, IOMode io_mode = SEQUENTIAL
		) {
			return Matrix< D, implementation, RIT, CIT, NIT >( nrows, ncols, 0UL );
		}

		/**
		 * @brief Build an identity matrix. Output matrix will contain
		 *        min( \a nrows, \a ncols ) non-zero elements, or less if \a k
		 *        is not zero.
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
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
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
			const size_t nrows,
			const size_t ncols,
			IOMode io_mode,
			const D identity_value = static_cast< D >(1),
			const long k = 0L
		) {
			std::unique_ptr< D[] > V( new D[ std::min( nrows, ncols ) ] );
			std::fill_n( V.get(), std::min( nrows, ncols ), identity_value );
			return internal::createIdentity_generic< D, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, k, io_mode, V.get()
			);
		}

		/**
		 * @brief Build an identity pattern matrix. Output matrix will contain
		 *        min( \a nrows, \a ncols ) non-zero elements, or less if \a k
		 *        is not zero.
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
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
		 */
		template<
			typename D,
			Descriptor descr = descriptors::no_operation,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = config::default_backend,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		>
		Matrix< void, implementation, RIT, CIT, NIT > eye(
			const size_t nrows,
			const size_t ncols,
			IOMode io_mode,
			const long k = 0L
		) {
			return internal::createIdentity_generic< void, descr, RIT, CIT, NIT, implementation >(
				nrows, ncols, k, io_mode
			);
		}

		/**
		 * @brief Build an identity matrix. Output matrix will contain
		 *        \a n non-zero elements.
		 *
		 * @note Alias for factory::eye< ... >( n, n, io_mode, identity_value ).
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam descr          The descriptor used to build the matrix.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param n               The number of rows/columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @param identity_value  The value of each non-zero element (default = 1)
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
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
			const size_t n,
			IOMode io_mode,
			const D identity_value = static_cast< D >(1)
		) { return eye< D, descr, RIT, CIT, NIT, implementation >( n, n, io_mode, identity_value ); }

		/**
		 * @brief Build an identity pattern matrix. Output matrix will contain
		 *        \a n non-zero elements.
		 *
		 * @note Alias for factory::eye< void, ... >( n, n, io_mode ).
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
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
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
		Matrix< void, implementation, RIT, CIT, NIT > identity(
			const size_t n,
			IOMode io_mode
		) { return eye< void, descr, RIT, CIT, NIT, implementation >( n, n, io_mode ); }

		/**
		 * @brief Build an identity matrix with the given values.
		 *        Output matrix will contain \a n non-zero elements.
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
		 * @param V               The iterator used to provide the values.
		 * @param V_length_limit  The maximum number of values to read from \a V
		 *                        (default = -1). The limit is \a n if negative.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
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
			const size_t n,
			IOMode io_mode,
			ValueIterator V,
			long V_length_limit = -1L,
			const typename std::enable_if<
				std::is_same<
					typename std::iterator_traits< ValueIterator >::value_type,
					D
				>::value,
				void
			>::type* = nullptr
		) {
			return internal::createIdentity_generic< D, descr, RIT, CIT, NIT, implementation, ValueIterator >(
				n, n, 0L, io_mode, V, V_length_limit
			);
		}

		/**
		 * @brief Build an identity matrix with the given values.
		 *        Output matrix will contain \a n non-zero elements.
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
		 * @param V               The iterator used to provide the values.
		 * @param V_length_limit  The maximum number of values to read from \a V
		 *                        (default = -1). The limit is \a n if negative.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 *
		 * \parblock
		 * \par Descriptors
		 * The following descriptors are supported:
		 * - descriptors::no_operation
		 * - descriptors::transpose_matrix
		 * \endparblock
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
			const size_t n,
			IOMode io_mode,
			const D * V,
			long V_length_limit = -1L
		) {
			return internal::createIdentity_generic< D, descr, RIT, CIT, NIT, implementation >(
				n, n, 0L, io_mode, V, V_length_limit
			);
		}

		/**
		 * @brief Build a dense matrix filled with a given value.
		 *        Output matrix will contain nrows * ncols non-zero elements.
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param value           The value of each non-zero element.
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend,
			typename std::enable_if< not std::is_void< D >::value, int >::type = 0
		>
		Matrix< D, implementation, RIT, CIT, NIT > full(
			const D value, const size_t nrows, const size_t ncols, IOMode io_mode
		) {
			Matrix< D, implementation, RIT, CIT, NIT > matrix( nrows, ncols, nrows * ncols );
			RC rc = set( matrix, value, Phase::RESIZE );
			assert( rc == SUCCESS );
			rc = rc ? rc : set( matrix, value, Phase::EXECUTE );
			assert( rc == SUCCESS );
			if( rc != SUCCESS ) {
				// Todo: Throw an exception or just return an empty matrix?
				// Nb: We should consider the distributed case if we throw an exception.
				(void) clear( matrix );
			}
			return matrix;
		}

		/**
		 * @brief Build a dense pattern matrix.
		 *        Output matrix will contain nrows * ncols non-zero elements.
		 *
		 * @note This method is specialised for pattern matrices (void non-zero type).
		 *
		 * @tparam D              The type of a non-zero element (void).
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		>
		Matrix< void, implementation, RIT, CIT, NIT > full(
			const size_t nrows, const size_t ncols, IOMode io_mode
		) {
			(void) io_mode;
			return Matrix< void, implementation, RIT, CIT, NIT >( nrows, ncols, nrows * ncols );
		}

		/**
		 * @brief Build a dense matrix filled with a given value.
		 * @note Alias for factory::full< ... >( value, nrows, ncols, io_mode ).
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param value           The value of each non-zero element.
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend,
			typename std::enable_if< not std::is_void< D >::value, int >::type = 0
		>
		Matrix< D, implementation, RIT, CIT, NIT > dense(
			const D value, const size_t nrows, const size_t ncols, IOMode io_mode
		) { return full< D, RIT, CIT, NIT, implementation >( value, nrows, ncols, io_mode ); }

		/**
		 * @brief Build a dense pattern matrix.
		 *
		 * @note Alias for factory::full< void, ... >( nrows, ncols, io_mode ).
		 * @note This method is specialised for pattern matrices (void non-zero type).
		 *
		 * @tparam D              The type of a non-zero element (void).
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		>
		Matrix< D, implementation, RIT, CIT, NIT > dense(
			const size_t nrows, const size_t ncols, IOMode io_mode
		) { return full< void, RIT, CIT, NIT, implementation >( nrows, ncols, io_mode ); }

		/**
		 * @brief Build a matrix filled with ones.
		 *
		 * @note Alias for factory::full( 1, nrows, ncols, io_mode ).
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > ones(
			const size_t nrows, const size_t ncols, IOMode io_mode
		) {
			static_assert( not std::is_void< D >::value, "factory::ones can not be called with a void type" );
			return full( static_cast< D >(1), nrows, ncols, io_mode );
		}

		/**
		 * @brief Build a matrix filled with zeros.
		 *
		 * @note Alias for factory::full( 0, nrows, ncols, io_mode ).
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param nrows            The number of rows of the matrix.
		 * @param ncols            The number of columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > zeros(
			const size_t nrows, const size_t ncols, IOMode io_mode
		) {
			static_assert( not std::is_void< D >::value, "factory::zeros can not be called with a void type" );
			return full( static_cast< D >(0), nrows, ncols, io_mode );
		}


	} // namespace factory

} // namespace grb

#endif // end _H_GRB_MATRIX_FACTORY

