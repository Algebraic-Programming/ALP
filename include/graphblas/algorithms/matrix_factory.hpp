
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
 * - full / dense
 * - ones
 * - zeros
 * -
 *
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
		 *        min( \a nrows, \a ncols ) non-zero elements.
		 *
		 * @tparam D              The type of a non-zero element.
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
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > eye(
			const size_t nrows,
			const size_t ncols,
			IOMode io_mode,
			const D identity_value = static_cast< D >(1),
			const long k = 0L
		) {
			const size_t k_abs = static_cast<size_t>( (k < 0L) ? -k : k );
			const size_t diag_length = ( k_abs >= nrows || k_abs >= ncols )
				? 0
				: std::min( std::min( nrows, ncols ), std::min( ncols - k_abs, nrows - k_abs ) );

			Matrix< D, implementation, RIT, CIT, NIT > matrix( nrows, ncols, diag_length );
			RIT * I = new RIT[ diag_length ];
			CIT * J = new CIT[ diag_length ];
			D   * V = new D  [ diag_length ];
			for( size_t i = 0; i < diag_length; ++i ) {
				I[ i ] = i;
				J[ i ] = i;
				V[ i ] = identity_value;
			}
			RC rc = buildMatrixUnique( matrix, I, J, V, diag_length, io_mode );
			assert( rc == SUCCESS );
			if( rc != SUCCESS ) {
				// Todo: Throw an exception or just return an empty matrix?
				// Nb: We should consider the distributed case if we throw an exception.
				(void) clear( matrix );
			}
			return matrix;
		}

		/**
		 * @brief Build an identity matrix. Output matrix will contain
		 *        \a n non-zero elements.
		 * @note Alias for factory::eye( n, n, io_mode, identity_value, 0 ).
		 *
		 * @tparam D              The type of a non-zero element.
		 * @tparam RIT            The type used for row indices.
		 * @tparam CIT            The type used for column indices.
		 * @tparam NIT            The type used for non-zero indices.
		 * @tparam implementation The backend implementation used to build
		 *                        the matrix (default: config::default_backend).
		 * @param n               The number of rows/columns of the matrix.
		 * @param io_mode         The I/O mode used to build the matrix.
		 * @param identity_value  The value of each non-zero element (default = 1)
		 * @return Matrix< D, implementation, RIT, CIT, NIT >
		 */
		template<
			typename D,
			typename RIT = config::RowIndexType,
			typename CIT = config::ColIndexType,
			typename NIT = config::NonzeroIndexType,
			Backend implementation = grb::config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > identity(
			const size_t n,
			IOMode io_mode,
			const D identity_value = static_cast< D >(1)
		) { return eye( nrows, ncols, io_mode, identity_value ); }

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
			Backend implementation = grb::config::default_backend
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
		 * @brief Build a dense matrix filled with a given value.
		 * @note Alias for factory::full( value, nrows, ncols, io_mode ).
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
			Backend implementation = grb::config::default_backend
		>
		Matrix< D, implementation, RIT, CIT, NIT > dense(
			const D value, const size_t nrows, const size_t ncols, IOMode io_mode
		) { return full( value, nrows, ncols, io_mode ); }

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
		) { return full( static_cast< D >(1), nrows, ncols, io_mode ); }

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
		) { return full( static_cast< D >(0), nrows, ncols, io_mode ); }


	} // namespace factory

} // namespace grb

#endif // end _H_GRB_MATRIX_FACTORY

