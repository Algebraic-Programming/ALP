
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
 * Various utility classes to compare matrice for equality.
 *
 * @author Alberto Scolari
 * @date 20/06/2022
 */

#ifndef _H_GRB_UTILS_MATRIX_CHECK
#define _H_GRB_UTILS_MATRIX_CHECK

#include <iostream>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <vector>

#include <graphblas.hpp>
#include <graphblas/utils/iterators/type_traits.hpp>


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * Class to sort nonzeroes based on ascending rows, descending columns.
			 */
			template<
				typename RowT,
				typename ColT,
				typename ValT
			>
			struct default_nz_sorter {

				inline bool operator()(
					const grb::internal::NonzeroStorage< RowT, ColT, ValT >& a,
					const grb::internal::NonzeroStorage< RowT, ColT, ValT >& b
				) const {
					if( a.i() != b.i() ) {
						return a.i() < b.i();
					}
					return a.j() < b.j();
				}
			};

			/**
			 * Whether two values of input iterators are equal.
			 */
			template<
				typename ValT,
				typename MatIterT,
				typename OrigIterT
			>
			inline bool compare_values(
				const MatIterT& a,
				const OrigIterT& b,
				typename std::enable_if< !std::is_same< ValT, void >::value >::type* =
					nullptr
			) {
				static_assert( grb::utils::is_alp_matrix_iterator< ValT, decltype( a )>::value,
					"MatIterT does not have {i,j,v}() interface" );
				static_assert( grb::utils::is_alp_matrix_iterator< ValT, decltype( b )>::value,
					"MatIterT does not have {i,j,v}() interface" );
				return a.v() == b.v();
			}

			/**
			 * Returns value equality for pattern-matrix input iterator, which is defined
			 * as true always.
			 */
			template<
				typename ValT,
				typename MatIterT,
				typename OrigIterT
			>
			inline bool compare_values(
				const MatIterT& a,
				const OrigIterT& b,
				typename std::enable_if< std::is_same< ValT, void >::value >::type* =
					nullptr
			) {
				(void)a;
				(void)b;
				return true;
			}

			/**
			 * Extracts the value to print from the given input iterator.
			 */
			template<
				typename ValT,
				typename IterT
			>
			inline ValT get_value(
				const IterT& a,
				typename std::enable_if< !std::is_same< ValT, void >::value >::type* =
					nullptr
			) {
				return a.v();
			}

			/**
			 * Extracts the value to print from the given input iterator, for pattern
			 * matrices.
			 */
			template<
				typename ValT,
				typename IterT
			>
			inline char get_value(
				const IterT &a,
				typename std::enable_if< std::is_same< ValT, void >::value >::type* =
					nullptr
			) {
				(void) a;
				return '\0'; // print nothing
			}

		} // end namespace grb::utils::internal

		/**
		 * Sorts nonzeroes in-place according to the sorting criterion implemented in
		 * default_nz_sorter. The boundaries are passed via random access iterators.
		 */
		template<
			typename RowT,
			typename ColT,
			typename ValT,
			typename IterT
		>
		void row_col_nz_sort(
			IterT begin,
			IterT end
		) {
			static_assert( std::is_move_assignable< decltype( *begin ) >::value,
				"*IterT must be move-assignable");
			static_assert( std::is_move_constructible< decltype( *begin ) >::value,
				"*IterT must be move-constructible");
			internal::default_nz_sorter< RowT, ColT, ValT > s;
			std::sort( begin, end, s );
		}

		/**
		 * Extracts all the nonzeroes from matrix \a mat and stores them in \a values.
		 *
		 * This is the value (non-pattern) variant.
		 */
		template<
			typename RowT,
			typename ColT,
			typename ValT,
			enum Backend implementation
		>
		void get_matrix_nnz(
			const Matrix< ValT, implementation > &mat,
			std::vector< grb::internal::NonzeroStorage< RowT, ColT, ValT > > &values,
			typename std::enable_if< !std::is_same< ValT, void >::value >::type* =
				nullptr
		) {
			auto beg1( mat.cbegin() );
			auto end1( mat.cend() );
			for( ; beg1 != end1; ++beg1 ) {
				values.emplace_back( beg1->first.first, beg1->first.second, beg1->second );
			}
		}

		/**
		 * Extracts all the nonzeroes from matrix \a mat and stores them in
		 * \a values.
		 *
		 * Pattern matrix variant.
		 */
		template<
			typename RowT,
			typename ColT,
			typename ValT,
			enum Backend implementation
		>
		void get_matrix_nnz(
			const Matrix< ValT, implementation > &mat,
			std::vector< grb::internal::NonzeroStorage< RowT, ColT, ValT > > &values,
			typename std::enable_if< std::is_same< ValT, void >::value >::type* =
				nullptr
		) {
			auto beg1( mat.cbegin() );
			auto end1( mat.cend() );
			for( ; beg1 != end1; ++beg1 ) {
				values.emplace_back( beg1->first, beg1->second );
			}
		}

		/**
		 * Compares two sequences of nonzeroes via the input iterators of
		 * the respective containers.
		 *
		 * Both sequences are passed via begin and end iterator. The nonzeroes of
		 * both sequences MUST ALREADY BE SORTED according to the same ordering.
		 * The iterators must implement the {i,j,v}() interface.
		 *
		 * @tparam ValT      value type
		 * @tparam MatIterT  iterator type for the first sequence
		 * @tparam OrigIterT iterator type for the second sequence
		 *
		 * @tparam implementation ALP backend
		 *
		 * @param nrows number of rows of the original matrix
		 * @param mat_begin beginning iterator for the first sequence
		 * @param mat_end end iterator for the first sequence
		 * @param origin_begin beginning iterator for the second sequence
		 * @param origin_end end iterator for the second sequence
		 * @param counted_values number of values being checked, i.e. minimum
		 *                       between the length of the two sequences
		 * @param outs ostream to print to
		 * @param log_all_differences whether to print all differences
		 *
		 * @return true if all nonzeroes are equal, i.e. have the same number of
		 *              elements, in the same order and with equal values
		 * @return false if any of the above conditions is not met
		 */
		template<
			typename ValT,
			typename MatIterT,
			typename OrigIterT,
			enum Backend implementation = config::default_backend
		>
		bool compare_non_zeroes(
			const size_t nrows,
			OrigIterT origin_begin,
			const OrigIterT origin_end,
			MatIterT mat_begin,
			const MatIterT mat_end,
			size_t &counted_values,
			std::ostream &outs = std::cout,
			const bool log_all_differences = false
		) {
			static_assert(
				grb::utils::is_alp_matrix_iterator<
					ValT, decltype( mat_begin )
				>::value,
				"MatIterT does not have {i,j,v}() interface"
			);
			static_assert(
				grb::utils::is_alp_matrix_iterator<
					ValT, decltype( origin_begin )
				>::value,
				"MatIterT does not have {i,j,v}() interface"
			);

			size_t counted = 0;

			const size_t pid= spmd<>::pid(), nprocs = spmd<>::nprocs();

			bool result = true;

			while( mat_begin != mat_end && origin_begin != origin_end ) {
				if( ::grb::internal::Distribution<
						implementation
					>::global_index_to_process_id(
						origin_begin.i(), nrows, nprocs
					) != pid
				) {
					// skip non-local non-zeroes
					(void) ++origin_begin;
					continue;
				}
				(void) counted++;
				const bool row_eq = mat_begin.i() == origin_begin.i();
				const bool col_eq = mat_begin.j() == origin_begin.j();
				const bool val_eq = internal::compare_values< ValT >( mat_begin,
					origin_begin );

				const bool all_match = row_eq && col_eq && val_eq;
				result &= all_match;
				if( !all_match && log_all_differences ) {
					outs << "-- different nz, matrix (" << mat_begin.i() << ", "
						<< mat_begin.j() << ")";
					if( !std::is_same< ValT, void >::value ) {
						outs << ": " << internal::get_value< ValT >( mat_begin );
					}
					outs << ", original (" << origin_begin.i() << ", " << origin_begin.j()
						<< ")";
					if( !std::is_same< ValT, void >::value ) {
						outs << ": " << internal::get_value< ValT >( origin_begin );
					}
					outs << std::endl;
				}
				(void) ++origin_begin;
				(void) ++mat_begin;
			}
			counted_values = counted;
			return result;
		}

		template<
			typename D, class Storage1, class Storage2,
			typename std::enable_if< std::is_void< D >::value, int >::type = 0
		>
		RC compare_internal_storage(
			const Storage1 &storage1,
			const Storage2 &storage2,
			const size_t n,
			const size_t nnz
		) {
			(void) nnz;

			for( size_t i = 0; i <= n; i++ ) {
				if( storage1.col_start[ i ] != storage2.col_start[ i ] ) {
					std::cerr << "Error: col_start[" << i << "] is different: "
						<< storage1.col_start[ i ] << " != " << storage2.col_start[ i ]
						<< std::endl;
					return FAILED;
				}
			}
			for( size_t i = 0; i < n; i++ ) {
				for( auto t = storage1.col_start[ i ]; t < storage1.col_start[ i + 1 ]; t++ ) {
					if( storage1.row_index[ t ] != storage2.row_index[ t ] ) {
						std::cerr << "Error: row_index[" << t << "] is different: "
							<< storage1.row_index[ t ] << " != " << storage2.row_index[ t ]
							<< std::endl;
						return FAILED;
					}
				}
			}
			return SUCCESS;
		}

		template<
			typename D, class Storage1, class Storage2,
			typename std::enable_if< not std::is_void< D >::value, int >::type = 0
		>
		RC compare_internal_storage(
			const Storage1 &storage1,
			const Storage2 &storage2,
			const size_t n,
			const size_t nnz
		) {
			RC rc = compare_internal_storage< void >( storage1, storage2, n, nnz );
			if( rc != SUCCESS ) {
				return rc;
			}

			for( size_t i = 0; i < nnz; i++ ) {
				if( storage1.values[ i ] != storage2.values[ i ] ) {
					std::cerr << "Error: values[" << i << "] is different: "
						<< storage1.values[ i ] << " != " << storage2.values[ i ]
						<< std::endl;
					return FAILED;
				}
			}
			return SUCCESS;
		}

		template< typename D1, typename D2 >
		RC compare_crs( const Matrix< D1 > &A, const Matrix< D2> &B ) {
			const size_t m = nrows( A );
			const size_t n = ncols( A );
			if( m != nrows( B ) || n != ncols( B ) ) {
				std::cerr << "Error: matrices have different dimensions:\n"
					<< "\t row count " << m << " != " << nrows( B ) << ";\n"
					<< "\t col count " << n << " != " << ncols( B ) << "\n";
				return FAILED;
			}
			const size_t nz = nnz( A );
			if( nz != nnz( B ) ) {
				std::cerr << "Error: matrices have different number of non-zeroes:\n"
					<< "\t " << nz << " != " << nnz( B ) << "\n";
				return FAILED;
			}

			return compare_internal_storage<
				typename std::conditional<
					std::is_void< D1 >::value ||
					std::is_void< D2 >::value,
					void,
					D1
				>::type
			> (
				grb::internal::getCRS( A ),
				grb::internal::getCRS( B ),
				nrows( A ),
				nnz( A )
			);
		}

		template< typename D1, typename D2 >
		RC compare_ccs( const Matrix< D1 > &A, const Matrix< D2> &B ) {
			if( nrows( A ) != nrows( B ) || ncols( A ) != ncols( B ) ) {
				std::cerr << "Error: matrices have different dimensions\n";
				return FAILED;
			}
			if ( nnz( A ) != nnz( B ) ) {
				std::cerr << "Error: matrices have different number of non-zeroes\n";
				return FAILED;
			}

			return compare_internal_storage<
				typename std::conditional<
					std::is_void< D1 >::value ||
					std::is_void< D2 >::value,
					void,
					D1
				>::type
			> (
				grb::internal::getCCS( A ),
				grb::internal::getCCS( B ),
				ncols( A ),
				nnz( A )
			);
		}

	} // namespace grb::utils

} // namespace grb

#endif // _H_GRB_UTILS_MATRIX_CHECK

