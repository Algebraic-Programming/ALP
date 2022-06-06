
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

#ifndef _H_GRB_UTILS_MATRIX_CHECK
#define _H_GRB_UTILS_MATRIX_CHECK

#include <iostream>
#include <cstddef>
#include <type_traits>
#include <algorithm>

#include <graphblas.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/NonZeroStorage.hpp>

namespace grb {
    namespace utils {

        template< typename RowT, typename ColT, typename ValT > struct __default_nz_sorter {
            inline bool operator()( const grb::utils::NonZeroStorage< RowT, ColT, ValT >& a,
                const grb::utils::NonZeroStorage< RowT, ColT, ValT >& b ) const {
                if( a.i() != b.i() ) {
                    return a.i() < b.i();
                }
                return a.j() < b.j();
            }
        };

        template< typename RowT, typename ColT, typename ValT, typename IterT > void row_col_nz_sort(
            IterT begin, IterT end ) {
            static_assert( std::is_move_assignable< decltype( *begin ) >::value, "*IterT must be move-assignable");
            static_assert( std::is_move_constructible< decltype( *begin ) >::value, "*IterT must be move-constructible");
            __default_nz_sorter< RowT, ColT, ValT > s;
            std::sort( begin, end, s );
        }


        template< typename RowT, typename ColT, typename ValT, enum Backend implementation >
		void get_matrix_nnz(
            const Matrix< ValT, implementation >& mat,
            std::vector< grb::utils::NonZeroStorage< RowT, ColT, ValT > >& values,
            typename std::enable_if< ! std::is_same< ValT, void >::value >::type* = nullptr ) {
            auto beg1( mat.cbegin() );
            auto end1( mat.cend() );
            for( ; beg1 != end1; ++beg1 ) {
                values.emplace_back( beg1->first.first, beg1->first.second, beg1->second );
            }

        }

        template< typename RowT, typename ColT, typename ValT, enum Backend implementation >
		void get_matrix_nnz(
            const Matrix< ValT, implementation >& mat,
            std::vector< grb::utils::NonZeroStorage< RowT, ColT, ValT > >& values,
            typename std::enable_if< std::is_same< ValT, void >::value >::type* = nullptr ) {
            auto beg1( mat.cbegin() );
            auto end1( mat.cend() );
            for( ; beg1 != end1; ++beg1 ) {
                values.emplace_back( beg1->first, beg1->second );
            }
        }


        template< typename ValT, typename MatIterT, typename OrigIterT >
            inline bool __compare_values( const MatIterT& a, const OrigIterT& b,
            typename std::enable_if< ! std::is_same< ValT, void >::value >::type* = nullptr ) {
            static_assert( grb::is_input_iterator< ValT, decltype( a )>::value, "MatIterT does not have {i,j,v}() interface" );
            static_assert( grb::is_input_iterator< ValT, decltype( b )>::value, "MatIterT does not have {i,j,v}() interface" );
            return a.v() == b.v();
        }

        template< typename ValT, typename MatIterT, typename OrigIterT >
            inline bool __compare_values( const MatIterT& a, const OrigIterT& b,
            typename std::enable_if< std::is_same< ValT, void >::value >::type* = nullptr ) {
            (void)a;
            (void)b;
            return true;
        }

        template< typename ValT, typename IterT > inline ValT __get_value( const IterT& a,
            typename std::enable_if< ! std::is_same< ValT, void >::value >::type* = nullptr ) {
            return a.v();
        }

        template< typename ValT, typename IterT > inline char __get_value( const IterT& a,
            typename std::enable_if< std::is_same< ValT, void >::value >::type* = nullptr ) {
            (void)a;
            return '\0'; // print nothing
        }

        /**
         * @brief Non-zeroes of both matrix and original data MUST ALREADY BE SORTED
         *  according to the same ordering. The iterators must implement the {i,j,v}() interface
         *
         * @tparam ValT
         * @tparam MatIterT
         * @tparam OrigIterT
         * @tparam implementation
         * @param nrows
         * @param mat_begin
         * @param mat_end
         * @param origin_begin
         * @param origin_end
         * @param num_local_matrix_nzs
         * @param num_local_original_nzs
         * @param outs
         * @return true
         * @return false
         */
        template< typename ValT, typename MatIterT, typename OrigIterT,
			enum Backend implementation = config::default_backend >
            bool compare_non_zeroes(
            std::size_t nrows,
            OrigIterT origin_begin,
            OrigIterT origin_end,
            MatIterT mat_begin,
            MatIterT mat_end,
            std::size_t& counted_values,
            std::ostream& outs = std::cout,
            bool log_all_differences = false
        ) {
            static_assert( grb::is_input_iterator< ValT, decltype( mat_begin )>::value,
                "MatIterT does not have {i,j,v}() interface" );
            static_assert( grb::is_input_iterator< ValT, decltype( origin_begin )>::value,
                "MatIterT does not have {i,j,v}() interface" );

            std::size_t counted{ 0 };

            const std::size_t pid{ spmd<>::pid() }, nprocs{ spmd<>::nprocs() };

            bool result{ true };

            while( mat_begin != mat_end && origin_begin != origin_end ) {
                if ( ::grb::internal::Distribution< implementation >::global_index_to_process_id(
                        origin_begin.i(), nrows, nprocs ) != pid ) {
                    // skip non-local non-zeroes
                    (void)++origin_begin;
                    continue;
                }
                (void)counted++;
                const bool row_eq{ mat_begin.i() == origin_begin.i() };
                const bool col_eq{ mat_begin.j() == origin_begin.j() };
                const bool val_eq{ __compare_values< ValT >( mat_begin, origin_begin ) };

				const bool all_match{ row_eq && col_eq && val_eq };
				result &= all_match;
				if ( ! all_match && log_all_differences ) {
					outs << "-- different nz, matrix (" << mat_begin.i() << ", " << mat_begin.j() << ")";
					if ( ! std::is_same< ValT, void >::value ) {
						outs << ": " << __get_value< ValT >( mat_begin );
					}
					outs << ", original (" << origin_begin.i() << ", " << origin_begin.j() << ")";
					if ( ! std::is_same< ValT, void >::value ) {
						outs << ": " << __get_value< ValT >( origin_begin );
					}
					outs << std::endl;
				}
                (void)++origin_begin;
                (void)++mat_begin;
            }
            counted_values = counted;
            return result;
        }
    } // namespace utils
} // namespace grb

#endif // _H_GRB_UTILS_MATRIX_CHECK

