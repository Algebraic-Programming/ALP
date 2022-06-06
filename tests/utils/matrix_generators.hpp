
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

#ifndef _GRB_UTILS_MATRIX_GENERATORS_
#define _GRB_UTILS_MATRIX_GENERATORS_

#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace grb {
    namespace utils {

        template< typename T > void compute_parallel_first_nonzero( T num_nonzeroes,
            T& num_nonzeroes_per_process, T& first_local_nonzero ) {
            T num_procs { spmd<>::nprocs() };
            num_nonzeroes_per_process = ( num_nonzeroes + num_procs - 1 ) / num_procs; // round up
            first_local_nonzero = std::min( num_nonzeroes_per_process * spmd<>::pid(),
				num_nonzeroes );
        }

        template< typename T > T compute_parallel_first_nonzero( T num_nonzeroes ) {
            T nnz_per_proc, first;
            compute_parallel_first_nonzero( num_nonzeroes, nnz_per_proc, first);
            return first;
        }

        template< typename T > T compute_parallel_last_nonzero( T num_nonzeroes ) {
            T num_non_zeroes_per_process, first_local_nonzero;
            compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
            return std::min( num_nonzeroes, first_local_nonzero + num_non_zeroes_per_process );
        }

        template< typename T > T compute_parallel_num_nonzeroes( T num_nonzereos ) {
            return compute_parallel_last_nonzero( num_nonzereos ) -
                compute_parallel_first_nonzero( num_nonzereos );
        }

		template< typename SizeT, typename DiffT > DiffT __compute_distance(
			SizeT a, SizeT b ) {
			const SizeT diff{ std::max( a, b ) - std::min( a, b ) };
			if ( diff > static_cast< SizeT >( std::numeric_limits< DiffT >::max() ) ) {
				throw std::range_error( "cannot represent difference" );
			}
			DiffT result{ static_cast< DiffT >( diff ) };
			return a >= b ? result : -result ;
		}

        struct __diag_coord_value {
            std::size_t coord;

            __diag_coord_value( std::size_t _c ): coord( _c ) {}
        };

        template< bool random > struct diag_iterator:
            public std::iterator<
                typename std::conditional< random, std::random_access_iterator_tag,
                std::forward_iterator_tag >::type,
                __diag_coord_value, long, __diag_coord_value*, __diag_coord_value& > {

            using row_coordinate_type = std::size_t;
            using column_coordinate_type = std::size_t;
            using nonzero_value_type = int;

            using input_sizes_t = const std::size_t;
            using self_t = diag_iterator< random >;

            diag_iterator( const self_t& ) = default;

            self_t& operator++() {
                _v.coord++;
                return *this;
            }

            self_t& operator+=( std::size_t offset ) {
                _v.coord += offset;
                return *this;
            }

            bool operator!=( const self_t& other ) const {
                return other._v.coord != this->_v.coord;
            }

            bool operator==( const self_t& other ) const {
                return !( this->operator!=( other ) );
            }

            typename self_t::difference_type operator-( const self_t& other ) const {
				return __compute_distance< std::size_t, typename self_t::difference_type >(
					this->_v.coord, other._v.coord );
            }

            typename self_t::pointer operator->() { return &_v; }

            typename self_t::reference operator*() { return _v; }

            row_coordinate_type i() const { return _v.coord; }

            column_coordinate_type j() const { return _v.coord; }

            nonzero_value_type v() const {
                return static_cast< nonzero_value_type >( _v.coord ) + 1;
            }

            static self_t make_begin( input_sizes_t& size ) {
                (void)size;
                return self_t( 0 );
            }

            static self_t make_end( input_sizes_t& size ) {
                return self_t( size );
            }

            static self_t make_parallel_begin( input_sizes_t& size ) {
                const std::size_t num_nonzeroes{ size };
                std::size_t num_non_zeroes_per_process, first_local_nonzero;
                compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
                return self_t( first_local_nonzero );
            }

            static self_t make_parallel_end( input_sizes_t& size ) {
                const std::size_t num_nonzeroes{ size };
                std::size_t last{ compute_parallel_last_nonzero( num_nonzeroes ) };
                return self_t( last );
            }

            static std::size_t compute_num_nonzeroes( std::size_t size ) {
                return size;
            }

        private:
            typename self_t::value_type _v;

            diag_iterator( std::size_t _c ): _v( _c ) {}

            diag_iterator(): _v( 0) {}

        };

        struct __band_coord_value {
            const std::size_t size;
            std::size_t row;
            std::size_t col;
            __band_coord_value() = delete;
            __band_coord_value( std::size_t _size, std::size_t _r, std::size_t _c ):
                size( _size ), row( _r ), col( _c ) {}
        };

        template< std::size_t BAND, bool random > struct band_iterator:
            public std::iterator<
                typename std::conditional< random, std::random_access_iterator_tag,
                std::forward_iterator_tag >::type,
                __band_coord_value, long, __band_coord_value*, __band_coord_value& > {

            static constexpr std::size_t MAX_ELEMENTS_PER_ROW{ BAND * 2 + 1 };
            static constexpr std::size_t PROLOGUE_ELEMENTS{ ( 3* BAND * BAND + BAND ) / 2 };

            using row_coordinate_type = std::size_t;
            using column_coordinate_type = std::size_t;
            using nonzero_value_type = int;
            using self_t = band_iterator< BAND, random >;
            using input_sizes_t = const std::size_t;

            band_iterator( const self_t& ) = default;

            self_t& operator++() {
                //std::cout << "INCREMENT" << std::endl;
                const std::size_t max_col{ std::min( _v.row + BAND, _v.size - 1 ) };
                if( _v.col < max_col ) {
                    _v.col++;
                } else {
                    _v.row++;
                    _v.col = _v.row < BAND ? 0 : _v.row - BAND;
                }
                return *this;
            }

            self_t& operator+=( std::size_t offset ) {

                //#pragma omp critical
                {
                    //std::cout << "-- ADVANCE offset " << offset << std::endl;
                    //std::cout << "i " << _v.row << " j " << _v.col << " v " << std::endl;
                    const std::size_t position{ coords_to_linear( _v.size, _v.row, _v.col ) };
                    //std::cout << "position is " << position << std::endl;
                    linear_to_coords( _v.size, position + offset, _v.row, _v.col );
                    //std::cout << "++ i " << _v.row << " j " << _v.col << " v " << std::endl;
                }
                return *this;
            }

            bool operator!=( const self_t& other ) const {
                return other._v.row != this->_v.row || other._v.col != this->_v.col;
            }

            bool operator==( const self_t& other ) const {
                return !( this->operator!=( other ) );
            }

            typename self_t::difference_type operator-( const self_t& other ) const {
                const std::size_t this_position{ coords_to_linear( _v.size, _v.row, _v.col ) };
                const std::size_t other_position{
                    coords_to_linear( other._v.size, other._v.row, other._v.col ) };
                /*
                std::cout << " this:: " << _v.size << " " <<  _v.row << " " << _v.col << std::endl;
                std::cout << " other:: " << other._v.size << " " <<  other._v.row << " " << other._v.col << std::endl;
                std::cout << " this position " << this_position << std::endl;
                std::cout << " other position " << other_position << std::endl;
                */
				return __compute_distance< std::size_t, typename self_t::difference_type >(
					this_position, other_position );
            }

            typename self_t::pointer operator->() { return &_v; }

            typename self_t::reference operator*() { return _v; }

            typename self_t::row_coordinate_type i() const { return _v.row; }

            typename self_t::column_coordinate_type j() const { return _v.col; }

            nonzero_value_type v() const {
                return _v.row == _v.col ? static_cast< int >( MAX_ELEMENTS_PER_ROW ) : -1;
            }

            static self_t make_begin( input_sizes_t& size ) {
                __check_size( size );
                return self_t( size, 0, 0 );
            }

            static self_t make_end( input_sizes_t& size ) {
                __check_size( size );
                std::size_t row, col;
                const std::size_t num_nonzeroes{ compute_num_nonzeroes( size ) };
                linear_to_coords( size, num_nonzeroes, row, col );
                //std::cout << "make_end: nnz " << num_nonzeroes << ", row " << row << ", col " << col << std::endl;
                return self_t( size, row, col );
            }

            static self_t make_parallel_begin( input_sizes_t& size ) {
                __check_size( size );
                const std::size_t num_nonzeroes{ compute_num_nonzeroes( size ) };
                std::size_t num_non_zeroes_per_process, first_local_nonzero;
                compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
                std::size_t row, col;
                linear_to_coords( size, first_local_nonzero, row, col );
                return self_t( size, row, col );
            }

            static self_t make_parallel_end( input_sizes_t& size ) {
                __check_size( size );
                const std::size_t num_nonzeroes{ compute_num_nonzeroes( size ) };
                /*
                std::size_t num_non_zeroes_per_process, first_local_nonzero;
                compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
                std::size_t last{ std::min( num_nonzeroes, first_local_nonzero + num_non_zeroes_per_process ) };
                */
                std::size_t last{ compute_parallel_last_nonzero( num_nonzeroes ) };
                std::size_t row, col;
                linear_to_coords( size, last, row, col );
                return self_t( size, row, col );
            }

            static std::size_t compute_num_nonzeroes( std::size_t size ) {
                return 2 * PROLOGUE_ELEMENTS + ( size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW;
            }

        private:
            typename self_t::value_type _v;

            band_iterator( std::size_t size, std::size_t row, std::size_t col ):
                _v( size, row, col ) {
                static_assert( BAND > 0, "BAND must be > 0");
            }

            band_iterator(): _v( 0, 0) {
                static_assert( BAND > 0, "BAND must be > 0");
            }

            static std::size_t __col_to_linear( std::size_t row, std::size_t col ) {
                std::size_t min_col{ row < BAND ? 0 : row - BAND };
                return col - min_col;
            }

            static std::size_t __coords_to_linear_in_prologue( std::size_t row, std::size_t col ) {
                //std::cout << " row " << row << " col " << col << std::endl;
                return row * BAND + row * ( row + 1 ) / 2 + __col_to_linear( row, col );
            }

            static std::size_t coords_to_linear( std::size_t matrix_size, std::size_t row, std::size_t col ) {
                if( row < BAND ) {
                    //std::cout << "here!!!" << std::endl;
                    return __coords_to_linear_in_prologue( row, col );
                }
                if( row < matrix_size - BAND ) {
                    //std::cout << "here 2!!!" << std::endl;
                    return PROLOGUE_ELEMENTS + ( row - BAND ) * MAX_ELEMENTS_PER_ROW + __col_to_linear( row, col );
                }
                if( row < matrix_size ) {
                    //std::cout << "here 3!!!" << std::endl;
                    std::size_t mat_size{ 2 * PROLOGUE_ELEMENTS + ( matrix_size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW };
                    std::size_t prologue_els{ __coords_to_linear_in_prologue( matrix_size - row - 1, matrix_size - col - 1 ) };
                    //std::cout << " mat_size " << mat_size << std::endl;
                    //std::cout << " prologue els " << prologue_els << std::endl;
                    return  mat_size - 1 - prologue_els; // transpose coordinates
                }
                // for points outside of matrix: project to prologue
                return 2 * PROLOGUE_ELEMENTS + ( matrix_size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW
                    + ( row - matrix_size ) * BAND + col + BAND - row;
            }

            static void __linear_to_coords_in_prologue( std::size_t position, std::size_t& row, std::size_t& col ) {
                std::size_t current_row{ 0 };
                //linear search
                for( ; position >= ( current_row + 1 + BAND ) && current_row < BAND; current_row++ ) {
                    position -= ( current_row + 1 + BAND );
                    //std::cout << "subtract " << ( _row + 1 + BAND ) << " get " << position << std::endl;
                }
                row = current_row;
                col = position;
            }

            static void linear_to_coords( std::size_t matrix_size, std::size_t position,
                std::size_t& row, std::size_t& col ) {
                if( position < PROLOGUE_ELEMENTS ) {
                    //std::cout << "in prologue" << std::endl;
                    __linear_to_coords_in_prologue( position, row, col );
                    return;
                }
                //std::cout << "out of prologue" << std::endl;
                position -= PROLOGUE_ELEMENTS;
                const std::size_t max_inner_rows{ matrix_size - 2 * BAND };
                if( position < max_inner_rows * MAX_ELEMENTS_PER_ROW ) {
                    const std::size_t inner_row{ position / MAX_ELEMENTS_PER_ROW };
                    row = BAND + inner_row;
                    position -= inner_row * MAX_ELEMENTS_PER_ROW;
                    col = row - BAND + position % MAX_ELEMENTS_PER_ROW;
                    return;
                }
                position -= ( matrix_size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW;
                if( position < PROLOGUE_ELEMENTS ) {
                    std::size_t end_row, end_col;

                    //__linear_to_coords_in_epilogue( position, end_row, end_col );
                    __linear_to_coords_in_prologue( PROLOGUE_ELEMENTS - 1 - position, end_row, end_col );
                    //std::cout << "== position " << PROLOGUE_ELEMENTS - 1 - position << ", end row " << end_row << ", end col " << end_col << std::endl;
                    row = matrix_size - 1 - end_row;
                    col = matrix_size - 1 - end_col;
                    //std::cout << "== final: row " << row << ", col " << col << std::endl;
                    return;
                }
                position -= PROLOGUE_ELEMENTS;
                row = matrix_size + position / ( BAND + 1 );
                col = row - BAND + position % ( BAND + 1 );
            }

            static void __check_size( std::size_t size ) {
                if( size < 2 * BAND + 1 ) {
                    throw std::domain_error( "matrix too small for band" );
                }
            }
        };

        struct __dense_mat_coord_value {
            const std::size_t cols;
            std::size_t offset;
            __dense_mat_coord_value() = delete;
            __dense_mat_coord_value( std::size_t _cols, std::size_t _off ):
                cols( _cols ), offset( _off ) {}
        };

        // simple iterator returning an incremental number
        // and generating a rectangular dense matrix
        template< typename ValT, bool random > struct dense_mat_iterator:
            public std::iterator<
                typename std::conditional< random, std::random_access_iterator_tag,
                std::forward_iterator_tag >::type,
                __dense_mat_coord_value, long, __dense_mat_coord_value*, __dense_mat_coord_value& > {

            using row_coordinate_type = std::size_t;
            using column_coordinate_type = std::size_t;
            using nonzero_value_type = ValT;
            using self_t = dense_mat_iterator< ValT, random >;
            using input_sizes_t = const std::array< std::size_t, 2 >;

            dense_mat_iterator( std::size_t _cols, std::size_t _off ): _v( _cols, _off ) {}

            dense_mat_iterator( const self_t& ) = default;

            self_t& operator++() {
                _v.offset++;
                return *this;
            }

            self_t& operator+=( std::size_t offset ) {
                _v.offset += offset;
                return *this;
            }

            bool operator!=( const self_t& other ) const {
                return other._v.offset != this->_v.offset;
            }

            bool operator==( const dense_mat_iterator& other ) const {
                return !( this->operator!=( other ) );
            }

            typename self_t::difference_type operator-( const self_t& other ) const {
				return __compute_distance< std::size_t, typename self_t::difference_type >(
					this->_v.offset, other._v.offset );
            }

            typename self_t::pointer operator->() { return &_v; }

            typename self_t::reference operator*() { return _v; }

            row_coordinate_type i() const { return _v.offset / _v.cols; }

            column_coordinate_type j() const { return _v.offset % _v.cols; }

            nonzero_value_type v() const {
                return static_cast< nonzero_value_type >( _v.offset ) + 1;
            }


            static self_t make_begin( input_sizes_t& sizes ) {
                return self_t( sizes[1], 0 );
            }

            static self_t make_end( input_sizes_t& sizes ) {
                const std::size_t num_nonzeroes{ compute_num_nonzeroes( sizes ) };
                return self_t( sizes[1], num_nonzeroes );
            }

            static self_t make_parallel_begin( input_sizes_t& sizes ) {
                std::size_t num_non_zeroes_per_process, first_local_nonzero;
                compute_parallel_first_nonzero( compute_num_nonzeroes( sizes ), num_non_zeroes_per_process, first_local_nonzero );
                return self_t( sizes[1], first_local_nonzero );
            }

            static self_t make_parallel_end( input_sizes_t& sizes ) {
                std::size_t last{ compute_parallel_last_nonzero( compute_num_nonzeroes( sizes ) ) };
                return self_t( sizes[1], last );
            }

            static std::size_t compute_num_nonzeroes( input_sizes_t& sizes ) {
                return sizes[0] * sizes[1];
            }

        private:
            typename self_t::value_type _v;
        };

    }
}

#endif // _GRB_UTILS_MATRIX_GENERATORS_

