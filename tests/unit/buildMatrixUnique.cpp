
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

#include <iostream>
#include <cstddef>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <array>
#include <vector>
#include <iterator>

#include <graphblas/iomode.hpp>

#define _GRB_BUILD_MATRIX_UNIQUE_TRACE

void __trace_build_matrix_iomode( grb::IOMode );

#include <graphblas.hpp>
#include <utils/assertions.hpp>
#include <graphblas/utils/NonZeroStorage.hpp>

using namespace grb;

static std::vector< grb::IOMode > __iomodes;

void __trace_build_matrix_iomode( grb::IOMode mode ) {
	__iomodes.push_back( mode );
}



#define LOG() std::cout
const char * MISMATCH_HIGHLIGHT_BEGIN{ "<< " };
const char * MISMATCH_HIGHLIGHT_END{ " >>" };
const char * NO_MISMATCH{ "" };


static void compute_parallel_first_nonzero( std::size_t num_nonzeroes,
	std::size_t& num_nonzeroes_per_process, std::size_t& first_local_nonzero ) {
	std::size_t num_procs { spmd<>::nprocs() };
	num_nonzeroes_per_process = ( num_nonzeroes + num_procs - 1 ) / num_procs; // round up
	first_local_nonzero = num_nonzeroes_per_process * spmd<>::pid();
}

static std::size_t compute_parallel_first_nonzero( std::size_t num_nonzeroes ) {
	std::size_t nnz_per_proc, first;
	compute_parallel_first_nonzero( num_nonzeroes, nnz_per_proc, first);
	return first;
}

static std::size_t compute_parallel_last_nonzero( std::size_t num_nonzeroes ) {
	std::size_t num_non_zeroes_per_process, first_local_nonzero;
	compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
	return std::min( num_nonzeroes, first_local_nonzero + num_non_zeroes_per_process );
}

static std::size_t compute_parallel_num_nonzeroes( std::size_t num_nonzereos ) {
	return compute_parallel_last_nonzero( num_nonzereos ) -
		compute_parallel_first_nonzero( num_nonzereos );
}

struct diag_iterator {

	using row_coordinate_type = std::size_t;
    using column_coordinate_type = std::size_t;
    using nonzero_value_type = int;

    struct coord_value {
		std::size_t coord;

		coord_value( std::size_t _c ): coord( _c ) {}
	};

	using iterator_category = std::random_access_iterator_tag;
	using value_type = coord_value;
	using pointer = coord_value*;
	using reference = coord_value&;
	using difference_type = signed long;
	using input_sizes_t = const std::size_t;

    diag_iterator( const diag_iterator& ) = default;

    diag_iterator& operator++() {
		_v.coord++;
		return *this;
	}

    diag_iterator& operator+=( std::size_t offset ) {
		_v.coord += offset;
		return *this;
	}

	bool operator!=( const diag_iterator& other ) const {
        return other._v.coord != this->_v.coord;
    }

    bool operator==( const diag_iterator& other ) const {
        return !( this->operator!=( other ) );
    }

    difference_type operator-( const diag_iterator& other ) const {
		const std::size_t diff{ std::max( _v.coord, other._v.coord ) -
			std::min( _v.coord, other._v.coord ) };
		difference_type result{ static_cast< difference_type >( diff ) };
        return _v.coord >= other._v.coord ? result : -result ;
    }

	pointer operator->() { return &_v; }

    reference operator*() { return _v; }

    row_coordinate_type i() const { return _v.coord; }

    column_coordinate_type j() const { return _v.coord; }

    nonzero_value_type v() const {
		return static_cast< nonzero_value_type >( _v.coord ) + 1;
	}

    static diag_iterator make_begin( input_sizes_t& size ) {
		(void)size;
		return diag_iterator( 0 );
	}

    static diag_iterator make_end( input_sizes_t& size ) {
		return diag_iterator( size );
	}

    static diag_iterator make_parallel_begin( input_sizes_t& size ) {
		const std::size_t num_nonzeroes{ size };
        std::size_t num_non_zeroes_per_process, first_local_nonzero;
		compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
        return diag_iterator( first_local_nonzero );
    }

    static diag_iterator make_parallel_end( input_sizes_t& size ) {
		const std::size_t num_nonzeroes{ size };
		std::size_t last{ compute_parallel_last_nonzero( num_nonzeroes ) };
        return diag_iterator( last );
    }

	static std::size_t compute_num_nonzeroes( std::size_t size ) {
		return size;
	}

private:
    value_type _v;

    diag_iterator( std::size_t _c ): _v( _c ) {}

    diag_iterator(): _v( 0) {}

};


template< std::size_t BAND > struct band_iterator {

	static constexpr std::size_t MAX_ELEMENTS_PER_ROW{ BAND * 2 + 1 };
	static constexpr std::size_t PROLOGUE_ELEMENTS{ ( 3* BAND * BAND + BAND ) / 2 };

    using row_coordinate_type = std::size_t;
    using column_coordinate_type = std::size_t;
    using nonzero_value_type = int;

    struct coord_value {
		const std::size_t size;
        row_coordinate_type row;
		column_coordinate_type col;
        coord_value() = delete;
        coord_value( std::size_t _size, row_coordinate_type _r, column_coordinate_type _c ):
			size( _size ), row( _r ), col( _c ) {}
    };

    using iterator_category = std::random_access_iterator_tag;
	using value_type = coord_value;
	using pointer = coord_value*;
	using reference = coord_value&;
	using difference_type = signed long;
	using input_sizes_t = const std::size_t;

    band_iterator( const band_iterator& ) = default;

    band_iterator& operator++() {
		const std::size_t max_col{ std::min( _v.row + BAND, _v.size - 1 ) };
		if( _v.col < max_col ) {
			_v.col++;
		} else {
			_v.row++;
			_v.col = _v.row < BAND ? 0 : _v.row - BAND;
		}
        return *this;
    }

    band_iterator& operator+=( std::size_t offset ) {

		/*
		#pragma omp critical
		{
			std::cout << "INCREMENT offset " << offset << std::endl;
			//std::cout << "-- i " << _v.row << " j " << _v.col << " v " << std::endl;
			const std::size_t position{ coords_to_linear( _v.size, _v.row, _v.col ) };
			linear_to_coords( _v.size, position + offset, _v.row, _v.col );
			std::cout << "++ i " << _v.row << " j " << _v.col << " v " << std::endl;
		}
		*/
		const std::size_t position{ coords_to_linear( _v.size, _v.row, _v.col ) };
		linear_to_coords( _v.size, position + offset, _v.row, _v.col );
        return *this;
    }

    bool operator!=( const band_iterator& other ) const {
        return other._v.row != this->_v.row || other._v.col != this->_v.col;
    }

    bool operator==( const band_iterator& other ) const {
        return !( this->operator!=( other ) );
    }

    difference_type operator-( const band_iterator& other ) const {
        const std::size_t this_position{ coords_to_linear( _v.size, _v.row, _v.col ) };
        const std::size_t other_position{
			coords_to_linear( other._v.size, other._v.row, other._v.col ) };
		/*
		std::cout << " this:: " << _v.size << " " <<  _v.row << " " << _v.col << std::endl;
		std::cout << " other:: " << other._v.size << " " <<  other._v.row << " " << other._v.col << std::endl;
		std::cout << " this position " << this_position << std::endl;
		std::cout << " other position " << other_position << std::endl;
		*/

		const std::size_t diff{ std::max( this_position, other_position ) -
			std::min( this_position, other_position ) };
		difference_type result{ static_cast< difference_type >( diff ) };
        return this_position >= other_position ? result : -result ;
    }

    pointer operator->() { return &_v; }

    reference operator*() { return _v; }

    row_coordinate_type i() const { return _v.row; }

    column_coordinate_type j() const { return _v.col; }

    nonzero_value_type v() const {
		return _v.row == _v.col ? static_cast< int >( MAX_ELEMENTS_PER_ROW ) : -1;
	}

    static band_iterator make_begin( input_sizes_t& size ) {
		__check_size( size );
		return band_iterator( size, 0, 0 );
	}

    static band_iterator make_end( input_sizes_t& size ) {
		__check_size( size );
		std::size_t row, col;
		const std::size_t num_nonzeroes{ compute_num_nonzeroes( size ) };
		linear_to_coords( size, num_nonzeroes, row, col );
		//std::cout << "make_end: nnz " << num_nonzeroes << ", row " << row << ", col " << col << std::endl;
		return band_iterator( size, row, col );
	}

    static band_iterator make_parallel_begin( input_sizes_t& size ) {
		__check_size( size );
		const std::size_t num_nonzeroes{ compute_num_nonzeroes( size ) };
        std::size_t num_non_zeroes_per_process, first_local_nonzero;
		compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process, first_local_nonzero );
		std::size_t row, col;
		linear_to_coords( size, first_local_nonzero, row, col );
        return band_iterator( size, row, col );
    }

    static band_iterator make_parallel_end( input_sizes_t& size ) {
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
        return band_iterator( size, row, col );
    }

	static std::size_t compute_num_nonzeroes( std::size_t size ) {
		return 2 * PROLOGUE_ELEMENTS + ( size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW;
	}

private:
    value_type _v;

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
			return 2 * PROLOGUE_ELEMENTS + ( row - 2 * BAND ) * MAX_ELEMENTS_PER_ROW
				- __coords_to_linear_in_prologue( matrix_size - col, matrix_size - row ); // transpose coordinates
		}
		// for points outside of matrix: project to prologue
		return 2 * PROLOGUE_ELEMENTS + ( row - 2 * BAND ) * MAX_ELEMENTS_PER_ROW
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

	static void __linear_to_coords_in_epilogue( std::size_t position, std::size_t& row, std::size_t& col ) {
		std::size_t offset_row{ 0 };
		//linear search
		for( ; position >= ( 2 * BAND - offset_row ) && offset_row <= BAND; offset_row++ ) {
			position -= ( 2 * BAND - offset_row );
			//std::cout << "subtract " << ( 2 * BAND - offset_row ) << " get " << position << std::endl;
		}
		row = offset_row;
		col = position + offset_row;
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
		std::size_t end_row, end_col;

		__linear_to_coords_in_epilogue( position, end_row, end_col );
		//std::cout << "== position " << position << ", end row " << end_row << ", end col " << end_col << std::endl;
		row = matrix_size - BAND + end_row;
		col = matrix_size - 2 * BAND + end_col;
		//std::cout << "final: position " << position << ", row " << row << ", col " << col << std::endl;
	}

	static void __check_size( std::size_t size ) {
		if( size < 2 * BAND + 1 ) {
			throw std::domain_error( "matrix too small for band" );
		}
	}
};


// simple iterator returning an incremental number
// and generating a rectangular dense matrix
template< typename ValueT = int > struct dense_mat_iterator {

	using row_coordinate_type = std::size_t;
    using column_coordinate_type = std::size_t;
    using nonzero_value_type = ValueT;

    struct coord_value {
		const std::size_t cols;
        row_coordinate_type offset;
        coord_value() = delete;
        coord_value( std::size_t _cols, row_coordinate_type _off ):
			cols( _cols ), offset( _off ) {}
    };

    using iterator_category = std::random_access_iterator_tag;
	using value_type = coord_value;
	using pointer = coord_value*;
	using reference = coord_value&;
	using difference_type = signed long;
	using input_sizes_t = const std::array< std::size_t, 2 >;

	dense_mat_iterator( std::size_t _cols, std::size_t _off ): _v( _cols, _off ) {}

    dense_mat_iterator( const dense_mat_iterator& ) = default;

    dense_mat_iterator& operator++() {
		_v.offset++;
		return *this;
	}

    dense_mat_iterator& operator+=( std::size_t offset ) {
		_v.offset += offset;
        return *this;
    }

    bool operator!=( const dense_mat_iterator& other ) const {
        return other._v.offset != this->_v.offset;
    }

    bool operator==( const dense_mat_iterator& other ) const {
        return !( this->operator!=( other ) );
    }

    difference_type operator-( const dense_mat_iterator& other ) const {
        const std::size_t this_position{ this->_v.offset };
        const std::size_t other_position{other._v.offset };
		const std::size_t diff{ std::max( this_position, other_position ) -
			std::min( this_position, other_position ) };
		difference_type result{ static_cast< difference_type >( diff ) };
        return this_position >= other_position ? result : -result ;
    }

    pointer operator->() { return &_v; }

    reference operator*() { return _v; }

    row_coordinate_type i() const { return _v.offset / _v.cols; }

    column_coordinate_type j() const { return _v.offset % _v.cols; }

    nonzero_value_type v() const {
		return static_cast< nonzero_value_type >( _v.offset ) + 1;
	}


    static dense_mat_iterator make_begin( input_sizes_t& sizes ) {
		return dense_mat_iterator( sizes[1], 0 );
	}

    static dense_mat_iterator make_end( input_sizes_t& sizes ) {
		const std::size_t num_nonzeroes{ compute_num_nonzeroes( sizes ) };
		return dense_mat_iterator( sizes[1], num_nonzeroes );
	}

    static dense_mat_iterator make_parallel_begin( input_sizes_t& sizes ) {
        std::size_t num_non_zeroes_per_process, first_local_nonzero;
		compute_parallel_first_nonzero( compute_num_nonzeroes( sizes ), num_non_zeroes_per_process, first_local_nonzero );
        return dense_mat_iterator( sizes[1], first_local_nonzero );
    }

    static dense_mat_iterator make_parallel_end( input_sizes_t& sizes ) {
		std::size_t last{ compute_parallel_last_nonzero( compute_num_nonzeroes( sizes ) ) };
        return dense_mat_iterator( sizes[1], last );
    }

	static std::size_t compute_num_nonzeroes( input_sizes_t& sizes ) {
		return sizes[0] * sizes[1];
	}

private:
    value_type _v;
};


template< typename T > void test_matrix_sizes_match( const Matrix< T >& mat1, const Matrix< T >& mat2) {
    ASSERT_EQ( grb::nrows( mat1 ), grb::nrows( mat2 ) );
    ASSERT_EQ( grb::ncols( mat1 ), grb::ncols( mat2 ) );
}

template< typename T > struct sorter {
	inline bool operator()( const utils::NonZeroStorage< std::size_t, std::size_t, T >& a,
		const utils::NonZeroStorage< std::size_t, std::size_t, T >& b ) const {
			if( a.i() != b.i() ) {
				return a.i() < b.i();
			}
			return a.j() < b.j();
		}
};


template< typename T > bool matrices_values_are_equal( const Matrix< T >& mat1, const Matrix< T >& mat2,
    bool log_all_differences = false ) {

    auto beg1( mat1.cbegin() );
    auto end1( mat1.cend() );
	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, int > > serial_values;
	for( ; beg1 != end1; ++beg1 ) {
		serial_values.emplace_back( beg1->first.first, beg1->first.second, beg1->second );
	}

    auto beg2( mat2.cbegin() );
    auto end2( mat2.cend() );
	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, int > > parallel_values;
	for( ; beg2 != end2; ++beg2 ) {
		parallel_values.emplace_back( beg2->first.first, beg2->first.second, beg2->second );
	}

	sorter< int > s;
	std::sort( serial_values.begin(), serial_values.end(), s );
	std::sort( parallel_values.begin(), parallel_values.end(), s );

    const std::size_t mat_size{ grb::nnz( mat1) };

	if( serial_values.size() != parallel_values.size() ) {
		LOG() << "the numbers of entries differ" << std::endl;
        return false;
	}

	if( serial_values.size() != mat_size ) {
		LOG() << "different number of non-zeroes: actual: " << serial_values.size()
			<< ", expected: " << mat_size << std::endl;
		return false;
	}

    bool match{ true };

	auto pit = parallel_values.cbegin();
	for( auto sit = serial_values.cbegin();
		sit != serial_values.cend(); ++sit, ++pit ) {

		const std::size_t row1{ sit->i() }, row2{ pit->i() };
        const std::size_t col1{ sit->j() }, col2{ pit->j() };
        const T val1{ sit->v() }, val2{ pit->v() };

        const bool row_mismatch{ row1 != row2  }, col_mismatch{ col1 != col2 }, v_mismatch{ val1 != val2 };
        const bool mismatch{ row_mismatch || col_mismatch || v_mismatch };
        match &= ! mismatch;

        if( mismatch ) {
			LOG() << "row ";
			if( row_mismatch ) {
				LOG() << MISMATCH_HIGHLIGHT_BEGIN << row1 << ", " << row2 << MISMATCH_HIGHLIGHT_END;
			} else {
				LOG() << row1;
			}
			LOG() << " col ";
			if( col_mismatch ) {
				LOG() << MISMATCH_HIGHLIGHT_BEGIN << col1 << ", " << col2 << MISMATCH_HIGHLIGHT_END;
			} else {
				LOG() << col1;
			}
			LOG() << " val ";
			if( v_mismatch ) {
				LOG() << MISMATCH_HIGHLIGHT_BEGIN << val1 << ", " << val2 << MISMATCH_HIGHLIGHT_END;
			} else {
				LOG() << val1;
			}
			LOG() << std::endl;

            if( ! log_all_differences ){
                return false;
            }
        } else {
            //LOG() << "row " << row1 << " col " << col1 << " val " << val1 << std::endl;
        }
	}

    return match;
}

static bool test_build_matrix_iomode( IOMode mode ) {
	bool res{ true };
	for( IOMode m : __iomodes ) {
		res &= ( m == mode );
	}
	__iomodes.clear();
	return res;
}


template< typename T, typename IterT, enum Backend implementation = config::default_backend >
	void test_sequential_and_parallel_matrix_generation(
		std::size_t nrows, std::size_t ncols,
		const typename IterT::input_sizes_t& iter_sizes ) {

    LOG() << "-- size " << nrows << " x " << ncols << std::endl;

	try {

		Matrix< T, implementation > sequential_matrix( nrows, ncols );
		IterT begin( IterT::make_begin( iter_sizes) );
		IterT end( IterT::make_end( iter_sizes) );
		const std::size_t num_nnz{ IterT::compute_num_nonzeroes( iter_sizes) };
		//std::cout << " with " << num_nnz << " non-zeroes" << std::endl;
		// test that iterator difference works properly
		ASSERT_EQ( end  - begin, static_cast< typename IterT::difference_type >( num_nnz ) );
		RC ret { buildMatrixUnique( sequential_matrix, begin, end, IOMode::SEQUENTIAL ) };
		ASSERT_RC_SUCCESS( ret );
		ASSERT_TRUE( test_build_matrix_iomode( IOMode::SEQUENTIAL ) );
		ASSERT_EQ( nnz( sequential_matrix ), num_nnz );

		Matrix< T, implementation > parallel_matrix( nrows, ncols );
		IterT pbegin( IterT::make_parallel_begin( iter_sizes) );
		IterT pend( IterT::make_parallel_end( iter_sizes) );
		ASSERT_EQ( pend  - pbegin, static_cast< typename IterT::difference_type >(
				compute_parallel_num_nonzeroes( num_nnz ) ) );
		ret = buildMatrixUnique( parallel_matrix, pbegin, pend, IOMode::PARALLEL );
		ASSERT_RC_SUCCESS( ret );
		// allow SEQUENTIAL iomode with sequential-only backends
		ASSERT_TRUE( test_build_matrix_iomode( IOMode::PARALLEL )
			|| ( implementation == Backend::BSP1D ) || ( implementation == Backend::reference ) );
		//std::cout << "expected non-zeroes " << num_nnz << " actual " << nnz( parallel_matrix ) << std::endl;
		ASSERT_EQ( nnz( parallel_matrix ), num_nnz );

		test_matrix_sizes_match( sequential_matrix, parallel_matrix );
		ASSERT_TRUE( matrices_values_are_equal( sequential_matrix, parallel_matrix, true ) );
	} catch ( std::exception& e ) {
		LOG() << "got exception: " << std::endl << "---------" << std::endl << "   " << e.what()
			<<  std::endl << "---------" << std::endl;
	}

	LOG() << "-- OK" << std::endl;
}

void grbProgram( const void *, const size_t, int & ) {

	std::initializer_list< std::size_t > diag_sizes{ spmd<>::nprocs(), spmd<>::nprocs() + 9,
		spmd<>::nprocs() + 16, 100003 };

	LOG() << "== Testing diagonal matrices" << std::endl;
    for( const std::size_t& mat_size : diag_sizes ) {
        test_sequential_and_parallel_matrix_generation< int, diag_iterator >( mat_size, mat_size, mat_size );
		//return;
    }

	std::initializer_list< std::size_t > band_sizes{ 17, 77, 107, 11467 };

   	for( const std::size_t& mat_size : band_sizes ) {
		LOG() << "== Testing matrix with band 1" << std::endl;
        test_sequential_and_parallel_matrix_generation< int, band_iterator< 1 > >(
			mat_size, mat_size, mat_size );
		LOG() << "== Testing matrix with band 2" << std::endl;
        test_sequential_and_parallel_matrix_generation< int, band_iterator< 2 > >(
			mat_size, mat_size, mat_size );
		LOG() << "== Testing matrix with band 7" << std::endl;
        test_sequential_and_parallel_matrix_generation< int, band_iterator< 7 > >(
			mat_size, mat_size, mat_size );
    }

	std::initializer_list< std::array< std::size_t, 2 > > matr_sizes{
		{ spmd<>::nprocs(), spmd<>::nprocs() }, { 77, 77 }, { 139, 139}
	};

	for( const std::array< std::size_t, 2 >& mat_size : matr_sizes ) {
		test_sequential_and_parallel_matrix_generation< int, dense_mat_iterator< int > >(
			mat_size[0], mat_size[1], mat_size );
	}


}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int error{0};
	Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, NULL, 0, error ) != SUCCESS ) {
		std::cout << "Test FAILED (test failed to launch)" << std::endl;
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cout << "Test FAILED" << std::endl;
	}

	return error;
}
