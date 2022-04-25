
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
#include <utility>
#include <unordered_map>
#include <cassert>
#include <sstream>
#include <type_traits>

#include <graphblas/backends.hpp>

#define _GRB_BUILD_MATRIX_UNIQUE_TRACE

void __trace_build_matrix_iomode( grb::Backend, bool );

#include <graphblas.hpp>
#include <utils/assertions.hpp>
#include <graphblas/utils/NonZeroStorage.hpp>

using namespace grb;

static std::vector< std::pair< Backend, bool > > __iomodes;

void __trace_build_matrix_iomode( grb::Backend backend, bool iterator_parallel ) {
	__iomodes.emplace_back( backend, iterator_parallel );
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

template< bool random = true > struct diag_iterator {

	using row_coordinate_type = std::size_t;
    using column_coordinate_type = std::size_t;
    using nonzero_value_type = int;

    struct coord_value {
		std::size_t coord;

		coord_value( std::size_t _c ): coord( _c ) {}
	};

	using iterator_category = typename std::conditional< random,
		std::random_access_iterator_tag, std::forward_iterator_tag >::type;
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


template< std::size_t BAND, bool random = true > struct band_iterator {

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

    using iterator_category = typename std::conditional< random,
		std::random_access_iterator_tag, std::forward_iterator_tag >::type;
	using value_type = coord_value;
	using pointer = coord_value*;
	using reference = coord_value&;
	using difference_type = signed long;
	using input_sizes_t = const std::size_t;

    band_iterator( const band_iterator& ) = default;

    band_iterator& operator++() {
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

    band_iterator& operator+=( std::size_t offset ) {

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


// simple iterator returning an incremental number
// and generating a rectangular dense matrix
template< typename ValueT = int, bool random = true  > struct dense_mat_iterator {

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


template< typename T > static void get_nnz_and_sort( const Matrix< T >& mat,
	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, T > >& values ) {
	auto beg1( mat.cbegin() );
    auto end1( mat.cend() );
	for( ; beg1 != end1; ++beg1 ) {
		values.emplace_back( beg1->first.first, beg1->first.second, beg1->second );
	}
	sorter< T > s;
	std::sort( values.begin(), values.end(), s );
}


template< typename T, enum Backend implementation >
	bool matrices_values_are_equal( const Matrix< T, implementation >& mat1,
	const Matrix< T, implementation >& mat2,
    bool log_all_differences = false ) {

	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, T > > serial_values;
	get_nnz_and_sort( mat1, serial_values );

	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, T > > parallel_values;
	get_nnz_and_sort( mat2, parallel_values );

    const std::size_t mat_size{ grb::nnz( mat1) };

	if( serial_values.size() != parallel_values.size() ) {
		LOG() << "the numbers of entries differ" << std::endl;
        return false;
	}

	if( serial_values.size() != mat_size && implementation != Backend::BSP1D ) {
		LOG() << "different number of non-zeroes: actual: " << serial_values.size()
			<< ", expected: " << mat_size << std::endl;
		return false;
	}

    bool match{ true };

	auto pit = parallel_values.cbegin();
	for( auto sit = serial_values.cbegin(); sit != serial_values.cend(); ++sit, ++pit ) {

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

using iomode_map_t = std::unordered_map< enum Backend, bool >;

static bool test_build_matrix_iomode( const iomode_map_t& iomap ) {
	bool res{ true };
	for( const std::pair< enum Backend, bool >& m : __iomodes ) {
		typename iomode_map_t::const_iterator pos{ iomap.find( m.first ) };
		if( pos == iomap.cend() ) {
			FAIL();
		}
		ASSERT_EQ( m.second, pos->second );
		res &= m.second == pos->second;
	}
	__iomodes.clear();
	return res;
}

template< typename T, typename IterT, enum Backend implementation = config::default_backend >
	void build_matrix_and_check(Matrix< T, implementation >& m, IterT begin, IterT end,
	std::size_t expected_num_global_nnz, std::size_t expected_num_local_nnz, IOMode mode ) {

	ASSERT_EQ( end - begin, static_cast< typename IterT::difference_type >( expected_num_local_nnz ) );

	RC ret { buildMatrixUnique( m, begin, end, mode ) };
	ASSERT_RC_SUCCESS( ret );
	ASSERT_EQ( nnz( m ), expected_num_global_nnz );
}


template< typename T, typename IterT, enum Backend implementation >
	void test_matrix_generation( Matrix< T, implementation > & sequential_matrix,
		Matrix< T, implementation > parallel_matrix,
		const typename IterT::input_sizes_t& iter_sizes ) {

	constexpr bool iterator_is_random{ std::is_same< typename std::iterator_traits<IterT>::iterator_category,
		std::random_access_iterator_tag>::value };
	iomode_map_t iomap( {
		std::pair< enum Backend, bool >( implementation, iterator_is_random
			&& ( ( implementation == Backend::reference_omp )
#ifdef _GRB_BSP1D_BACKEND
			|| (implementation == Backend::BSP1D && _GRB_BSP1D_BACKEND == Backend::reference_omp )
#endif
			) )
#ifdef _GRB_BSP1D_BACKEND
		, std::pair< enum Backend, bool >( _GRB_BSP1D_BACKEND,
			( _GRB_BSP1D_BACKEND == Backend::reference_omp ) )
#endif
	} );

	if( spmd<>::pid() == 0 ) {
		LOG() << ">> " << ( iterator_is_random ? "RANDOM" : "FORWARD" ) << " ITERATOR ";
		LOG() << "-- size " << nrows( sequential_matrix ) << " x "
			<< ncols( sequential_matrix ) << std::endl;
	}

	//Matrix< T, implementation > sequential_matrix( nrows, ncols );
	const std::size_t num_nnz{ IterT::compute_num_nonzeroes( iter_sizes) };
	build_matrix_and_check( sequential_matrix, IterT::make_begin( iter_sizes),
		IterT::make_end( iter_sizes), num_nnz, num_nnz, IOMode::SEQUENTIAL );
	ASSERT_TRUE( test_build_matrix_iomode( iomap ) );

	//Matrix< T, implementation > parallel_matrix( nrows, ncols );
	const std::size_t par_num_nnz{ compute_parallel_num_nonzeroes( num_nnz ) };

	build_matrix_and_check( parallel_matrix, IterT::make_parallel_begin( iter_sizes),
		IterT::make_parallel_end( iter_sizes), num_nnz, par_num_nnz, IOMode::PARALLEL );
	ASSERT_TRUE( test_build_matrix_iomode( iomap ) );

	test_matrix_sizes_match( sequential_matrix, parallel_matrix );
	ASSERT_TRUE( matrices_values_are_equal( sequential_matrix, parallel_matrix, false ) );

	if( spmd<>::pid() == 0 ) {
		LOG() << "<< OK" << std::endl;
	}
}

template< typename T, typename ParIterT, typename SeqIterT, enum Backend implementation = config::default_backend >
	void test_sequential_and_parallel_matrix_generation(
		std::size_t nrows, std::size_t ncols,
		const typename ParIterT::input_sizes_t& iter_sizes ) {

		Matrix< T, implementation > par_sequential_matrix( nrows, ncols );
		Matrix< T, implementation > par_parallel_matrix( nrows, ncols );
		test_matrix_generation< T, ParIterT, implementation >( par_sequential_matrix, par_parallel_matrix, iter_sizes );

		Matrix< T, implementation > seq_sequential_matrix( nrows, ncols );
		Matrix< T, implementation > seq_parallel_matrix( nrows, ncols );
		test_matrix_generation< T, SeqIterT, implementation >( seq_sequential_matrix, seq_parallel_matrix, iter_sizes );

		// cross-check parallel vs sequential
		ASSERT_TRUE( matrices_values_are_equal( par_parallel_matrix, seq_parallel_matrix, false ) );
}

template< enum Backend implementation = config::default_backend > void test_matrix_from_vectors() {
	constexpr std::size_t mat_size{ 7 };
	std::vector< std::size_t > rows{ 0, 0, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5 };
	std::vector< std::size_t > cols{ 1, 3, 3, 2, 4, 0, 2, 0, 1, 2, 3, 4, 5 };
	std::vector< int > values{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
	assert( rows.size() == cols.size() && rows.size() == values.size() );

	Matrix< int, implementation > mat( mat_size, mat_size );
	const std::size_t per_proc{ ( values.size() + spmd<>::nprocs() - 1 ) / spmd<>::nprocs() };
	const std::size_t first{ per_proc * spmd<>::pid() };
	const std::size_t num_local{ std::min( per_proc, values.size() - first ) };

#ifdef _DEBUG
	for( unsigned i{ 0 }; i < spmd<>::nprocs(); i++) {
		if( spmd<>::pid() == i ) {
			std::cout << "process " << i << " from " << first << " num " << num_local << std::endl;
		}
		spmd<>::barrier();
	}
#endif

	RC ret{ buildMatrixUnique( mat, rows.data() + first, cols.data() + first,
		values.data() + first, num_local, IOMode::PARALLEL ) };

	ASSERT_RC_SUCCESS( ret );

	std::vector< utils::NonZeroStorage< std::size_t, std::size_t, int > > sorted_values;
	get_nnz_and_sort( mat, sorted_values );

	const std::size_t pid{ spmd<>::pid() }, nprocs{ spmd<>::nprocs() };
	const std::size_t local_nnz{ static_cast< std::size_t > ( std::count_if(
		rows.cbegin(), rows.cend(), [pid,mat_size,nprocs]( std::size_t row ) {
				return internal::Distribution< implementation >::global_index_to_process_id(
					row, mat_size, nprocs
				) == pid;
			}
		)
	) };

#ifdef _DEBUG
	for( unsigned i{ 0 }; i < spmd<>::nprocs(); i++) {
		if( spmd<>::pid() == i ) {
			std::cout << "process " << i << " local count " << local_nnz << std::endl;
		}
		spmd<>::barrier();
	}
#endif


	ASSERT_EQ( sorted_values.size(), local_nnz );
	std::size_t k{ 0 };
	for( auto it{ sorted_values.cbegin() }; it != sorted_values.cend() && k < values.size(); ++it, ++k ) {
		ASSERT_EQ( it->i(), rows[ k ] );
		ASSERT_EQ( it->j(), cols[ k ] );
		ASSERT_EQ( it->v(), values[ k ] );
	}
	ASSERT_EQ( k, local_nnz );

	if( spmd<>::pid() == 0 ) {
		LOG() << "-- OK" << std::endl;
	}
}

static const char* const std_caption{ "got exception: " };

static void print_exception_text( const char * text, const char * caption = std_caption ) {
	std::stringstream stream;
	if( spmd<>::nprocs() > 1UL ) {
		stream << "Machine " << spmd<>::pid() << " - ";
	}
	stream << caption << std::endl
		<< ">>>>>>>>" << std::endl
		<< text << std::endl
		<< "<<<<<<<<" << std::endl;
	LOG() << stream.str();
}

void grbProgram( const void *, const size_t, int &error ) {

	try {

		std::initializer_list< std::size_t > diag_sizes{ spmd<>::nprocs(), spmd<>::nprocs() + 9,
			spmd<>::nprocs() + 16, 100003 };

		if( spmd<>::pid() == 0 ) {
			LOG() << "==== Testing diagonal matrices" << std::endl;
		}
		for( const std::size_t& mat_size : diag_sizes ) {
			test_sequential_and_parallel_matrix_generation< int, diag_iterator< true >,diag_iterator< false > >(
				mat_size, mat_size, mat_size );
		}

		std::initializer_list< std::size_t > band_sizes{ 17, 77, 107, 11467 };

		for( const std::size_t& mat_size : band_sizes ) {
			if( spmd<>::pid() == 0 ) {
				LOG() << "==== Testing matrix with band 1" << std::endl;
			}
			test_sequential_and_parallel_matrix_generation< int, band_iterator< 1, true >, band_iterator< 1, false > >(
				mat_size, mat_size, mat_size );

			if( spmd<>::pid() == 0 ) {
				LOG() << "==== Testing matrix with band 2" << std::endl;
			}
			test_sequential_and_parallel_matrix_generation< int, band_iterator< 2, true >, band_iterator< 2, false > >(
				mat_size, mat_size, mat_size );

			if( spmd<>::pid() == 0 ) {
				LOG() << "==== Testing matrix with band 7" << std::endl;
			}
			test_sequential_and_parallel_matrix_generation< int, band_iterator< 7, true >, band_iterator< 7, false > >(
				mat_size, mat_size, mat_size );
		}

		std::initializer_list< std::array< std::size_t, 2 > > matr_sizes{
			{ spmd<>::nprocs(), spmd<>::nprocs() }, { 77, 70 }, { 130, 139 }
		};

		if( spmd<>::pid() == 0 ) {
			LOG() << "==== Testing dense matrices" << std::endl;
		}
		for( const std::array< std::size_t, 2 >& mat_size : matr_sizes ) {
			test_sequential_and_parallel_matrix_generation< int, dense_mat_iterator< int, true >, dense_mat_iterator< int, false > >(
				mat_size[0], mat_size[1], mat_size );
		}

		if( spmd<>::pid() == 0 ) {
			LOG() << "==== Testing sparse matrix from user's vectors" << std::endl;
		}
		test_matrix_from_vectors();

	} catch ( const std::exception& e ) {
		print_exception_text( e.what() );
		error = 1;
	} catch( ... ) {
		LOG() << "unknown exception" <<std::endl;
		error = 1;
	}
}

int main( int argc, char ** argv ) {
	(void)argc;
	std::cout << "Functional test executable: " << argv[ 0 ] << std::endl;

	int error{ 0 };

	Launcher< AUTOMATIC > launcher;
	if( launcher.exec( &grbProgram, nullptr, 0, error ) != SUCCESS ) {
		std::cout << "Could not launch test" << std::endl;
		error = 255;
	}
	if( error == 0 ) {
		std::cout << "Test OK" << std::endl;
	} else {
		std::cout << "Test FAILED" << std::endl;
	}

	return error;
}

