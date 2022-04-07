
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

#include <graphblas.hpp>
#include <utils/assertions.hpp>

using namespace grb;

#define LOG() std::cout
const char * MISMATCH_HIGHLIGHT_BEGIN{ "<< " };
const char * MISMATCH_HIGHLIGHT_END{ " >>" };
const char * NO_MISMATCH{ " >>" };

struct diag_iterator {

    using row_coordinate_type = std::size_t;
    using column_coordinate_type = std::size_t;
    using nonzero_value_type = int;

    struct coord_value {
        row_coordinate_type coord;

        coord_value() = delete;

        coord_value( row_coordinate_type _c ): coord( _c ) {}

        row_coordinate_type i() const {
            return coord;
        }

        column_coordinate_type j() const {
            return coord;
        }

        nonzero_value_type v() const {
            return static_cast< nonzero_value_type >( coord );
        }
    };

    using iterator_category = std::random_access_iterator_tag;
	using value_type = coord_value;
	using pointer = coord_value*;
	using reference = coord_value&;
	using difference_type = signed long;


    value_type _v;

    diag_iterator() = delete;

    diag_iterator( std::size_t position ): _v( position ) {}

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
        return this->_v.coord - other._v.coord;
    }

    pointer operator->() {
        return &_v;
    }

    reference operator*() {
        return _v;
    }

    row_coordinate_type i() const {
        return _v.coord;
    }

    column_coordinate_type j() const {
        //LOG() << "building diag " << _v.coord << std::endl;
        return _v.coord;
    }

    nonzero_value_type v() const {
        return static_cast< int >( _v.coord );
    }

    static diag_iterator make_begin( std::size_t ) {
        return diag_iterator( 0 );
    }

    static diag_iterator make_end( std::size_t size ) {
        return diag_iterator( size );
    }

    static diag_iterator make_parallel_begin( std::size_t size ) {
        std::size_t num_procs { spmd<>::nprocs() };
        std::size_t per_process { ( size + num_procs - 1 ) / num_procs }; // round up
        return diag_iterator( per_process * spmd<>::pid() );
    }

    static diag_iterator make_parallel_end( std::size_t size ) {
        std::size_t num_procs { spmd<>::nprocs() };
        std::size_t per_process { ( size + num_procs - 1 ) / num_procs }; // round up
        std::size_t last{ std::min( size, per_process * ( spmd<>::pid() + 1 ) ) };
        return diag_iterator( last );
    }
};


template< typename T > void test_matrix_sizes_match( const Matrix< T >& mat1, const Matrix< T >& mat2) {
    ASSERT_EQ( grb::nrows( mat1), grb::nrows( mat2) );
    ASSERT_EQ( grb::ncols( mat1), grb::ncols( mat2) );
}


template< typename T > bool matrices_values_are_equal( const Matrix< T >& mat1, const Matrix< T >& mat2,
    bool log_all_differences = false ) {

    auto beg1( mat1.cbegin() );
    auto end1( mat1.cend() );

    auto beg2( mat2.cbegin() );
    auto end2( mat2.cend() );

    std::size_t mat_size{ grb::nrows( mat1) };

    if( ( beg1 != end1 ) != ( beg2 != end2 ) ) {
        LOG() << "matrix initial iterators do not match" << std::endl;
        return false;
    }

    bool match{ true };
    std::size_t i{0};
    while( beg1 != end1 && beg2 != end2 ) {

        std::size_t row1{ beg1->first.first }, row2{ beg2->first.first };
        std::size_t col1{ beg1->first.second }, col2{ beg2->first.second };
        T val1{ beg1->second }, val2{ beg2->second };

        bool row_mismatch{ row1 != row2  }, col_mismatch{ col1 != col2 }, v_mismatch{ val1 != val2 };
        bool mismatch{ row_mismatch || col_mismatch || v_mismatch };
        match &= ! mismatch;

        if( mismatch ) {
            LOG() << "MISMATCH: row " << ( row_mismatch? MISMATCH_HIGHLIGHT_BEGIN : NO_MISMATCH ) << row1
                    << ", " << row2 << ( row_mismatch? MISMATCH_HIGHLIGHT_END : NO_MISMATCH )
                << ", col " << ( col_mismatch? MISMATCH_HIGHLIGHT_BEGIN : NO_MISMATCH ) << col1
                    << ", " << col2 << ( col_mismatch? MISMATCH_HIGHLIGHT_END : NO_MISMATCH )
                << "row " << ( v_mismatch? MISMATCH_HIGHLIGHT_BEGIN : NO_MISMATCH ) << val1
                    << ", " << val2 << ( v_mismatch? MISMATCH_HIGHLIGHT_END : NO_MISMATCH );
            if( ! log_all_differences ){
                return false;
            }
        } else {
            //LOG() << "row " << row1 << " col " << col1 << " val " << val1 << std::endl;
        }

        ++beg1;
        ++beg2;
        if ( i >= mat_size ) {
            return false;
        }
        i++;
    }

    if( ( beg1 != end1 ) != ( beg2 != end2 ) ) {
        LOG() << "matrix final iterators do not match" << std::endl;
        return false;
    }

    return match;
}

template< typename T > void test_diagonal_matrix( std::size_t mat_size_1 ) {

    std::cout << "testing diagonal matrix of size " << mat_size_1 << std::endl;

    Matrix< int > sequential_matrix( mat_size_1, mat_size_1 );
    diag_iterator begin( diag_iterator::make_begin( mat_size_1) );
    diag_iterator end( diag_iterator::make_end( mat_size_1) );
    RC ret { buildMatrixUnique( sequential_matrix, begin, end, IOMode::SEQUENTIAL ) };
    ASSERT_RC_SUCCESS( ret );

    Matrix< int > parallel_matrix( mat_size_1, mat_size_1 );
    diag_iterator pbegin( diag_iterator::make_parallel_begin( mat_size_1) );
    diag_iterator pend( diag_iterator::make_parallel_end( mat_size_1) );
    ret = buildMatrixUnique( parallel_matrix, pbegin, pend, IOMode::PARALLEL );
    ASSERT_RC_SUCCESS( ret );

    test_matrix_sizes_match( sequential_matrix, parallel_matrix );
    ASSERT_TRUE( matrices_values_are_equal( sequential_matrix, parallel_matrix ) );
}

void grbProgram( const void *, const size_t, int & ) {

	std::initializer_list< std::size_t > sizes{ spmd<>::nprocs(), spmd<>::nprocs() + 9,
		spmd<>::nprocs() + 16, 100003 };

    for( std::size_t mat_size : sizes ) {
        test_diagonal_matrix< int >( mat_size );
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

	// done
	return error;
}
