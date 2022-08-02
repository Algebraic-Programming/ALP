
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
 * Various utility classes to generate matrices of different shapes.
 *
 * Matrix generators conform to the STL random access iterator specification,
 * but the tag can be set to forward iterator via a boolean template parameter
 * for testing purposes.
 *
 * @author Alberto Scolari
 * @date 20/06/2022
 */

#ifndef _GRB_UTILS_MATRIX_GENERATORS
#define _GRB_UTILS_MATRIX_GENERATORS

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <iterator>
#include <algorithm>


namespace grb {

	namespace utils {

		/**
		 * Computes the first nonzero ID as well as the number of nonzeroes per
		 * process.
		 *
		 * From the number of total nonzeroes \a num_nonzeroes, compute the maximum
		 * number of per-process nonzeroes \a num_nonzeroes_per_process as well as
		 * the first nonzero on the current process into \a first_local_nonzero; if
		 * there are more processes than nonzeroes, i.e.,
		 *   - \a num_nonzeroes_per_process * \a processes > \a num_nonzeroes,
		 * then sets \a first_local_nonzero to \a num_nonzeroes.
		 *
		 * \warning The returned nonzeroes per process is an upper bound.
		 *
		 * @param[in]  num_nonzeroes The number of nonzeroes to store.
		 * @param[out] num_nonzeroes_per_process Upper bound on the number of
		 *                                       nonzeroes per process.
		 * @param[out] first_local_nonzero The first ID of the block of nonzeroes
		 *                                 local to this process.
		 */
		template< typename T >
		void compute_parallel_first_nonzero(
			const T num_nonzeroes,
			T &num_nonzeroes_per_process,
			T &first_local_nonzero
		) {
			const T num_procs = spmd<>::nprocs();
			num_nonzeroes_per_process = (num_nonzeroes + num_procs - 1) / num_procs;
			first_local_nonzero =
				std::min(
					num_nonzeroes_per_process * spmd<>::pid(),
					num_nonzeroes
				);
		}

		/**
		 * Computes the index of the first nonzero for the current process.
		 *
		 * Relies on #compute_parallel_first_nonzero and ignores the returned upper
		 * bound.
		 */
		template< typename T >
		T compute_parallel_first_nonzero( const T num_nonzeroes ) {
			T nnz_per_proc, first;
			compute_parallel_first_nonzero( num_nonzeroes, nnz_per_proc, first );
			return first;
		}

		/**
		 * Computes the index of the last parallel nonzero + 1 (i.e., exclusive).
		 *
		 * Local nonzeroes are thus in the range
		 * 	[ compute_parallel_first_nonzero( num_nonzeroes ) ,
		 * 		compute_parallel_last_nonzero( num_nonzeroes ) )
		 */
		template< typename T >
		T compute_parallel_last_nonzero( const T num_nonzeroes ) {
			T num_non_zeroes_per_process, first_local_nonzero;
			compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process,
				first_local_nonzero );
			return std::min( num_nonzeroes, first_local_nonzero +
				num_non_zeroes_per_process );
		}

		/**
		 * Returns the number of nonzeroes stored locally
		 */
		template< typename T >
		T compute_parallel_num_nonzeroes( const T num_nonzereos ) {
			return compute_parallel_last_nonzero( num_nonzereos ) -
				compute_parallel_first_nonzero( num_nonzereos );
		}

		namespace internal {

			/**
			 * Computes the difference between \a a and \a b and returns it as the given
			 * type \a DiffT.
			 *
			 * Raises an exception if \a DiffT cannot store the difference.
			 */
			template<
				typename SizeT,
				typename DiffT
			>
			DiffT compute_distance(
				const SizeT a,
				const SizeT b
			) {
				const SizeT diff = std::max( a, b ) - std::min( a, b );
				if( diff > static_cast< SizeT >( std::numeric_limits< DiffT >::max() ) ) {
					throw std::range_error( "cannot represent difference" );
				}
				DiffT result = static_cast< DiffT >( diff );
				return a >= b ? result : -result ;
			}

			/**
			 * Stores the coordinate for a generator of diagonal matrices.
			 */
			struct DiagCoordValue {
				size_t coord;
				DiagCoordValue( size_t _c ): coord( _c ) {}
			};

			/**
			 * Stores row and column values for a band matrix
			 */
			struct BandCoordValueType {
				const size_t size;
				size_t row;
				size_t col;
				BandCoordValueType() = delete;
				BandCoordValueType(
					size_t _size,
					size_t _r,
					size_t _c
				) noexcept :
					size( _size ),
					row( _r ),
					col( _c )
				{}

			};

			/**
			 * Store the number of columns and the current nonzero index of a dense
			 * matrix; the coordinates can be retrieved from these values.
			 */
			struct DenseMatCoordValueType {
				const size_t cols;
				size_t offset;
				DenseMatCoordValueType() = delete;
				DenseMatCoordValueType( const size_t _cols, const size_t _off ) noexcept :
					cols( _cols ), offset( _off )
				{}

			};

		} // end namespace ``grb::utils::internal''

		/**
		 * Random access iterator to generate a diagonal matrix.
		 *
		 * Values are set equal to the coordinate plus one.
		 *
		 * @tparam random whether the iterator is a random access one, forward
		 *                iterator otherwise
		 */
		template< bool random >
		class DiagIterator {

			public:

				using SelfType = DiagIterator< random >;
				using value_type = internal::DiagCoordValue;


			private:

				typename SelfType::value_type _v;

				DiagIterator( const size_t _c ): _v( _c ) {}

				DiagIterator(): _v( 0 ) {}


			public:

				// STL iterator type members
				using iterator_category = typename std::conditional< random,
					std::random_access_iterator_tag, std::forward_iterator_tag >::type;
				using difference_type = long;
				using pointer = internal::DiagCoordValue *;
				using reference = internal::DiagCoordValue &;

				using RowIndexType = size_t;
				using ColumnIndexType = size_t;
				using ValueType = int;

				using InputSizesType = const size_t;

				DiagIterator( const SelfType & ) = default;

				SelfType & operator++() noexcept {
					_v.coord++;
					return *this;
				}

				SelfType & operator+=( size_t offset ) noexcept {
					_v.coord += offset;
					return *this;
				}

				bool operator!=( const SelfType &other ) const {
					return other._v.coord != this->_v.coord;
				}

				bool operator==( const SelfType &other ) const {
					return !( this->operator!=( other ) );
				}

				typename SelfType::difference_type operator-(
					const SelfType &other
				) const {
					return internal::compute_distance<
						size_t, typename SelfType::difference_type
					>( this->_v.coord, other._v.coord );
				}

				typename SelfType::pointer operator->() { return &_v; }

				typename SelfType::reference operator*() { return _v; }

				RowIndexType i() const { return _v.coord; }

				ColumnIndexType j() const { return _v.coord; }

				ValueType v() const {
					return static_cast< ValueType >( _v.coord ) + 1;
				}

				static SelfType make_begin( InputSizesType &size ) {
					(void) size;
					return SelfType( 0 );
				}

				static SelfType make_end( InputSizesType &size ) {
					return SelfType( size );
				}

				static SelfType make_parallel_begin( InputSizesType &size ) {
					const size_t num_nonzeroes = size;
					size_t num_non_zeroes_per_process, first_local_nonzero;
					compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process,
						first_local_nonzero );
					return SelfType( first_local_nonzero );
				}

				static SelfType make_parallel_end( InputSizesType &size ) {
					const size_t num_nonzeroes = size;
					size_t last = compute_parallel_last_nonzero( num_nonzeroes );
					return SelfType( last );
				}

				static size_t compute_num_nonzeroes( const size_t size ) {
					return size;
				}

		};

		/**
		 * Iterator to generate a band matrix of band \a BAND.
		 *
		 * Random acces iff \a random is <tt>true</tt>.
		 */
		template<
			size_t BAND,
			bool random
		>
		class BandIterator {

			public:

				// STL iterator type members
				using iterator_category = typename std::conditional<
						random, std::random_access_iterator_tag, std::forward_iterator_tag
					>::type;
				using value_type = internal::BandCoordValueType;
				using difference_type = long;
				using pointer = internal::BandCoordValueType *;
				using reference = internal::BandCoordValueType &;


			private:

				using SelfType = BandIterator< BAND, random >;

				typename SelfType::value_type _v;

				BandIterator( const size_t size, const size_t row, const size_t col ) :
					_v( size, row, col ) {
					static_assert( BAND > 0, "BAND must be > 0");
				}

				BandIterator() : _v( 0, 0 ) {
					static_assert( BAND > 0, "BAND must be > 0");
				}

				static size_t col_to_linear( const size_t row, const size_t col ) {
					size_t min_col = row < BAND ? 0 : row - BAND;
					return col - min_col;
				}

				static size_t coords_to_linear_in_prologue(
					const size_t row, const size_t col
				) {
					return row * BAND + row * (row + 1) / 2 + col_to_linear( row, col );
				}

				static size_t coords_to_linear(
					const size_t matrix_size, const size_t row, const size_t col
				) {
					if( row < BAND ) {
						return coords_to_linear_in_prologue( row, col );
					}
					if( row < matrix_size - BAND ) {
						return PROLOGUE_ELEMENTS + ( row - BAND ) * MAX_ELEMENTS_PER_ROW +
							col_to_linear( row, col );
					}
					if( row < matrix_size ) {
						const size_t mat_size = 2 * PROLOGUE_ELEMENTS +
							(matrix_size - 2 * BAND) * MAX_ELEMENTS_PER_ROW;
						const size_t prologue_els = coords_to_linear_in_prologue(
							matrix_size - row - 1, matrix_size - col - 1
						);
						return mat_size - prologue_els - 1; // transpose coordinates
					}
					// for points outside of matrix: project to prologue
					return 2 * PROLOGUE_ELEMENTS +
						(matrix_size - 2 * BAND) * MAX_ELEMENTS_PER_ROW +
						(row - matrix_size) * BAND +
						col + BAND - row;
				}

				static void linear_to_coords_in_prologue(
					size_t position,
					size_t &row,
					size_t &col
				) {
					size_t current_row = 0;
					//linear search
					for( ;
						position >= ( current_row + 1 + BAND ) && current_row < BAND;
						(void) current_row++
					) {
						position -= ( current_row + 1 + BAND );
					}
					row = current_row;
					col = position;
				}

				static void linear_to_coords(
					const size_t matrix_size,
					size_t position,
					size_t &row,
					size_t &col
				) {
					if( position < PROLOGUE_ELEMENTS ) {
						linear_to_coords_in_prologue( position, row, col );
						return;
					}
					position -= PROLOGUE_ELEMENTS;
					const size_t max_inner_rows = matrix_size - 2 * BAND;
					if( position < max_inner_rows * MAX_ELEMENTS_PER_ROW ) {
						const size_t inner_row = position / MAX_ELEMENTS_PER_ROW;
						row = BAND + inner_row;
						position -= inner_row * MAX_ELEMENTS_PER_ROW;
						col = row - BAND + position % MAX_ELEMENTS_PER_ROW;
						return;
					}
					position -= ( matrix_size - 2 * BAND ) * MAX_ELEMENTS_PER_ROW;
					if( position < PROLOGUE_ELEMENTS ) {
						size_t end_row, end_col;

						linear_to_coords_in_prologue( PROLOGUE_ELEMENTS - 1 - position, end_row,
							end_col );
						row = matrix_size - 1 - end_row;
						col = matrix_size - 1 - end_col;
						return;
					}
					position -= PROLOGUE_ELEMENTS;
					row = matrix_size + position / ( BAND + 1 );
					col = row - BAND + position % ( BAND + 1 );
				}

				static void check_size( const size_t size ) {
					if( size < 2 * BAND + 1 ) {
						throw std::domain_error( "matrix too small for band" );
					}
				}


			public:

				static constexpr size_t MAX_ELEMENTS_PER_ROW = BAND * 2 + 1;
				static constexpr size_t PROLOGUE_ELEMENTS = (3 * BAND * BAND + BAND) / 2;

				using RowIndexType = size_t;
				using ColumnIndexType = size_t;
				using ValueType = int;
				using InputSizesType = const size_t;

				BandIterator( const SelfType & ) = default;

				SelfType & operator++() noexcept {
					const size_t max_col = std::min( _v.row + BAND, _v.size - 1 );
					if( _v.col < max_col ) {
						(void) _v.col++;
					} else {
						(void) _v.row++;
						_v.col = _v.row < BAND ? 0 : _v.row - BAND;
					}
					return *this;
				}

				SelfType & operator+=( size_t offset ) noexcept {
					const size_t position = coords_to_linear( _v.size, _v.row, _v.col );
					linear_to_coords( _v.size, position + offset, _v.row, _v.col );
					return *this;
				}

				bool operator!=( const SelfType &other ) const {
					return other._v.row != this->_v.row || other._v.col != this->_v.col;
				}

				bool operator==( const SelfType &other ) const {
					return !( this->operator!=( other ) );
				}

				typename SelfType::difference_type operator-(
					const SelfType &other
				) const {
					const size_t this_position = coords_to_linear( _v.size, _v.row, _v.col );
					const size_t other_position =
						coords_to_linear( other._v.size, other._v.row, other._v.col );
					return internal::compute_distance<
						size_t, typename SelfType::difference_type
					>( this_position, other_position );
				}

				typename SelfType::pointer operator->() { return &_v; }

				typename SelfType::reference operator*() { return _v; }

				typename SelfType::RowIndexType i() const { return _v.row; }

				typename SelfType::ColumnIndexType j() const { return _v.col; }

				ValueType v() const {
					return _v.row == _v.col ? static_cast< int >( MAX_ELEMENTS_PER_ROW ) : -1;
				}

				static SelfType make_begin( InputSizesType &size ) {
					check_size( size );
					return SelfType( size, 0, 0 );
				}

				static SelfType make_end( InputSizesType &size ) {
					check_size( size );
					size_t row, col;
					const size_t num_nonzeroes = compute_num_nonzeroes( size );
					linear_to_coords( size, num_nonzeroes, row, col );
					return SelfType( size, row, col );
				}

				static SelfType make_parallel_begin( InputSizesType &size ) {
					check_size( size );
					const size_t num_nonzeroes = compute_num_nonzeroes( size );
					size_t num_non_zeroes_per_process, first_local_nonzero;
					compute_parallel_first_nonzero( num_nonzeroes, num_non_zeroes_per_process,
						first_local_nonzero );
					size_t row, col;
					linear_to_coords( size, first_local_nonzero, row, col );
					return SelfType( size, row, col );
				}

				static SelfType make_parallel_end( InputSizesType &size ) {
					check_size( size );
					const size_t num_nonzeroes = compute_num_nonzeroes( size );
					size_t last = compute_parallel_last_nonzero( num_nonzeroes );
					size_t row, col;
					linear_to_coords( size, last, row, col );
					return SelfType( size, row, col );
				}

				static size_t compute_num_nonzeroes( const size_t size ) {
					return 2 * PROLOGUE_ELEMENTS + (size - 2 * BAND) * MAX_ELEMENTS_PER_ROW;
				}

		};

		/**
		 * Iterator generating a dense matrix of value type \a ValT.
		 *
		 * Random access iff \a random is true.
		 */
		template<
			typename ValT,
			bool random
		>
		class DenseMatIterator {

			public:

				// STL iterator type members
				using iterator_category = typename std::conditional<
						random, std::random_access_iterator_tag, std::forward_iterator_tag
					>::type;
				using value_type = internal::DenseMatCoordValueType;
				using difference_type = long;
				using pointer = internal::DenseMatCoordValueType *;
				using reference = internal::DenseMatCoordValueType &;


			private:

				using SelfType = DenseMatIterator< ValT, random >;

				typename SelfType::value_type _v;


			public:

				using RowIndexType = size_t;
				using ColumnIndexType = size_t;
				using ValueType = ValT;
				using InputSizesType = const std::array< size_t, 2 >;

				DenseMatIterator(
					size_t _cols,
					size_t _off
				) noexcept :
					_v( _cols, _off )
				{}

				DenseMatIterator( const SelfType& ) = default;

				SelfType& operator++() noexcept {
					_v.offset++;
					return *this;
				}

				SelfType& operator+=( size_t offset ) noexcept {
					_v.offset += offset;
					return *this;
				}

				bool operator!=( const SelfType& other ) const {
					return other._v.offset != this->_v.offset;
				}

				bool operator==( const DenseMatIterator& other ) const {
					return !( this->operator!=( other ) );
				}

				typename SelfType::difference_type operator-( const SelfType& other ) const {
					return internal::compute_distance<
						size_t, typename SelfType::difference_type
					>( this->_v.offset, other._v.offset );
				}

				typename SelfType::pointer operator->() { return &_v; }

				typename SelfType::reference operator*() { return _v; }

				RowIndexType i() const { return _v.offset / _v.cols; }

				ColumnIndexType j() const { return _v.offset % _v.cols; }

				ValueType v() const {
					return static_cast< ValueType >( _v.offset ) + 1;
				}

				static SelfType make_begin( InputSizesType &sizes ) {
					return SelfType( sizes[1], 0 );
				}

				static SelfType make_end( InputSizesType &sizes ) {
					const size_t num_nonzeroes = compute_num_nonzeroes( sizes );
					return SelfType( sizes[1], num_nonzeroes );
				}

				static SelfType make_parallel_begin( InputSizesType &sizes ) {
					size_t num_non_zeroes_per_process, first_local_nonzero;
					compute_parallel_first_nonzero( compute_num_nonzeroes( sizes ),
						num_non_zeroes_per_process, first_local_nonzero );
					return SelfType( sizes[1], first_local_nonzero );
				}

				static SelfType make_parallel_end( InputSizesType &sizes ) {
					size_t last = compute_parallel_last_nonzero(
						compute_num_nonzeroes( sizes ) );
					return SelfType( sizes[1], last );
				}

				static size_t compute_num_nonzeroes( InputSizesType &sizes ) {
					return sizes[0] * sizes[1];
				}

		};

	} // end namespace grb::utils

} // end namespace grb

#endif // _GRB_UTILS_MATRIX_GENERATORS

