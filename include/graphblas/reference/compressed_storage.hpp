
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

/*
 * @author A. N. Yzelman
 * @date 10th of August, 2016
 */

#ifndef _H_GRB_REFERENCE_COMPRESSED_STORAGE
#define _H_GRB_REFERENCE_COMPRESSED_STORAGE

#include <cstring> //std::memcpy

#if reference == reference_omp
 #define _H_GRB_REFERENCE_OMP_COMPRESSED_STORAGE
 #include <omp.h>
#endif


namespace grb {

	namespace internal {

		template< typename D, typename IND, typename SIZE >
		class Compressed_Storage;

		namespace {

			/**
			 * Copies <tt>row_index<tt> and <tt>col_index<tt> from a
			 * given Compressed_Storage to another.
			 *
			 * Performs no safety checking. Performs no (re-)allocations.
			 *
			 * @param[out] output The container to copy the coordinates to.
			 * @param[in] input   The container to copy the coordinates from.
			 * @param[in] nz      The number of nonzeroes in the \a other container.
			 * @param[in] m       The index dimension of the \a other container.
			 * @param[in] k       The start position to copy from (inclusive).
			 * @param[in] end     The end position to copy to (exclusive).
			 *
			 * The copy range is 2nz + m + 1, i.e.,
			 *   -# 0 <= start <  2nz + m + 1
			 *   -# 0 <  end   <= 2nz + m + 1
			 *
			 * Concurrent calls to this function are allowed iff they consist of
			 * disjoint ranges \a start and \a end. The copy is guaranteed to be
			 * complete if the union of ranges spans 0 to 2nz + m + 1.
			 */
			template<
				typename OutputType, typename OutputIND, typename OutputSIZE,
				typename InputType, typename InputIND, typename InputSIZE
			>
			static inline void copyCoordinatesFrom(
				Compressed_Storage< OutputType, OutputIND, OutputSIZE > &output,
				const Compressed_Storage< InputType, InputIND, InputSIZE > &input,
				const size_t nz, const size_t m,
				size_t k, size_t end
			) {
				static_assert( std::is_convertible< InputIND, OutputIND >::value,
					"InputIND must be convertible to OutputIND"
				);
				static_assert( std::is_convertible< InputSIZE, OutputSIZE >::value,
					"InputSIZE must be convertible to OutputSIZE"
				);

				if( k < nz ) {
					const size_t loop_end = std::min( nz, end );
					assert( k <= loop_end );
#ifdef _H_GRB_REFERENCE_OMP_COMPRESSED_STORAGE
					#pragma omp for simd
#endif
					for( size_t i = k; i < loop_end; ++i ) {
						output.row_index[ i ] = static_cast< OutputIND >( input.row_index[ i ] );
					}
					k = 0;
				} else {
					assert( k >= nz );
					k -= nz;
				}
				if( end <= nz ) {
					return;
				}
				end -= nz;
				if( k < m + 1 ) {
					const size_t loop_end = std::min( m + 1, end );
					assert( k <= loop_end );
#ifdef _H_GRB_REFERENCE_OMP_COMPRESSED_STORAGE
					#pragma omp for simd
#endif
					for( size_t i = k; i < loop_end; ++i ) {
						output.col_start[ i ] = static_cast< OutputSIZE >( input.col_start[ i ] );
					}
#ifndef NDEBUG
					for( size_t chk = k; chk < loop_end - 1; ++chk ) {
						assert( input.col_start[ chk ] <= input.col_start[ chk + 1 ] );
						assert( output.col_start[ chk ] <= output.col_start[ chk + 1 ] );
					}
#endif
				}
			}
		} // namespace

		/**
		 * Basic functionality for a compressed storage format (CRS/CSR or CCS/CSC).
		 *
		 * FOR INTERNAL USE ONLY.
		 *
		 * This is a very unsafe wrapper class around three arrays. Use with care.
		 *
		 * @tparam D    The nonzero value types.
		 * @tparam IND  The matrix coordinate type.
		 * @tparam SIZE The start offset index type.
		 *
		 * The matrix dimension must be encodeable in \a IND. The number of nonzeroes
		 * must be encodeable in \a SIZE.
		 */
		template< typename D, typename IND, typename SIZE >
		class Compressed_Storage {

			private:

				/** Point to our own type. */
				typedef Compressed_Storage< D, IND, SIZE > SelfType;

				/**
				 * Resets all arrays to a null pointer.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur.
				 */
				void clear() {
					values = nullptr;
					row_index = nullptr;
					col_start = nullptr;
				}


			public:

				/** The value array. */
				D * __restrict__ values;

				/** The row index values. */
				IND * __restrict__ row_index;

				/** The column start indices. */
				SIZE * __restrict__ col_start;


			private:

				/** Implements a move from another instance */
				void moveFromOther( SelfType &other ) {
					row_index = other.row_index;
					col_start = other.col_start;
					values = other.values;
					other.clear();
				}


			public:

				/** The matrix nonzero iterator class. */
				template< class ActiveDistribution >
				class ConstIterator : public std::iterator<
					std::forward_iterator_tag,
					std::pair< std::pair< const size_t, const size_t >, const D >
				> {

					private:

						/** The value array. */
						const D * __restrict__ values;

						/** The row index values. */
						const IND * __restrict__ row_index;

						/** The column start indices. */
						const SIZE * __restrict__ col_start;

						/** Current index. */
						size_t k;

						/** Max row. */
						size_t m;

						/** Max column. */
						size_t n;

						/** Current row. */
						size_t row;

						/** My user process ID. */
						size_t s;

						/** Total number of user processes. */
						size_t P;

						/** Current nonzero. */
						std::pair< std::pair< size_t, size_t >, D > nonzero;


					public:

						// ALP typedefs
						typedef size_t RowIndexType;
						typedef size_t ColumnIndexType;
						typedef D ValueType;

						/** Base constructor. */
						ConstIterator() noexcept : values( nullptr ),
							row_index( nullptr ), col_start( nullptr ),
							k( 0 ), m( 0 ), n( 0 ), row( 1 ), s( 0 ), P( 1 )
						{
							nonzero.first.first = 1;
							nonzero.first.second = 1;
						}

						/** Copy constructor. */
						ConstIterator( const ConstIterator &other ) noexcept :
							values( other.values ),
							row_index( other.row_index ), col_start( other.col_start ),
							k( other.k ), m( other.m ), n( other.n ),
							row( other.row ), s( other.s ), P( other.P ),
							nonzero( other.nonzero )
						{
#ifdef _DEBUG
							std::cout << "Matrix< reference >::const_iterator copy-constructor "
								<< "called\n";
#endif
						}

						/** Move constructor. */
						ConstIterator( ConstIterator &&other ) {
#ifdef _DEBUG
							std::cout << "Matrix< reference >::const_iterator move-constructor "
								<< "called\n";
#endif
							values = std::move( other.values );
							row_index = std::move( other.row_index );
							col_start = std::move( other.col_start );
							k = std::move( other.k );
							m = std::move( other.m );
							n = std::move( other.n );
							row = std::move( other.row );
							s = std::move( other.s );
							P = std::move( other.P );
							nonzero = std::move( other.nonzero );
						}

						/** Non-trivial constructor. */
						ConstIterator(
							const Compressed_Storage &_storage,
							const size_t _m, const size_t _n, const size_t _nz,
							const bool end, const size_t _s = 0, const size_t _P = 1
						) noexcept :
							values( _storage.values ),
							row_index( _storage.row_index ), col_start( _storage.col_start ), k( 0 ),
							m( _m ), n( _n ),
							s( _s ), P( _P )
						{
#ifdef _DEBUG
							std::cout << "Compressed_Storage::Const_Iterator constructor called, "
								<< "with storage " << ( &_storage ) << ", "
								<< "m " << _m << ", n " << _n << ", and end " << end << ".\n";
#endif
							if( _nz == 0 || _m == 0 || _n == 0 || end ) {
								row = m;
								return;
							}

							// skip to first non-empty row
							for( row = 0; row < m; ++row ) {
								if( col_start[ row ] != col_start[ row + 1 ] ) {
									break;
								}
							}

							if( row < m ) {
#ifdef _DEBUG
								std::cout << "\tInitial pair, pre-translated at " << row << ", "
									<< row_index[ k ] << " with value " << values[ k ] << ". "
									<< "P = " << P << ", row = " << row << ".\n";
#endif
								const size_t col_pid = ActiveDistribution::offset_to_pid(
									row_index[ k ], n, P );
								const size_t col_off = ActiveDistribution::local_offset(
									n, col_pid, P );
								nonzero.first.first = ActiveDistribution::local_index_to_global(
									row, m, s, P );
								nonzero.first.second = ActiveDistribution::local_index_to_global(
									row_index[ k ] - col_off, n, col_pid, P );
								nonzero.second = values[ k ];
#ifdef _DEBUG
								std::cout << "\tInitial pair at " << nonzero.first.first << ", "
									<< nonzero.first.second << " with value " << nonzero.second << ". "
									<< "P = " << P << ", row = " << row << ".\n";
#endif
							}
						}

						/** Copy assignment. */
						ConstIterator & operator=( const ConstIterator &other ) noexcept {
#ifdef _DEBUG
							std::cout << "Matrix (reference) const-iterator copy-assign operator "
								<< "called\n";
#endif
							values = other.values;
							row_index = other.row_index;
							col_start = other.col_start;
							k = other.k;
							m = other.m;
							n = other.n;
							row = other.row;
							s = other.s;
							P = other.P;
							nonzero = other.nonzero;
							return *this;
						}

						/** Move assignment. */
						ConstIterator & operator=( ConstIterator &&other ) {
#ifdef _DEBUG
							std::cout << "Matrix (reference) const-iterator move-assign operator "
								<< "called\n";
#endif
							values = std::move( other.values );
							row_index = std::move( other.row_index );
							col_start = std::move( other.col_start );
							k = std::move( other.k );
							m = std::move( other.m );
							n = std::move( other.n );
							row = std::move( other.row );
							s = std::move( other.s );
							P = std::move( other.P );
							nonzero = other.nonzero;
							return *this;
						}

						/** Whether two iterators compare equal. */
						bool operator==( const ConstIterator &other ) const noexcept {
#ifdef _DEBUG
							std::cout << "Compressed_Storage::Const_Iterator operator== called "
								<< "with k ( " << k << ", " << other.k << " ), "
								<< " m ( " << m << ", " << other.m << " )\n";
#endif
							assert( values == other.values );
							assert( row_index == other.row_index );
							assert( col_start == other.col_start );
							assert( m == other.m );
							assert( n == other.n );
							assert( s == other.s );
							assert( P == other.P );
#ifndef NDEBUG
							if( k == other.k ) {
								assert( row == other.row );
							}
#endif
							if( row == other.row ) {
								return k == other.k;
							} else {
								return false;
							}
						}

						/** Whether two iterators do not compare equal. */
						bool operator!=( const ConstIterator &other ) const noexcept {
#ifdef _DEBUG
							std::cout << "Compressed_Storage::Const_Iterator operator!= called "
								<< "with k ( " << k << ", " << other.k << " ), "
								<< "row ( " << row << ", " << other.row << " ), "
								<< "m ( " << m << ", " << other.m << " )\n";
#endif
							assert( values == other.values );
							assert( row_index == other.row_index );
							assert( m == other.m );
							assert( n == other.n );
							assert( col_start == other.col_start );
							assert( s == other.s );
							assert( P == other.P );
							if( row == other.row ) {
								if( row == m ) {
									return false;
								}
								return k != other.k;
							} else {
								return true;
							}
						}

						/** Move to the next iterator. */
						ConstIterator & operator++() noexcept {
#ifdef _DEBUG
							std::cout << "Compressed_Storage::operator++ called\n";
#endif
							if( row == m ) {
								return *this;
							}
							assert( row < m );
							assert( k < col_start[ row + 1 ] );
							(void) ++k;
							while( row < m && k == col_start[ row + 1 ] ) {
								(void) ++row;
							}
							if( row < m ) {
#ifdef _DEBUG
								std::cout << "\tupdated triple, pre-translated at ( " << row << ", "
									<< row_index[ k ] << " ): " << values[ k ] << "\n";
#endif
								const size_t col_pid = ActiveDistribution::offset_to_pid(
									row_index[ k ], n, P
								);
								const size_t col_off = ActiveDistribution::local_offset(
									n, col_pid, P );
								assert( col_off <= row_index[ k ] );
								nonzero.first.first = ActiveDistribution::local_index_to_global(
									row, m, s, P );
								nonzero.first.second = ActiveDistribution::local_index_to_global(
									row_index[ k ] - col_off, n, col_pid, P );
								nonzero.second = values[ k ];
#ifdef _DEBUG
								std::cout << "\tupdated triple at ( " << nonzero.first.first << ", "
									<< nonzero.first.second << " ): " << nonzero.second << "\n";
#endif
							} else {
								assert( row == m );
								k = 0;
							}
							return *this;
						}

						/** Return a const-reference to the current nonzero. */
						const std::pair< std::pair< size_t, size_t >, D > &
						operator*() const noexcept {
							assert( row < m );
							return nonzero;
						}

						/** Return a pointer to the current nonzero. */
						const std::pair< std::pair< size_t, size_t >, D > *
						operator->() const noexcept {
							assert( row < m );
							return &nonzero;
						}

						/** ALP-specific extension that returns the row coordinate. */
						const size_t & i() const noexcept {
							return nonzero.first.first;
						}

						/** ALP-specific extension that returns the column coordinate. */
						const size_t & j() const noexcept {
							return nonzero.first.second;
						}

						/** ALP-specific extension that returns the nonzero value. */
						const D & v() const noexcept {
							return nonzero.second;
						}

				};

				/** Base constructor (NULL-initialiser). */
				Compressed_Storage() :
					values( nullptr ), row_index( nullptr ), col_start( nullptr )
				{}

				/** Non-shallow copy constructor. */
				explicit Compressed_Storage(
					const Compressed_Storage< D, IND, SIZE > &other
				) : values( other.values ),
					row_index( other.row_index ), col_start( other.col_start )
				{}

				/** Move constructor. */
				Compressed_Storage( Compressed_Storage< D, IND, SIZE > &&other ) :
					values( other.values ),
					row_index( other.row_index ), col_start( other.col_start )
				{
					moveFromOther( other );
				}

				/** Assign from temporary. */
				SelfType& operator=( SelfType &&other ) {
					moveFromOther( other );
					return *this;
				}

				/**
				 * @returns The value array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				D * getValues() noexcept {
					return values;
				}

				/**
				 * @returns The index array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				IND * getIndices() noexcept {
					return row_index;
				}

				/**
				 * @returns The offset array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				SIZE * getOffsets() noexcept {
					return col_start;
				}

				/**
				 * @returns Const-version of the offset array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				const SIZE * getOffsets() const noexcept {
					return col_start;
				}

				/**
				 * Gets the current raw pointers of the resizable arrays that are being used
				 * by this instance.
				 */
				void getPointers( void ** pointers ) {
					*pointers++ = values;
					*pointers++ = row_index;
				}

				/**
				 * Replaces the existing arrays with the given ones.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur. The new memory arrays given to this
				 * function are left untouched (until they are used by other calls to this
				 * class).
				 */
				void replace(
					const void * __restrict__ const new_vals,
					const void * __restrict__ const new_ind
				) {
					values = const_cast< D * __restrict__ >(
						static_cast< const D *__restrict__ >(new_vals)
					);
					row_index = const_cast< IND * __restrict__ >(
						static_cast< const IND *__restrict__ >(new_ind)
					);
				}

				/**
				 * Replaces an existing start array with a given one.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur. The new memory area given to this function
				 * are left untouched (until they are used by other calls to this class).
				 */
				void replaceStart( const void * const new_start ) {
					col_start = const_cast< SIZE * __restrict__ >(
						static_cast< const SIZE *__restrict__>(new_start)
					);
				}

				/**
				 * @returns The range for use with #copyFrom.
				 */
				static inline size_t copyFromRange( const size_t nz, const size_t m ) {
					return 2 * nz + m + 1;
				}

				/**
				 * Copies coordinates from a given Compressed_Storage, then fills the
				 * values with the given identity.
				 *
				 * Performs no safety checking. Performs no (re-)allocations.
				 *
				 * @tparam use_id   If set to <tt>true</tt>, use \a id instead of values in
				 *                  \a other.
				 * @param[in] other The container to copy from.
				 * @param[in] nz    The number of nonzeroes in the \a other container.
				 * @param[in] m     The index dimension of the \a other container.
				 * @param[in] start The start position to copy from (inclusive).
				 * @param[in] end   The end position to copy to (exclusive).
				 * @param[in] id    A pointer to a value overriding those in \a other.
				 *                  Will only be used if and only if \a use_id is set
				 *                  to <tt>true</tt>.
				 * The copy range is 2nz + m + 1, i.e.,
				 *   -# 0 <= start <  2nz + m + 1
				 *   -# 0 <  end   <= 2nz + m + 1
				 *
				 * Concurrent calls to this function are allowed iff they consist of
				 * disjoint ranges \a start and \a end. The copy is guaranteed to be
				 * complete if the union of ranges spans 0 to 2nz + m + 1.
				 */
				template<
					bool useId = true,
					typename InputType, typename InputIND, typename InputSIZE,
					typename ValueType
				>
				void copyFrom(
					const Compressed_Storage< InputType, InputIND, InputSIZE > &other,
					const size_t nz, const size_t m,
					const size_t start, size_t end,
					const ValueType * __restrict__ id,
					const typename std::enable_if< useId, void >::type * = nullptr
				) {
#ifdef _DEBUG
					std::cout << "CompressedStorage::copyFrom (cast) called with range "
						<< start << "--" << end << ". The identity " << (*id) << " will be used.\n";
#endif
					assert( start <= end );
					size_t k = start;
					if( k < nz ) {
						const size_t loop_end = std::min( nz, end );
						assert( k <= loop_end );
						std::fill_n( values + k, loop_end - k, *id );
						k = 0;
					} else {
						assert( k >= nz );
						k -= nz;
					}
					if( end <= nz ) {
						return;
					}
					end -= nz;

					copyCoordinatesFrom( *this, other, nz, m, k, end );
				}

				/**
				 * Copies contents from a given Compressed_Storage.
				 *
				 * Performs no safety checking. Performs no (re-)allocations.
				 *
				 * @param[in] other The container to copy from.
				 * @param[in] nz    The number of nonzeroes in the \a other container.
				 * @param[in] m     The index dimension of the \a other container.
				 * @param[in] start The start position to copy from (inclusive).
				 * @param[in] end   The end position to copy to (exclusive).
				 *
				 * The copy range is 2nz + m + 1, i.e.,
				 *   -# 0 <= start <  2nz + m + 1
				 *   -# 0 <  end   <= 2nz + m + 1
				 *
				 * Concurrent calls to this function are allowed iff they consist of
				 * disjoint ranges \a start and \a end. The copy is guaranteed to be
				 * complete if the union of ranges spans 0 to 2nz + m + 1.
				 */
				template<
					bool useId = false,
					typename InputType, typename InputIND, typename InputSIZE
				>
				void copyFrom(
					const Compressed_Storage< InputType, InputIND, InputSIZE > &other,
					const size_t nz, const size_t m,
					const size_t start, size_t end,
					const typename std::enable_if< !useId, void >::type * = nullptr
				) {
					static_assert( !std::is_void< InputType >::value,
						"InputType must not be void"
					);
#ifdef _DEBUG
					std::cout << "CompressedStorage::copyFrom called with range "
						<< start << "--" << end << ". No identity will be used.\n";
#endif
					size_t k = start;
					if( k < nz ) {
						const size_t loop_end = std::min( nz, end );
#ifdef _DEBUG
						std::cout << "\t value range " << k << " -- " << loop_end << "\n";
#endif
						assert( k <= loop_end );

						GRB_UTIL_IGNORE_CLASS_MEMACCESS;
#ifdef _H_GRB_REFERENCE_OMP_COMPRESSED_STORAGE
						#pragma omp for simd
#endif
						for( size_t i = k; i < loop_end; ++i ) {
							values[ i ] = static_cast< D >( other.values[ i ] );
						}
						GRB_UTIL_RESTORE_WARNINGS;

						k = 0;
					} else {
						assert( k >= nz );
						k -= nz;
					}
					if( end <= nz ) {
						return;
					}
					end -= nz;

					copyCoordinatesFrom( *this, other, nz, m, k, end );
				}

				/**
				 * Writes a nonzero to the given position. Does \em not update the
				 * \a col_start array. Does not perform any type checking.
				 *
				 * @tparam fwd_it Forward iterator type for the nonzero.
				 *
				 * @param[in] pos Where to store this nonzero.
				 * @param[in] row Whether to store the row or column index.
				 * @param[in] it  The nonzero iterator.
				 *
				 * This function shall not change the position of the iterator passed.
				 */
				template< typename fwd_it >
				void recordValue( const size_t &pos, const bool row, const fwd_it &it ) {
					row_index[ pos ] = row ? it.i() : it.j();
					values[ pos ] = it.v();
#ifdef _DEBUG
					std::cout << "\t nonzero at position " << it.i() << " by " << it.j()
						<< " is stored at position " << pos << " has value " << it.v() << ".\n";
#endif
				}

				/**
				 * Returns the size of the raw arrays, in bytes. This function is used to
				 * facilitate memory allocation. No bounds checking of any kind will be
				 * performed.
				 *
				 * @param[out] sizes Where the array byte sizes have to be written to. This
				 *                   must point to an array of at least three values of type
				 *                   \a size_t. The first value refers to the number of
				 *                   bytes required to store the nonzero value array. The
				 *                   second refers to the bytes required to store the
				 *                   minor indices. The third refers to the bytes required
				 *                   to store the major indices.
				 * @param[in] nonzeroes The number of nonzeroes to be stored.
				 */
				void getAllocSize( size_t * sizes, const size_t nonzeroes ) {
					*sizes++ = nonzeroes * sizeof( D );   // values array
					*sizes++ = nonzeroes * sizeof( IND ); // index array
				}

				/**
				 * Returns the size of the start array, in bytes. This function is used to
				 * facilitate memory allocation. No bounds checking of any kind will be
				 * performed.
				 *
				 * @param[out] size     Where the size (in bytes) will be stored.
				 * @param[in]  dim_size The size of the major dimension.
				 */
				void getStartAllocSize( size_t * size, const size_t dim_size ) {
					*size = ( dim_size + 1 ) * sizeof( SIZE );
				}

				/**
				 * Retrieves a reference to the k-th stored nonzero.
				 *
				 * @tparam ReturnType The requested nonzero type.
				 *
				 * @param[in] k    The index of the nonzero to return.
				 * @param[in] ring The semiring under which to interpret the requested
				 *                 nonzero.
				 *
				 * @return The requested nonzero, cast to the appropriate domain
				 *         depending on \a ring.
				 */
				template< typename ReturnType >
				inline const ReturnType getValue(
					const size_t k, const ReturnType &
				) const noexcept {
					return static_cast< ReturnType >( values[ k ] );
				}

#ifdef _DEBUG
				/**
				 * For _DEBUG tracing, define a function that prints the value to a string.
				 *
				 * @param[in] k The index of the value to print.
				 *
				 * @returns A pretty-printed version of the requested value.
				 */
				inline std::string getPrintValue( const size_t k ) const noexcept {
					std::ostringstream oss;
					oss << "values[ " << k << " ] = " << values[ k ];
					return oss.str();
				}
#endif

				/**
				 * Helper function to set a nonzero value. Only records the value itself,
				 * does nothing to update the matrix' nonzero structure.
				 *
				 * @param[in]  k  Where to store the given nonzero value.
				 * @param[in] val Which value to store.
				 */
				inline void setValue( const size_t k, const D & val ) noexcept {
					values[ k ] = val;
				}

		};

		/**
		 * Template specialisation for pattern matrices.
		 *
		 * FOR INTERNAL USE ONLY. Use allowed only via semirings.
		 *
		 * This is a very unsafe wrapper class around two arrays. Use with care.
		 *
		 * @tparam IND  The matrix coordinate type.
		 * @tparam SIZE The nonzero type.
		 *
		 * The matrix dimension must be encodeable in \a IND. The number of nonzeroes
		 * must be encodeable in \a SIZE.
		 */
		template< typename IND, typename SIZE >
		class Compressed_Storage< void, IND, SIZE > {

			public:

				/** The row index values. */
				IND * __restrict__ row_index;

				/** The column start indices. */
				SIZE * __restrict__ col_start;


			private:

				/** Point to our own type */
				typedef Compressed_Storage< void, IND, SIZE > SelfType;

				/** Implements a move from another instance */
				void moveFromOther( SelfType &other ) {
					row_index = other.row_index;
					col_start = other.col_start;
					other.clear();
				}


			public:

				/** The matrix nonzero iterator class. */
				template< class ActiveDistribution >
				class ConstIterator : public std::iterator<
					std::forward_iterator_tag, std::pair< const size_t, const size_t >
				> {

					private:

						/** The row index values. */
						const IND * __restrict__ row_index;

						/** The column start indices. */
						const SIZE * __restrict__ col_start;

						/** Current index. */
						size_t k;

						/** Max row. */
						size_t m;

						/** Max column. */
						size_t n;

						/** Current row. */
						size_t row;

						/** My user process ID. */
						size_t s;

						/** Total number of user processes. */
						size_t P;

						/** Current nonzero. */
						std::pair< size_t, size_t > nonzero;


					public:

						// ALP typedefs
						typedef size_t RowIndexType;
						typedef size_t ColumnIndexType;
						typedef void ValueType;

						/** Base constructor. */
						ConstIterator() noexcept :
							row_index( nullptr ), col_start( nullptr ),
							k( 0 ), m( 0 ), n( 0 ), row( 1 ),
							s( 0 ), P( 1 )
						{
#ifdef _DEBUG
							std::cout << "Iterator default constructor (pattern specialisation) "
								<< "called\n";
#endif
						}

						/** Copy constructor. */
						ConstIterator( const ConstIterator &other ) noexcept :
							row_index( other.row_index ), col_start( other.col_start ),
							k( other.k ), m( other.m ), n( other.n ), row( other.row ),
							s( 0 ), P( 1 ), nonzero( other.nonzero )
						{
#ifdef _DEBUG
							std::cout << "Iterator copy constructor (pattern specialisation) "
								<< "called\n";
#endif
						}

						/** Move constructor. */
						ConstIterator( ConstIterator &&other ) {
#ifdef _DEBUG
							std::cout << "Iterator move constructor (pattern specialisation) "
								<< "called\n";
#endif
							row_index = std::move( other.row_index );
							col_start = std::move( other.col_start );
							k = std::move( other.k );
							m = std::move( other.m );
							n = std::move( other.n );
							row = std::move( other.row );
							s = std::move( other.s );
							P = std::move( other.P );
							nonzero = std::move( other.nonzero );
						}

						/** Non-trivial constructor. */
						ConstIterator(
							const Compressed_Storage &_storage,
							const size_t _m, const size_t _n, const size_t _nz,
							const bool end, const size_t _s = 0, const size_t _P = 1
						) noexcept :
							row_index( _storage.row_index ), col_start( _storage.col_start ),
							k( 0 ), m( _m ), n( _n ),
							s( _s ), P( _P )
						{
#ifdef _DEBUG
							std::cout << "Iterator constructor (pattern specialisation) called\n";
#endif
							if( _nz == 0 || _m == 0 || _n == 0 || end ) {
								row = m;
								return;
							}

							// skip to first non-empty row
							for( row = 0; row < m; ++row ) {
								if( col_start[ row ] != col_start[ row + 1 ] ) {
									break;
								}
							}

							if( row < m ) {
								const size_t col_pid = ActiveDistribution::offset_to_pid(
									row_index[ k ], n, P );
								const size_t col_off = ActiveDistribution::local_offset(
									n, col_pid, P );
								nonzero.first = ActiveDistribution::local_index_to_global(
									row, m, s, P );
								nonzero.second = ActiveDistribution::local_index_to_global(
									row_index[ k ] - col_off, n, col_pid, P );
							}
						}

						/** Copy assignment. */
						ConstIterator & operator=( const ConstIterator &other ) noexcept {
#ifdef _DEBUG
							std::cout << "Iterator copy-assign operator (pattern specialisation) "
								<< "called\n";
#endif
							row_index = other.row_index;
							col_start = other.col_start;
							k = other.k;
							m = other.m;
							n = other.n;
							row = other.row;
							s = other.s;
							P = other.P;
							nonzero = other.nonzero;
							return *this;
						}

						/** Move assignment. */
						ConstIterator & operator=( ConstIterator &&other ) {
#ifdef _DEBUG
							std::cout << "Iterator move-assign operator (pattern specialisation) "
								<< "called\n";
#endif
							row_index = std::move( other.row_index );
							col_start = std::move( other.col_start );
							k = std::move( other.k );
							m = std::move( other.m );
							n = std::move( other.n );
							row = std::move( other.row );
							s = std::move( other.s );
							P = std::move( other.P );
							nonzero = std::move( other.nonzero );
							return *this;
						}

						/** Whether two iterators compare equal. */
						bool operator==( const ConstIterator &other ) const noexcept {
							assert( row_index == other.row_index );
							assert( col_start == other.col_start );
							assert( m == other.m );
							assert( n == other.n );
							assert( s == other.s );
							assert( P == other.P );
#ifndef NDEBUG
							if( k == other.k ) {
								assert( row == other.row );
							}
#endif
							if( row == other.row ) {
								return k == other.k;
							} else {
								return false;
							}
						}

						/** Whether two iterators do not compare equal. */
						bool operator!=( const ConstIterator &other ) const noexcept {
							assert( row_index == other.row_index );
							assert( col_start == other.col_start );
							assert( m == other.m );
							assert( n == other.n );
							assert( s == other.s );
							assert( P == other.P );
							if( row == other.row ) {
								if( row == m ) {
									return false;
								}
								return k != other.k;
							} else {
								return true;
							}
						}

						/** Move to the next iterator. */
						ConstIterator & operator++() noexcept {
							if( row == m ) {
								return *this;
							}
							assert( row < m );
							assert( k < col_start[ row + 1 ] );
							(void) ++k;
							while( row < m && k == col_start[ row + 1 ] ) {
								(void) ++row;
							}
							if( row < m ) {
								const size_t col_pid = ActiveDistribution::offset_to_pid(
									row_index[ k ], n, P );
								const size_t col_off = ActiveDistribution::local_offset(
									n, col_pid, P );
								nonzero.first = ActiveDistribution::local_index_to_global(
									row, m, s, P );
								nonzero.second = ActiveDistribution::local_index_to_global(
									row_index[ k ] - col_off, n, col_pid, P );
							} else {
								assert( row == m );
								k = 0;
							}
							return *this;
						}

						/** Return a const-reference to the current nonzero. */
						const std::pair< size_t, size_t > & operator*() const noexcept {
							assert( row < m );
							return nonzero;
						}

						/** Return a pointer to the current nonzero. */
						const std::pair< size_t, size_t > * operator->() const noexcept {
							assert( row < m );
							return &nonzero;
						}

						/** ALP-specific extension that returns the row coordinate. */
						const size_t & i() const noexcept {
							return nonzero.first;
						}

						/** ALP-specific extension that returns the column coordinate. */
						const size_t & j() const noexcept {
							return nonzero.second;
						}

				};

				/** Base constructor (NULL-initialiser). */
				Compressed_Storage() : row_index( nullptr ), col_start( nullptr ) {}

				/** Non-shallow copy constructor. */
				explicit Compressed_Storage(
					const Compressed_Storage< void, IND, SIZE > &other
				) :
					row_index( other.row_index ), col_start( other.col_start )
				{}

				/** Move constructor. */
				Compressed_Storage< void, IND, SIZE >( SelfType &&other ) {
					moveFromOther( other );
				}

				/** Assign from temporary. */
				SelfType& operator=( SelfType &&other ) {
					moveFromOther( other );
					return *this;
				}

				/**
				 * Resets all arrays to a null pointer.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur.
				 */
				void clear() {
					row_index = nullptr;
					col_start = nullptr;
				}

				/**
				 * @returns A null pointer (since this is a pattern matrix).
				 */
				char * getValues() noexcept {
					return nullptr;
				}

				/**
				 * @returns The index array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				IND * getIndices() noexcept {
					return row_index;
				}

				/**
				 * @returns The offset array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				SIZE * getOffsets() noexcept {
					return col_start;
				}

				/**
				 * @returns Const-version of the offset array.
				 *
				 * \warning Does not check for <tt>NULL</tt> pointers.
				 */
				const SIZE * getOffsets() const noexcept {
					return col_start;
				}

				/**
				 * Gets the current raw pointers of the resizable arrays that are being used
				 * by this instance.
				 */
				void getPointers( void ** pointers ) {
					*pointers++ = nullptr;
					*pointers++ = row_index;
				}

				/**
				 * Replaces the existing arrays with the given ones.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur.
				 */
				void replace( void * const new_vals, void * const new_ind ) {
#ifdef NDEBUG
					(void) new_vals;
#endif
					assert( new_vals == nullptr );
					row_index = static_cast< IND * __restrict__ >( new_ind );
				}

				/**
				 * Replaces an existing start array with a given one.
				 *
				 * Does not perform any actions on pre-existing arrays, if any. Use with
				 * care or memory leaks may occur.
				 */
				void replaceStart( void * const new_start ) {
					col_start = static_cast< SIZE * __restrict__ >( new_start );
				}

				/**
				 * \internal Returns a shorter range compared to the non-pattern version.
				 */
				static inline size_t copyFromRange( const size_t nz, const size_t m ) {
					return nz + m + 1;
				}

				/**
				 * \internal copyFrom specialisation for pattern matrices.
				 */
				template<
					bool unusedValue = false,
					typename InputType, typename InputIND, typename InputSIZE,
					typename UnusedType = std::nullptr_t
				>
				void copyFrom(
					const Compressed_Storage< InputType, InputIND, InputSIZE > &other,
					const size_t nz, const size_t m, const size_t start, size_t end,
					const UnusedType * __restrict__ = nullptr
				) {
					(void) unusedValue;
					// the unusedValue template is meaningless in the case of
					// pattern matrices, but is retained to keep the API
					// the same as with the non-pattern case.
#ifdef _DEBUG
					std::cout << "CompressedStorage::copyFrom (void) called with range "
						<< start << "--" << end << "\n";
#endif
					copyCoordinatesFrom( *this, other, nz, m, start, end );
				}

				/**
				 * Writes a nonzero to the given position. Does \em not update the
				 * \a col_start array. Does not perform any type checking.
				 *
				 * @tparam fwd_it Forward iterator type for this nonzero.
				 *
				 * @param[in] pos   Where to store this nonzero.
				 * @param[in] row Whether to store the row or column index.
				 * @param[in] it  The nonzero iterator.
				 *
				 * This function shall not change the position of the iterator passed.
				 */
				template< typename fwd_it >
				void recordValue( const size_t &pos, const bool row, const fwd_it &it ) {
					row_index[ pos ] = row ? it.i() : it.j();
					// values are ignored for pattern matrices
#ifdef _DEBUG
					std::cout << "\t nonzero at position " << it.i() << " by " << it.j()
						<< " is stored at position " << pos << ". "
						<< "It records no nonzero value as this is a pattern matrix.\n";
#endif
				}

				/**
				 * Returns the size of the raw arrays, in bytes. This function is used to
				 * facilitate memory allocation. No bounds checking of any kind will be
				 * performed.
				 *
				 * @param[out] sizes Where the array byte sizes have to be written to. This
				 *                   must point to an array of at least three values of type
				 *                   \a size_t. The first value refers to the number of
				 *                   bytes required to store the nonzero value array. The
				 *                   second refers to the bytes required to store the
				 *                   minor indices. The third refers to the bytes required
				 *                   to store the major indices.
				 * @param[in] nonzeroes The number of nonzeroes to be stored.
				 */
				void getAllocSize( size_t * sizes, const size_t nonzeroes ) {
					*sizes++ = 0;                         // this is a pattern matrix, so do not
					                                      // allocate values array
					*sizes++ = nonzeroes * sizeof( IND ); // index array
				}

				/**
				 * Returns the size of the start array, in bytes. This function is used to
				 * facilitate memory allocation. No bounds checking of any kind will be
				 * performed.
				 *
				 * @param[out] size     Where the size (in bytes) will be stored.
				 * @param[in]  dim_size The size of the major dimension.
				 */
				void getStartAllocSize( size_t * const size, const size_t dim_size ) {
					*size = ( dim_size + 1 ) * sizeof( SIZE );
				}

				/**
				 * Retrieves the k-th stored nonzero.
				 *
				 * @tparam Ring The generalised semiring under which the nonzeroes of this
				 *              matrix are to be interpreted.
				 *
				 * @param[in] k    The index of the nonzero to return.
				 * @param[in] ring The semiring under which to interpret the requested
				 *                 nonzero.
				 *
				 * @return The requested nonzero, cast to the appropriate value using the
				 *         Semiring::getOne function.
				 */
				template< typename ReturnType >
				inline const ReturnType getValue(
					const size_t, const ReturnType &identity
				) const noexcept {
					return identity;
				}

#ifdef _DEBUG
				/**
				 * For _DEBUG tracing, define a function that prints the value to a string.
				 *
				 * @returns A pretty-printed version of the requested value.
				 */
				inline std::string getPrintValue( const size_t ) const noexcept {
					return "\"1\"";
				}
#endif

				/**
				 * Specialisation for void matrices: function translates to no-op.
				 */
				template< typename D >
				inline void setValue( const size_t k, const D &val ) noexcept {
					(void) k;
					(void) val;
				}

		};

	} // end namespace grb::internal

} // end namespace grb

#endif // end `_H_GRB_REFERENCE_COMPRESSED_STORAGE'

