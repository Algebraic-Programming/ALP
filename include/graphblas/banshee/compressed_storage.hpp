
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
 */

#ifndef _H_GRB_BANSHEE_COMPRESSED_STORAGE
#define _H_GRB_BANSHEE_COMPRESSED_STORAGE

namespace grb {

	namespace internal {

		/**
		 * Basic functionality for a compressed storage format (CRS/CSR or CCS/CSC).
		 *
		 * FOR INTERNAL USE ONLY.
		 *
		 * This is a very unsafe wrapper class around three arrays. Use with care.
		 *
		 * @tparam D    The nonzero value types.
		 * @tparam IND  The matrix coordinate type.
		 * @tparam SIZE The nonzero type.
		 *
		 * The matrix dimension must be encodeable in \a IND. The number of nonzeroes
		 * must be encodeable in \a SIZE.
		 */
		template< typename D, typename IND, typename SIZE >
		class Compressed_Storage {

		private:
			/**
			 * Resets all arrays to NULL.
			 *
			 * Does not perform any actions on pre-existing arrays, if any. Use with
			 * care or memory leaks may occur.
			 */
			void clear() {
				values = NULL;
				row_index = NULL;
				col_start = NULL;
			}

		public:
			/** The value array. */
			D * __restrict__ values;

			/** The row index values. */
			IND * __restrict__ row_index;

			/** The column start indices. */
			SIZE * __restrict__ col_start;

			/** Base constructor (NULL-initialiser). */
			Compressed_Storage() : values( NULL ), row_index( NULL ), col_start( NULL ) {}

			/** Non-shallow copy constructor. */
			explicit Compressed_Storage( const Compressed_Storage< D, IND, SIZE > & other ) : values( other.values ), row_index( other.row_index ), col_start( other.col_start ) {}

			/** Move constructor. */
			Compressed_Storage( Compressed_Storage< D, IND, SIZE > && other ) : values( other.values ), row_index( other.row_index ), col_start( other.col_start ) {
				// the values are now owned by this container, so no memory leak when clear is called
				other.clear();
			}

			/**
			 * @returns The value array.
			 *
			 * \warning Does not check for <tt>NULL</tt> pointers.
			 */
			D * getValues() const noexcept {
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
			 * Gets the current raw pointers of the resizable arrays that are being used by this instance.
			 */
			void getPointers( void ** pointers ) {
				*pointers++ = values;
				*pointers++ = row_index;
			}
			/**
			 * Replaces the existing arrays with the given ones.
			 *
			 * Does not perform any actions on pre-existing arrays, if any. Use with
			 * care or memory leaks may occur.
			 */
			void replace( void * __restrict__ const new_vals, void * __restrict__ const new_ind ) {
				values = static_cast< D * __restrict__ >( new_vals );
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
			 * Copies contents from a given Compressed_Storage.
			 *
			 * Performs no safety checking. Performs no (re-)allocations.
			 *
			 * @param[in] other The container to copy from.
			 * @param[in] nz    The number of nonzeroes in the \a other container.
			 * @param[in] m     The index dimension of the \a other container.
			 */
			void copyFrom( const Compressed_Storage< D, IND, SIZE > & other, const size_t nz, const size_t m ) {
				memcpy( values, other.values, static_cast< size_t >( nz ) * sizeof( D ) );
				memcpy( row_index, other.row_index, static_cast< size_t >( nz ) * sizeof( IND ) );
				memcpy( col_start, other.col_start, ( static_cast< size_t >( m ) + 1 ) * sizeof( SIZE ) );
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
			void recordValue( const size_t & pos, const bool row, const fwd_it & it ) {
				row_index[ pos ] = row ? it.i() : it.j();
				values[ pos ] = it.v();
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
			inline const ReturnType getValue( const size_t k, const ReturnType & ) const noexcept {
				return static_cast< ReturnType >( values[ k ] );
			}

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

			/** Base constructor (NULL-initialiser). */
			Compressed_Storage() : row_index( NULL ), col_start( NULL ) {}

			/** Non-shallow copy constructor. */
			explicit Compressed_Storage( const Compressed_Storage< void, IND, SIZE > & other ) : row_index( other.row_index ), col_start( other.col_start ) {}

			/** Move constructor. */
			Compressed_Storage( Compressed_Storage< void, IND, SIZE > && other ) : row_index( other.row_index ), col_start( other.col_start ) {
				// the values are now owned by this container, so no memory leak when clear is called
				other.clear();
			}

			/**
			 * Resets all arrays to NULL.
			 *
			 * Does not perform any actions on pre-existing arrays, if any. Use with
			 * care or memory leaks may occur.
			 */
			void clear() {
				row_index = NULL;
				col_start = NULL;
			}

			/**
			 * @returns <tt>NULL</tt> (since this is a pattern matrix).
			 */
			char * getValues() const noexcept {
				return NULL;
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
			 * Gets the current raw pointers of the resizable arrays that are being used by this instance.
			 */
			void getPointers( void ** pointers ) {
				*pointers++ = NULL;
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
				(void)new_vals;
#endif
				assert( new_vals == NULL );
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
			 * Copies contents from a given Compressed_Storage.
			 *
			 * Performs no safety checking. Performs no (re-)allocations.
			 *
			 * @param[in] other The container to copy from.
			 * @param[in] nz    The number of nonzeroes in the \a other container.
			 * @param[in] m     The index dimension of the \a other container.
			 */
			void copyFrom( const Compressed_Storage< void, IND, SIZE > & other, const size_t nz, const size_t m ) {
				memcpy( row_index, other.row_index, static_cast< size_t >( nz ) * sizeof( IND ) );
				memcpy( col_start, other.col_start, ( static_cast< size_t >( m ) + 1 ) * sizeof( SIZE ) );
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
			void recordValue( const size_t & pos, const bool row, const fwd_it & it ) {
				row_index[ pos ] = row ? it.i() : it.j();
				// values are ignored for pattern matrices
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
				*sizes++ = 0;                         // this is a pattern matrix, so do not allocate values array
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
			inline const ReturnType getValue( const size_t, const ReturnType & identity ) const noexcept {
				return identity;
			}

			/**
			 * Specialisation for void matrices: function translates to no-op.
			 */
			template< typename D >
			inline void setValue( const size_t k, const D & val ) noexcept {
				(void)k;
				(void)val;
			}
		};

	} // namespace internal

} // namespace grb

#endif // end `_H_GRB_BANSHEE_COMPRESSED_STORAGE'
