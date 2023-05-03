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
 * Provides the nonblocking matrix container.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */


#ifndef _H_GRB_NONBLOCKING_MATRIX
#define _H_GRB_NONBLOCKING_MATRIX

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <sstream> //std::stringstream
#include <stdexcept>
#include <utility>

#include <assert.h>

#include <graphblas/algorithms/hpcg/ndim_matrix_builders.hpp>
#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/reference/NonzeroWrapper.hpp>
#include <graphblas/reference/compressed_storage.hpp>
#include <graphblas/reference/init.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/utils/DMapper.hpp>
#include <graphblas/utils/autodeleter.hpp>
#include <graphblas/utils/iterators/utils.hpp>

#include "forward.hpp"

namespace grb {	

	namespace internal {

		//template< typename DataType, typename RIT, typename CIT, typename NIT >
		//Matrix< DataType, reference, RIT, CIT, NIT > & getRefMatrix( Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept;

		//template< typename DataType, typename RIT, typename CIT, typename NIT >
		//const Matrix< DataType, reference, RIT, CIT, NIT > & getRefMatrix( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getNonzeroCapacity( const grb::Matrix< D, nonblocking, RIT, CIT, NIT > & A ) noexcept {
			return A.cap;
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getCurrentNonzeroes( const grb::Matrix< D, nonblocking, RIT, CIT, NIT > & A ) noexcept {
			return A.nz;
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		void setCurrentNonzeroes( grb::Matrix< D, nonblocking, RIT, CIT, NIT > & A, const size_t nnz ) noexcept {
			A.nz = nnz;
		}

		/**
		 * \internal
		 *
		 * Retrieves internal SPA buffers.
		 *
		 * @param[out] coorArr Pointer to the bitmask array
		 * @param[out] coorBuf Pointer to the stack
		 * @param[out] valBuf  Pointer to the value buffer
		 * @param[in]    k     If 0, the row-wise SPA is returned
		 *                     If 1, the column-wise SPA is returned
		 *                     Any other value is not allowed
		 * @param[in]    A     The matrix of which to return the associated SPA
		 *                     data structures.
		 *
		 * @tparam InputType The type of the value buffer.
		 *
		 * \endinternal
		 */
		template< typename InputType, typename RIT, typename CIT, typename NIT >
		void getMatrixBuffers( char *& coorArr, char *& coorBuf, InputType *& valbuf, const unsigned int k, const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > & A ) noexcept {
			assert( k < 2 );
			coorArr = const_cast< char * >( A.coorArr[ k ] );
			coorBuf = const_cast< char * >( A.coorBuf[ k ] );
			valbuf = const_cast< InputType * >( A.valbuf[ k ] );
		}

		template< Descriptor descr,
			bool input_dense,
			bool output_dense,
			bool masked,
			bool left_handed,
			template< typename >
			class One,
			typename IOType,
			class AdditiveMonoid,
			class Multiplication,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename RowColType,
			typename NonzeroType,
			typename Coords >
		void vxm_inner_kernel_scatter( RC & rc,
			Vector< IOType, nonblocking, Coords > & destination_vector,
			IOType * __restrict__ const & destination,
			const size_t & destination_range,
			const Vector< InputType1, nonblocking, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, nonblocking, Coords > & mask_vector,
			const InputType3 * __restrict__ const & mask,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & src_local_to_global,
			const std::function< size_t( size_t ) > & dst_global_to_local );

		template< Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename >
			class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords,
			typename RIT,
			typename CIT,
			typename NIT >
		RC vxm_generic( Vector< IOType, nonblocking, Coords > & u,
			const Vector< InputType3, nonblocking, Coords > & mask,
			const Vector< InputType1, nonblocking, Coords > & v,
			const Vector< InputType4, nonblocking, Coords > & v_mask,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & A,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & row_l2g,
			const std::function< size_t( size_t ) > & row_g2l,
			const std::function< size_t( size_t ) > & col_l2g,
			const std::function< size_t( size_t ) > & col_g2l );

	} // namespace internal

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nrows( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t ncols( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nnz( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	RC resize( Matrix< DataType, nonblocking, RIT, CIT, NIT > &, const size_t ) noexcept;

	template< class ActiveDistribution, typename Func, typename DataType, typename RIT, typename CIT, typename NIT >
	RC eWiseLambda( const Func f, const Matrix< DataType, nonblocking, RIT, CIT, NIT > & A, const size_t s, const size_t P );

	/**
	 * A GraphBLAS matrix, nonblocking implementation.
	 *
	 * Uses Compressed Column Storage (CCS) plus Compressed Row Storage (CRS).
	 *
	 * \warning This implementation prefers speed over memory efficiency.
	 *
	 * @tparam D The type of a nonzero element.
	 *
	 * \internal
	 * @tparam RowIndexType The type used for row indices
	 * @tparam ColIndexType The type used for column indices
	 * @tparam NonzeroIndexType The type used for nonzero indices
	 * \endinternal
	 */
	template< typename D, typename RowIndexType, typename ColIndexType, typename NonzeroIndexType >
	class Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > {

		static_assert( ! grb::is_object< D >::value, "Cannot create an ALP matrix of ALP objects!" );

		//template< typename DataType, typename RIT, typename CIT, typename NIT >
		//friend Matrix< DataType, reference, RIT, CIT, NIT > & internal::getRefMatrix( Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept;

		//template< typename DataType, typename RIT, typename CIT, typename NIT >
		//friend const Matrix< DataType, reference, RIT, CIT, NIT > & internal::getRefMatrix( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept;

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nrows( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t ncols( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nnz( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend RC clear( Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend RC resize( Matrix< DataType, nonblocking, RIT, CIT, NIT > &, const size_t ) noexcept;

		template< class ActiveDistribution, typename Func, typename DataType, typename RIT, typename CIT, typename NIT >
		friend RC eWiseLambda( const Func, const Matrix< DataType, nonblocking, RIT, CIT, NIT > &, const size_t, const size_t );

		template< Descriptor descr,
			bool input_dense,
			bool output_dense,
			bool masked,
			bool left_handed,
			template< typename >
			class One,
			typename IOType,
			class AdditiveMonoid,
			class Multiplication,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename RowColType,
			typename NonzeroType,
			typename Coords >
		friend void internal::vxm_inner_kernel_scatter( RC & rc,
			Vector< IOType, nonblocking, Coords > & destination_vector,
			IOType * __restrict__ const & destination,
			const size_t & destination_range,
			const Vector< InputType1, nonblocking, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, nonblocking, Coords > & mask_vector,
			const InputType3 * __restrict__ const & mask,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & src_local_to_global,
			const std::function< size_t( size_t ) > & dst_global_to_local );

		template< Descriptor descr,
			bool masked,
			bool input_masked,
			bool left_handed,
			template< typename >
			class One,
			class AdditiveMonoid,
			class Multiplication,
			typename IOType,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename InputType4,
			typename Coords,
			typename RIT,
			typename CIT,
			typename NIT >
		friend RC internal::vxm_generic( Vector< IOType, nonblocking, Coords > & u,
			const Vector< InputType3, nonblocking, Coords > & mask,
			const Vector< InputType1, nonblocking, Coords > & v,
			const Vector< InputType4, nonblocking, Coords > & v_mask,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & A,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & row_l2g,
			const std::function< size_t( size_t ) > & row_g2l,
			const std::function< size_t( size_t ) > & col_l2g,
			const std::function< size_t( size_t ) > & col_g2l );

		/* ********************
		        IO friends
		   ******************** */

		template< Descriptor descr, typename InputType, typename RIT, typename CIT, typename NIT, typename fwd_iterator >
		friend RC buildMatrixUnique( Matrix< InputType, nonblocking, RIT, CIT, NIT > &, fwd_iterator, const fwd_iterator, const IOMode );

		friend internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > & internal::getCRS<>( Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & A ) noexcept;

		friend const internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > & internal::getCRS<>(
			const Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & A ) noexcept;

		friend internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > & internal::getCCS<>( Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & A ) noexcept;

		friend const internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > & internal::getCCS<>(
			const Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & A ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getNonzeroCapacity( const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getCurrentNonzeroes( const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::setCurrentNonzeroes( grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &, const size_t ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::getMatrixBuffers( char *&, char *&, InputType *&, const unsigned int, const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend uintptr_t getID( const Matrix< InputType, nonblocking, RIT, CIT, NIT > & );

	private:
		// Matrix< D, reference, RowIndexType, ColIndexType, NonzeroIndexType > ref;

		/** Our own type. */
		typedef Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > self_type;

		/**
		 * \internal Returns the required global buffer size for a matrix of the
		 *           given dimensions.
		 */
		static size_t reqBufSize( const size_t m, const size_t n ) {
			// static checks
			constexpr size_t globalBufferUnitSize = sizeof( RowIndexType ) + sizeof( ColIndexType ) + grb::utils::SizeOf< D >::value;
			static_assert( globalBufferUnitSize >= sizeof( NonzeroIndexType ),
				"We hit here a configuration border case which the implementation does not "
				"handle at present. Please submit a bug report." );
			// compute and return
			return std::max( ( std::max( m, n ) + 1 ) * globalBufferUnitSize,
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::threads() * config::CACHE_LINE_SIZE::value() * utils::SizeOf< D >::value
#else
				static_cast< size_t >( 0 )
#endif
			);
		}

		/** The Row Compressed Storage */
		class internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > CRS;

		/** The Column Compressed Storage */
		class internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > CCS;

		/** The determinstically-obtained ID of this container. */
		uintptr_t id;

		/** Whether to remove #id on destruction. */
		bool remove_id;

		/**
		 * The number of rows.
		 *
		 * \internal Not declared const to be able to implement move in an elegant
		 *           way.
		 */
		size_t m;

		/**
		 * The number of columns.
		 *
		 * \internal Not declared const to be able to implement move in an elegant
		 *           way.
		 */
		size_t n;

		/** The nonzero capacity (in elements). */
		size_t cap;

		/** The current number of nonzeroes. */
		size_t nz;

		/** Array buffer space required for SPA used in symbolic phases. */
		char * __restrict__ coorArr[ 2 ];

		/** Stack buffer space required for SPA used in symbolic phases. */
		char * __restrict__ coorBuf[ 2 ];

		/** Value buffer space required for symbolic phases. */
		D * __restrict__ valbuf[ 2 ];

		/**
		 * Six utils::AutoDeleter objects to free matrix resources automatically
		 * once these go out of scope. We interpret each resource as a block of
		 * bytes, hence we choose \a char as datatype here. The amount of bytes
		 * is controlled by the internal::Compressed_Storage class.
		 */
		utils::AutoDeleter< char > _deleter[ 6 ];

		/**
		 * #utils::AutoDeleter objects that, different from #_deleter, are not
		 * retained e.g. when pinning a matrix.
		 */
		utils::AutoDeleter< char > _local_deleter[ 6 ];

		/**
		 * Internal constructor for manual construction of matrices.
		 *
		 * Should be followed by a manual call to #initialize.
		 */
		// Matrix() : ref() {}
		Matrix() : id( std::numeric_limits< uintptr_t >::max() ), remove_id( false ), m( 0 ), n( 0 ), cap( 0 ), nz( 0 ) {}

		/*
		Matrix( const D * __restrict__ const _values,
		    const ColIndexType * __restrict__ const _column_indices,
		    const NonzeroIndexType * __restrict__ const _offset_array,
		    const size_t _m,
		    const size_t _n,
		    const size_t _cap,
		    char * __restrict__ const buf1 = nullptr,
		    char * __restrict__ const buf2 = nullptr,
		    D * __restrict__ const buf3 = nullptr ) :
		    ref( _values, _column_indices, _offset_array, _m, _n, _cap, buf1, buf2, buf3 ) {}
		*/
		Matrix( const D * __restrict__ const _values,
			const ColIndexType * __restrict__ const _column_indices,
			const NonzeroIndexType * __restrict__ const _offset_array,
			const size_t _m,
			const size_t _n,
			const size_t _cap,
			char * __restrict__ const buf1 = nullptr,
			char * __restrict__ const buf2 = nullptr,
			D * __restrict__ const buf3 = nullptr ) :
			id( std::numeric_limits< uintptr_t >::max() ),
			remove_id( false ), m( _m ), n( _n ), cap( _cap ), nz( _offset_array[ _m ] ), coorArr { nullptr, buf1 }, coorBuf { nullptr, buf2 }, valbuf { nullptr, buf3 } {
			assert( ( _m > 0 && _n > 0 ) || _column_indices[ 0 ] == 0 );
			CRS.replace( _values, _column_indices );
			CRS.replaceStart( _offset_array );
			// CCS is not initialised (and should not be used)
			if( ! internal::template ensureReferenceBufsize< char >( reqBufSize( m, n ) ) ) {
				throw std::runtime_error( "Could not resize global buffer" );
			}
		}

		/**
		 * Takes care of the initialisation of a new matrix.
		 */
		void initialize( const uintptr_t * const id_in, const size_t rows, const size_t cols, const size_t cap_in ) {
#ifdef _DEBUG
			std::cerr << "\t in Matrix< nonblocking >::initialize...\n"
					  << "\t\t matrix size " << rows << " by " << cols << "\n"
					  << "\t\t requested capacity " << cap_in << "\n";				  
#endif

			// dynamic checks
			assert( id == std::numeric_limits< uintptr_t >::max() );
			assert( ! remove_id );
			if( rows >= static_cast< size_t >( std::numeric_limits< RowIndexType >::max() ) ) {
				throw std::overflow_error( "Number of rows larger than configured "
										   "RowIndexType maximum!" );
			}
			if( cols >= static_cast< size_t >( std::numeric_limits< ColIndexType >::max() ) ) {
				throw std::overflow_error( "Number of columns larger than configured "
										   "ColIndexType maximum!" );
			}

			// memory allocations
			RC alloc_ok = SUCCESS;
			char * alloc[ 8 ] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
			/*
			if( ! internal::template ensureReferenceBufsize< char >( reqBufSize( rowscoorBufcoorBuf, cols ) ) ) {
				throw std::runtime_error( "Could not resize global buffer" );
			}
			*/
			if( rows > 0 && cols > 0 ) {
				// check whether requested capacity is sensible
				if( cap_in / rows > cols || cap_in / cols > rows || ( cap_in / rows == cols && ( cap_in % rows > 0 ) ) || ( cap_in / cols == rows && ( cap_in % cols > 0 ) ) ) {
#ifdef _DEBUG
					std::cerr << "\t\t Illegal capacity requested\n";
#endif
					throw std::runtime_error( toString( ILLEGAL ) );
				}
				// get sizes of arrays that we need to allocate
				size_t sizes[ 12 ];
				sizes[ 0 ] = internal::Coordinates< nonblocking >::arraySize( rows );
				sizes[ 1 ] = internal::Coordinates< nonblocking >::arraySize( cols );
				sizes[ 2 ] = internal::Coordinates< nonblocking >::bufferSize( rows );
				sizes[ 3 ] = internal::Coordinates< nonblocking >::bufferSize( cols );
				sizes[ 4 ] = rows * internal::SizeOf< D >::value;
				sizes[ 5 ] = cols * internal::SizeOf< D >::value;
				CRS.getStartAllocSize( &( sizes[ 6 ] ), rows );
				CCS.getStartAllocSize( &( sizes[ 7 ] ), cols );
				if( cap_in > 0 ) {
					CRS.getAllocSize( &( sizes[ 8 ] ), cap_in );
					CCS.getAllocSize( &( sizes[ 10 ] ), cap_in );
				} else {
					sizes[ 8 ] = sizes[ 9 ] = sizes[ 10 ] = sizes[ 11 ] = 0;
				}
				// allocate required arrays
				alloc_ok = utils::alloc( "grb::Matrix< T, nonblocking >::Matrix()", "initial capacity allocation", coorArr[ 0 ], sizes[ 0 ], false, _local_deleter[ 0 ], coorArr[ 1 ], sizes[ 1 ], false,
					_local_deleter[ 1 ], coorBuf[ 0 ], sizes[ 2 ], false, _local_deleter[ 2 ], coorBuf[ 1 ], sizes[ 3 ], false, _local_deleter[ 3 ], alloc[ 6 ], sizes[ 4 ], false, _local_deleter[ 4 ],
					alloc[ 7 ], sizes[ 5 ], false, _local_deleter[ 5 ], alloc[ 0 ], sizes[ 6 ], true, _deleter[ 0 ], alloc[ 1 ], sizes[ 7 ], true, _deleter[ 1 ], alloc[ 2 ], sizes[ 8 ], true,
					_deleter[ 2 ], alloc[ 3 ], sizes[ 9 ], true, _deleter[ 3 ], alloc[ 4 ], sizes[ 10 ], true, _deleter[ 4 ], alloc[ 5 ], sizes[ 11 ], true, _deleter[ 5 ] );
			} else {
				const size_t sizes[ 2 ] = { rows * internal::SizeOf< D >::value, cols * internal::SizeOf< D >::value };
				coorArr[ 0 ] = coorArr[ 1 ] = nullptr;
				coorBuf[ 0 ] = coorBuf[ 1 ] = nullptr;
				alloc_ok = utils::alloc(
					"grb::Matrix< T, nonblocking >::Matrix()", "empty allocation", alloc[ 6 ], sizes[ 0 ], false, _local_deleter[ 4 ], alloc[ 7 ], sizes[ 1 ], false, _local_deleter[ 5 ] );
			}

			// check allocation status
			if( alloc_ok == OUTOFMEM ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix construction" );
			} else if( alloc_ok != SUCCESS ) {
				throw std::runtime_error( toString( alloc_ok ) );
			}
#ifdef _DEBUG
			if( rows > 0 && cols > 0 ) {
				std::cerr << "\t\t allocations for an " << m << " by " << n << " matrix "
						  << "have successfully completed.\n";
			} else {
				std::cerr << "\t\t allocations for an empty matrix have successfully "
						  << "completed.\n";
			}
#endif
			// either set ID or retrieve one
			if( id_in != nullptr ) {
				assert( ! remove_id );
				id = *id_in;
#ifdef _DEBUG
				std::cerr << "\t\t inherited ID " << id << "\n";
#endif
			} else {
				if( rows > 0 && cols > 0 && id_in == nullptr ) {
					id = internal::reference_mapper.insert( reinterpret_cast< uintptr_t >( alloc[ 0 ] ) );
					remove_id = true;
#ifdef _DEBUG
					std::cerr << "\t\t assigned new ID " << id << "\n";
#endif
				}
			}

			// all OK, so set and exit
			m = rows;
			n = cols;
			nz = 0;
			if( m > 0 && n > 0 ) {
				cap = cap_in;
			}
			valbuf[ 0 ] = reinterpret_cast< D * >( alloc[ 6 ] );
			valbuf[ 1 ] = reinterpret_cast< D * >( alloc[ 7 ] );
			if( m > 0 && n > 0 ) {
				CRS.replaceStart( alloc[ 0 ] );
				CCS.replaceStart( alloc[ 1 ] );
				CRS.replace( alloc[ 2 ], alloc[ 3 ] );
				CCS.replace( alloc[ 4 ], alloc[ 5 ] );
			}
		}

		/** Implements a move. */
		/*void moveFromOther( self_type && other ) {
		    ref.moveFromOther( std::move( other.ref ) );
		}
		*/
		void moveFromOther( self_type && other ) {
			// move from other
			CRS = std::move( other.CRS );
			CCS = std::move( other.CCS );
			id = other.id;
			remove_id = other.remove_id;
			m = other.m;
			n = other.n;
			cap = other.cap;
			nz = other.nz;
			for( unsigned int i = 0; i < 2; ++i ) {
				coorArr[ i ] = other.coorArr[ i ];
				coorBuf[ i ] = other.coorBuf[ i ];
				valbuf[ i ] = other.valbuf[ i ];
			}
			for( unsigned int i = 0; i < 6; ++i ) {
				_deleter[ i ] = std::move( other._deleter[ i ] );
				_local_deleter[ i ] = std::move( other._local_deleter[ i ] );
			}

			// invalidate other fields
			for( unsigned int i = 0; i < 2; ++i ) {
				other.coorArr[ i ] = other.coorBuf[ i ] = nullptr;
				other.valbuf[ i ] = nullptr;
			}
			other.id = std::numeric_limits< uintptr_t >::max();
			other.remove_id = false;
			other.m = 0;
			other.n = 0;
			other.cap = 0;
			other.nz = 0;
		}

		/**
		 * Sets CRS and CCS offset arrays to zero.
		 *
		 * Does not clear any other field.
		 *
		 * It relies on the values stored in the m and n fields for the sizes.
		 */
		void clear_cxs_offsets() {
			// two-phase strategy: fill until minimum, then continue filling the larger
			// array.
			size_t min_dim = static_cast< size_t >( std::min( m, n ) );
			size_t max_dim = static_cast< size_t >( std::max( m, n ) );
			NonzeroIndexType * const larger = max_dim == m ? CRS.col_start : CCS.col_start;

			// fill until minimum
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, min_dim );
#else
			size_t start = 0;
			size_t end = min_dim;
#endif
				for( size_t i = start; i < end; ++i ) {
					CRS.col_start[ i ] = 0;
					CCS.col_start[ i ] = 0;
				}
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::localRange( start, end, min_dim, max_dim );
#else
			start = min_dim;
			end = max_dim;
#endif
				for( size_t i = start; i < end; ++i ) {
					larger[ i ] = 0;
				}
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
			}
#endif
		}

		/** @see Matrix::clear */
		/*
		RC clear() {
		    return ref.clear();
		}
		*/
		RC clear() {
			// update nonzero count
			nz = 0;

			// catch trivial case
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}

			// catch uninitialised case
			if( CRS.col_start == nullptr || CCS.col_start == nullptr ) {
				// sanity check
				assert( CRS.col_start == nullptr && CCS.col_start == nullptr );
			} else {
				// clear offsets
				clear_cxs_offsets();
			}

			// done
			return SUCCESS;
		}

		/** @see grb::resize() */
		/*
		RC resize( const size_t nonzeroes ) {
		    return ref.resize( nonzeroes );
		}
		*/
		RC resize( const size_t nonzeroes ) {
			std::cout << "NONBLOCKING/matrix.hpp - > resize() has been called " << std :: endl;
			// check for trivial case
			if( m == 0 || n == 0 || nonzeroes == 0 ) {
				// simply do not do anything and return
				return SUCCESS;
			}

			// do not do anything if current capacity is sufficient
			if( nonzeroes <= cap ) {
				return SUCCESS;
			}

			if( nonzeroes >= static_cast< size_t >( std::numeric_limits< NonzeroIndexType >::max() ) ) {
				return OVERFLW;
			}

			// allocate and catch errors
			char * alloc[ 4 ] = { nullptr, nullptr, nullptr, nullptr };
			size_t sizes[ 4 ];
			// cache old allocation data
			size_t old_sizes[ 4 ] = { 0, 0, 0, 0 };
			size_t freed = 0;
			if( cap > 0 ) {
				CRS.getAllocSize( &( old_sizes[ 0 ] ), cap );
				CCS.getAllocSize( &( old_sizes[ 2 ] ), cap );
			}

			// compute new required sizes
			CRS.getAllocSize( &( sizes[ 0 ] ), nonzeroes );
			CCS.getAllocSize( &( sizes[ 2 ] ), nonzeroes );

			// construct a description of the matrix we are allocating for
			std::stringstream description;
			description << ", for " << nonzeroes << " nonzeroes in an " << m << " "
						<< "times " << n << " matrix.\n";

			// do allocation
			RC ret = utils::alloc( "grb::Matrix< T, nonblocking >::resize", description.str(), alloc[ 0 ], sizes[ 0 ], true, _deleter[ 2 ], alloc[ 1 ], sizes[ 1 ], true, _deleter[ 3 ], alloc[ 2 ],
				sizes[ 2 ], true, _deleter[ 4 ], alloc[ 3 ], sizes[ 3 ], true, _deleter[ 5 ] );

			if( ret != SUCCESS ) {
				// exit function without side-effects
				return ret;
			}

			// put allocated arrays in their intended places
			CRS.replace( alloc[ 0 ], alloc[ 1 ] );
			CCS.replace( alloc[ 2 ], alloc[ 3 ] );

			// if we had old data emplaced
			if( cap > 0 ) {
				for( unsigned int i = 0; i < 4; ++i ) {
					if( old_sizes[ i ] > 0 ) {
						freed += sizes[ i ];
					}
				}
				if( config::MEMORY::report( "grb::Matrix< T, nonblocking >::resize", "freed (or will eventually free)", freed, false ) ) {
					std::cout << ", for " << cap << " nonzeroes "
							  << "that this matrix previously contained.\n";
				}
			}

			// set new capacity
			cap = nonzeroes;

			// done, return error code
			return SUCCESS;
		}

		/**
		 * @see Matrix::buildMatrixUnique.
		 *
		 * This dispatcher calls the sequential or the parallel implementation based
		 * on the tag of the input iterator of type \p input_iterator.
		 */
		/*
		template< Descriptor descr = descriptors::no_operation, typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end ) {

		    return ref.buildMatrixUnique( _start, _end );
		}
		*/
		template< Descriptor descr = descriptors::no_operation, typename InputIterator >
		RC buildMatrixUnique( const InputIterator & _start, const InputIterator & _end, const IOMode mode ) {
			// here we can safely ignore the mode and dispatch based only on the
			// iterator type since in shared memory the input data reside by definition
			// all on the same machine
			(void)mode;
			static_assert( utils::is_alp_matrix_iterator< D, InputIterator >::value,
				"the given iterator is not a valid input iterator, "
				"see the ALP specification for input iterators" );
			typename std::iterator_traits< InputIterator >::iterator_category category;
			return buildMatrixUniqueImpl( _start, _end, category );
		}

		/**
		 * When given a forward iterator tag, calls the sequential implementation of
		 * buildMatrixUnique.
		 */
		template< typename fwd_iterator >
		RC buildMatrixUniqueImpl( const fwd_iterator & _start, const fwd_iterator & _end, std::forward_iterator_tag ) {
			return buildMatrixUniqueImplSeq( _start, _end );
		}

		/**
		 * The sequential implementation of buildMatrixUnique.
		 */
		template< typename fwd_iterator >
		RC buildMatrixUniqueImplSeq( const fwd_iterator & _start, const fwd_iterator & _end ) {
#ifdef _DEBUG
			std::cout << " fwrd acces iterator " << '\n';
			std::cout << "buildMatrixUnique called with " << cap << " nonzeroes.\n";
			std::cout << "buildMatrixUnique: input is\n";
			for( fwd_iterator it = _start; it != _end; ++it ) {
				std::cout << "\t" << it.i() << ", " << it.j() << "\n";
			}
			std::cout << "buildMatrixUnique: end input.\n";
#endif
			// detect trivial case
			if( _start == _end || m == 0 || n == 0 ) {
				return SUCCESS;
			}
			// keep count of nonzeroes
			nz = 0;

			// counting sort, phase 1
			clear_cxs_offsets();
			for( fwd_iterator it = _start; it != _end; ++it ) {
				if( utils::check_input_coordinates( it, m, n ) != SUCCESS ) {
					return MISMATCH;
				}
				(void)++( CRS.col_start[ it.i() ] );
				(void)++( CCS.col_start[ it.j() ] );
				(void)++nz;
			}

			// check if we can indeed store nz values
			if( nz >= static_cast< size_t >( std::numeric_limits< grb::config::NonzeroIndexType >::max() ) ) {
				return OVERFLW;
			}

			// put final entries in offset arrays
			CRS.col_start[ m ] = nz;
			CCS.col_start[ n ] = nz;

			// allocate enough space
			resize( nz );

			// make counting sort array cumulative
			for( size_t i = 1; i < m; ++i ) {
#ifdef _DEBUG
				std::cout << "There are " << CRS.col_start[ i ] << " "
						  << "nonzeroes at row " << i << "\n";
#endif
				CRS.col_start[ i ] += CRS.col_start[ i - 1 ];
			}

			// make counting sort array cumulative
			for( size_t i = 1; i < n; ++i ) {
#ifdef _DEBUG
				std::cout << "There are " << CCS.col_start[ i ] << " "
						  << "nonzeroes at column " << i << "\n";
#endif
				CCS.col_start[ i ] += CCS.col_start[ i - 1 ];
			}

			// counting sort, phase 2
			fwd_iterator it = _start;
			for( size_t k = 0; it != _end; ++k, ++it ) {
				const size_t crs_pos = --( CRS.col_start[ it.i() ] );
				CRS.recordValue( crs_pos, false, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
						  << "is stored at CRS position " << static_cast< size_t >( crs_pos ) << ".\n";
#endif
				const size_t ccs_pos = --( CCS.col_start[ it.j() ] );
				CCS.recordValue( ccs_pos, true, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) "
						  << "is stored at CCS position " << static_cast< size_t >( ccs_pos ) << ".\n";
#endif
			}
#ifdef _DEBUG
			for( size_t i = 0; i <= m; ++i ) {
				std::cout << "row_start[ " << i << " ] = " << CRS.col_start[ i ] << "." << std::endl;
			}
			for( size_t i = 0; i <= n; ++i ) {
				std::cout << "col_start[ " << i << " ] = " << CCS.col_start[ i ] << "." << std::endl;
			}
#endif
			// done
			return SUCCESS;
		}

	public:
		/** @see Matrix::value_type */
		typedef D value_type;

		// Matrix( const size_t rows, const size_t columns, const size_t nz ) : ref( rows, columns, nz ) {}
		Matrix( const size_t rows, const size_t columns, const size_t nz ) : Matrix() {
#ifdef _DEBUG
			std::cout << "In grb::Matrix constructor (nonblocking, with requested "
					  << "capacity)\n";
#endif			
			initialize( nullptr, rows, columns, nz );
		}

		// Matrix( const size_t rows, const size_t columns ) : ref( rows, columns ) {}
		Matrix( const size_t rows, const size_t columns ) : Matrix( rows, columns, std::max( rows, columns ) ) {
#ifdef _DEBUG
			std::cout << "In grb::Matrix constructor (nonblocking, default capacity)\n";
#endif
		}

		/**
		 * \internal
		 * \todo See below code comment
		 * \endinternal
		 */
		/*
		Matrix( const Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & other ) : ref( other.ref ) {
		    // TODO: the pipeline should be executed once level-3 primitives are
		    //       implemented. In the current implementation matrices may be used only
		    //       as the input of SpMV
		}
		*/

		/**
		 * \parblock
		 * \par Performance semantics
		 * This backend specifies the following performance semantics for this
		 * constructor:
		 *   -# first, the performance semantics of a constructor call with arguments
		 *          nrows( other ), ncols( other ), capacity( other )
		 *      applies.
		 *   -# then, the performance semantics of a call to grb::set apply.
		 * \endparblock
		 */
		Matrix( const Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > & other ) : Matrix( other.m, other.n, other.cap ) {
#ifdef _DEBUG
			std::cerr << "In grb::Matrix (nonblocking) copy-constructor\n"
					  << "\t source matrix has " << other.nz << " nonzeroes\n";
#endif
			nz = other.nz;

			// if empty, return; otherwise copy
			if( nz == 0 ) {
				return;
			}

#ifdef _H_GRB_REFERENCE_OMP_MATRIX
#pragma omp parallel
#endif
			{
				size_t range = CRS.copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				size_t start, end;
				config::OMP::localRange( start, end, 0, range );
#else
				const size_t start = 0;
				size_t end = range;
#endif
				CRS.copyFrom( other.CRS, nz, m, start, end );
				range = CCS.copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::localRange( start, end, 0, range );
#else
				end = range;
#endif
				CCS.copyFrom( other.CCS, nz, n, start, end );
			}
		}

		/*
		Matrix( self_type && other ) noexcept : ref( std::move( other.ref ) ) {
		    // TODO: the pipeline should be executed once level-3 primitives are
		    //       implemented. In the current implementation matrices may be used only
		    //       as the input of SpMV
		}
		*/

		/** \internal No implementation notes. */
		Matrix( self_type && other ) noexcept {
			moveFromOther( std::forward< self_type >( other ) );
		}

		/*
		self_type & operator=( self_type && other ) noexcept {
		    ref = std::move( other.ref );
		    return *this;
		}
		*/
		/** \internal No implementation notes. */
		self_type & operator=( self_type && other ) noexcept {
			moveFromOther( std::forward< self_type >( other ) );
			return *this;
		}

		~Matrix() {
			// the pipeline is executed before memory deallocation
			internal::le.execution( this );
		}

		/*
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		begin( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
		    return ref.begin( mode, s, P );
		}
		*/

		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 *       (GitHub issue 32)
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		begin( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > IteratorType;
#ifdef _DEBUG
			std::cout << "In grb::Matrix<T,reference>::cbegin\n";
#endif
			return IteratorType( CRS, m, n, nz, false, s, P );
		}

		/*
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		end( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
		    return ref.end( mode, s, P );
		}
		*/
		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 *       (GitHub issue 32)
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		end( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > IteratorType;
			return IteratorType( CRS, m, n, nz, true, s, P );
		}

		/*
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > cbegin( const IOMode mode = PARALLEL ) const {
		    return ref.cbegin( mode );
		}
		*/
		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 *       (GitHub issue 32)
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > cbegin( const IOMode mode = PARALLEL ) const {
			return begin< ActiveDistribution >( mode );
		}

		/*
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > cend( const IOMode mode = PARALLEL ) const {
		    return ref.cend( mode );
		}
		*/
		/**
		 * \internal No implementation notes.
		 *
		 * \todo should we specify performance semantics for retrieving iterators?
		 *       (GitHub issue 32)
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType >::template ConstIterator< ActiveDistribution > cend( const IOMode mode = PARALLEL ) const {
			return end< ActiveDistribution >( mode );
		}
	};

	// template specialisation for GraphBLAS type traits
	template< typename D, typename RIT, typename CIT, typename NIT >
	struct is_container< Matrix< D, nonblocking, RIT, CIT, NIT > > {
		/** A nonblocking Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	// internal getters implementation
	namespace internal {
		/*
		template< typename DataType, typename RIT, typename CIT, typename NIT >
		inline Matrix< DataType, reference, RIT, CIT, NIT > & getRefMatrix( Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept {
		    return ( A.ref );
		}

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		inline const Matrix< DataType, reference, RIT, CIT, NIT > & getRefMatrix( const Matrix< DataType, nonblocking, RIT, CIT, NIT > & A ) noexcept {
		    return ( A.ref );
		}
		*/

	} // namespace internal

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_MATRIX''
