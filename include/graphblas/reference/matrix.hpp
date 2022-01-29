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

#if ! defined _H_GRB_REFERENCE_MATRIX || defined _H_GRB_REFERENCE_OMP_MATRIX
#define _H_GRB_REFERENCE_MATRIX

#ifdef _DEBUG
#include <cstdio>
#endif

#include <numeric> //std::accumulate
#include <sstream> //std::stringstream

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/reference/blas1.hpp>
#include <graphblas/reference/compressed_storage.hpp>
#include <graphblas/reference/init.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/autodeleter.hpp>
#include <graphblas/utils/pattern.hpp> //for help with dealing with pattern matrix input

#include "forward.hpp"

namespace grb {

#ifndef _H_GRB_REFERENCE_OMP_MATRIX
	namespace internal {

		template< typename D >
		class SizeOf {
		public:
			static constexpr size_t value = sizeof( D );
		};

		template<>
		class SizeOf< void > {
		public:
			static constexpr size_t value = 0;
		};

	} // namespace internal
#endif

	namespace internal {

		template< typename D >
		size_t & getNonzeroCapacity( grb::Matrix< D, reference > & A ) noexcept {
			return A.cap;
		}
		template< typename D >
		size_t & getCurrentNonzeroes( grb::Matrix< D, reference > & A ) noexcept {
			return A.nz;
		}
		template< typename D >
		void setCurrentNonzeroes( grb::Matrix< D, reference > & A, const size_t nnz ) noexcept {
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
		template< typename InputType >
		void getMatrixBuffers(
			char * &coorArr, char * &coorBuf, InputType * &valbuf,
			const unsigned int k, const grb::Matrix< InputType, reference > &A
		) noexcept {
			coorArr = const_cast< char * >( A.coorArr[ k ] );
			coorBuf = const_cast< char * >( A.coorBuf[ k ] );
			valbuf = const_cast< InputType * >( A.valbuf[ k ] );
		}

		template< Descriptor descr,
			bool input_dense, bool output_dense,
			bool masked,
			bool left_handed,
			template< typename > class One,
			typename IOType,
			class AdditiveMonoid, class Multiplication,
			typename InputType1, typename InputType2, typename InputType3,
			typename RowColType, typename NonzeroType,
			typename Coords
		>
		void vxm_inner_kernel_scatter( RC & rc,
			Vector< IOType, reference, Coords > & destination_vector,
			IOType * __restrict__ const & destination,
			const size_t & destination_range,
			const Vector< InputType1, reference, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, reference, Coords > & mask_vector,
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
			typename Coords >
		RC vxm_generic( Vector< IOType, reference, Coords > & u,
			const Vector< InputType3, reference, Coords > & mask,
			const Vector< InputType1, reference, Coords > & v,
			const Vector< InputType4, reference, Coords > & v_mask,
			const Matrix< InputType2, reference > & A,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & row_l2g,
			const std::function< size_t( size_t ) > & row_g2l,
			const std::function< size_t( size_t ) > & col_l2g,
			const std::function< size_t( size_t ) > & col_g2l );
	} // namespace internal

	template< typename DataType >
	size_t nrows( const Matrix< DataType, reference > & ) noexcept;

	template< typename DataType >
	size_t ncols( const Matrix< DataType, reference > & ) noexcept;

	template< typename DataType >
	size_t nnz( const Matrix< DataType, reference > & ) noexcept;

	template< typename InputType >
	RC clear( Matrix< InputType, reference > & ) noexcept;

	template< typename DataType >
	RC resize( Matrix< DataType, reference > &, const size_t ) noexcept;

	template< class ActiveDistribution, typename Func, typename DataType >
	RC eWiseLambda( const Func f, const Matrix< DataType, reference > & A, const size_t s, const size_t P );

	/**
	 * A GraphBLAS matrix, reference implementation.
	 *
	 * Uses Compressed Column Storage (CCS) plus Compressed Row Storage (CRS).
	 *
	 * \warning This implementation prefers speed over memory efficiency.
	 *
	 * @tparam D  The type of a nonzero element.
	 */
	template< typename D >
	class Matrix< D, reference > {

		static_assert( ! grb::is_object< D >::value, "Cannot create a GraphBLAS matrix of GraphBLAS objects!" );

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, reference > & ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, reference > & ) noexcept;

		template< typename DataType >
		friend size_t nnz( const Matrix< DataType, reference > & ) noexcept;

		template< typename InputType >
		friend RC clear( Matrix< InputType, reference > & ) noexcept;

		template< typename DataType >
		friend RC resize( Matrix< DataType, reference > &, const size_t ) noexcept;

		template< class ActiveDistribution, typename Func, typename DataType >
		friend RC eWiseLambda( const Func, const Matrix< DataType, reference > &, const size_t, const size_t );

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
			Vector< IOType, reference, Coords > & destination_vector,
			IOType * __restrict__ const & destination,
			const size_t & destination_range,
			const Vector< InputType1, reference, Coords > & source_vector,
			const InputType1 * __restrict__ const & source,
			const size_t & source_index,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > & matrix,
			const Vector< InputType3, reference, Coords > & mask_vector,
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
			typename Coords >
		friend RC internal::vxm_generic( Vector< IOType, reference, Coords > & u,
			const Vector< InputType3, reference, Coords > & mask,
			const Vector< InputType1, reference, Coords > & v,
			const Vector< InputType4, reference, Coords > & v_mask,
			const Matrix< InputType2, reference > & A,
			const AdditiveMonoid & add,
			const Multiplication & mul,
			const std::function< size_t( size_t ) > & row_l2g,
			const std::function< size_t( size_t ) > & row_g2l,
			const std::function< size_t( size_t ) > & col_l2g,
			const std::function< size_t( size_t ) > & col_g2l );

		/* ********************
		        IO friends
		   ******************** */

		template< Descriptor descr, typename InputType, typename fwd_iterator >
		friend RC buildMatrixUnique( Matrix< InputType, reference > &, fwd_iterator, const fwd_iterator, const IOMode );

		friend internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & internal::getCRS<>( Matrix< D, reference > & A ) noexcept;

		friend const internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & internal::getCRS<>( const Matrix< D, reference > & A ) noexcept;

		friend internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & internal::getCCS<>( Matrix< D, reference > & A ) noexcept;

		friend const internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & internal::getCCS<>( const Matrix< D, reference > & A ) noexcept;

		template< typename InputType >
		friend size_t & internal::getNonzeroCapacity( grb::Matrix< InputType, reference > & ) noexcept;

		template< typename InputType >
		friend size_t & internal::getCurrentNonzeroes( grb::Matrix< InputType, reference > & ) noexcept;

		template< typename InputType >
		friend void internal::setCurrentNonzeroes( grb::Matrix< InputType, reference > &, const size_t ) noexcept;

		template< typename InputType >
		friend void internal::getMatrixBuffers( char *&, char *&, InputType *&, const unsigned int, const grb::Matrix< InputType, reference > & ) noexcept;

	private:
		/** Our own type. */
		typedef Matrix< D, reference > self_type;

		/** The Row Compressed Storage */
		class internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > CRS;

		/** The Column Compressed Storage */
		class internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > CCS;

		/**
		 * The number of rows.
		 *
		 * \internal Not declared const to be able to implement move in an elegant way.
		 */
		size_t m;

		/**
		 * The number of columns.
		 *
		 * \internal Not declared const to be able to implement move in an elegant way.
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

		/** Implements a move. */
		void moveFromOther( self_type &&other ) {
			// move from other
			CRS = std::move( other.CRS );
			CCS = std::move( other.CCS );
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
			}

			// invalidate other fields
			for( unsigned int i = 0; i < 2; ++i ) {
				other.coorArr[ i ] = other.coorBuf[ i ] = nullptr;
				other.valbuf[ i ] = nullptr;
			}
			other.m = 0;
			other.n = 0;
			other.cap = 0;
			other.nz = 0;
		}

		/** @see Matrix::clear */
		RC clear() {
			// update nonzero count
			nz = 0;

			// catch trivial case
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}

			// catch uninitialised case
			if( CRS.col_start == NULL || CCS.col_start == NULL ) {
				// sanity check
				assert( CRS.col_start == NULL && CCS.col_start == NULL );
				// nothing to do
				return SUCCESS;
			}

			// do both CRS and CCS simultaneously as far as possible
			const size_t min_dim = ( m < n ) ? m : n;
			for( size_t i = 0; i < min_dim; ++i ) {
				CRS.col_start[ i ] = CCS.col_start[ i ] = 0;
			}

			// do last remainder of either CRS or CCS
			if( min_dim == m ) {
				for( size_t i = min_dim; i < n; ++i ) {
					CCS.col_start[ i ] = 0;
				}
			} else {
				for( size_t i = min_dim; i < m; ++i ) {
					CRS.col_start[ i ] = 0;
				}
			}

			// done
			return SUCCESS;
		}

		/** Allocates the start arrays of the #CRS and #CCS structures. */
		RC allocCompressedStorage() noexcept {
			// check for trivial case
			if( m == 0 || n == 0 ) {
				// simply do not do anything and return
				return SUCCESS;
			}

			// allocate and catch errors
			char * alloc[ 2 ] = { NULL, NULL };
			size_t sizes[ 2 ];
			CRS.getStartAllocSize( &( sizes[ 0 ] ), m );
			CCS.getStartAllocSize( &( sizes[ 1 ] ), n );

			// construct descriptor of this matrix
			std::stringstream description;
			description << ", for an " << m << " times " << n << " matrix";

			// allocate
			RC ret = utils::alloc( "grb::Matrix< T, reference > (constructor)", description.str(), alloc[ 0 ], sizes[ 0 ], true, _deleter[ 0 ], alloc[ 1 ], sizes[ 1 ], true, _deleter[ 1 ] );

			if( ret != SUCCESS ) {
				// exit without side effects
				return ret;
			}

			// put allocated arrays in their intended places
			CRS.replaceStart( alloc[ 0 ] );
			CCS.replaceStart( alloc[ 1 ] );

			// done, return error code
			return SUCCESS;
		}

		/** @see grb::resize() */
		RC resize( const size_t nonzeroes ) {
			// check for trivial case
			if( m == 0 || n == 0 || nonzeroes == 0 ) {
				// simply do not do anything and return
				return SUCCESS;
			}

			// do not do anything if current capacity is sufficient
			if( nonzeroes <= cap ) {
				return SUCCESS;
			}

			if( nonzeroes >= static_cast< size_t >( std::numeric_limits< grb::config::NonzeroIndexType >::max() ) ) {
				return OVERFLW;
			}

			// allocate and catch errors
			char * alloc[ 4 ] = { NULL, NULL, NULL, NULL };
			size_t sizes[ 4 ];
			// cache old allocation data
			size_t old_sizes[ 4 ];
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
			description << ", for " << nonzeroes << " nonzeroes in an " << m << " times " << n << " matrix.\n";

			// do allocation
			RC ret = utils::alloc( "grb::Matrix< T, reference >::resize", description.str(), alloc[ 0 ], sizes[ 0 ], true, _deleter[ 2 ], alloc[ 1 ], sizes[ 1 ], true, _deleter[ 3 ], alloc[ 2 ],
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
				if( config::MEMORY::report( "grb::Matrix< T, reference "
											">::resize",
						"freed (or will eventually free)", freed, false ) ) {
					std::cout << ", for " << cap
							  << " nonzeroes this matrix previously "
								 "contained.\n";
				}
			}

			// set new capacity
			cap = nonzeroes;

			// done, return error code
			return SUCCESS;
		}

		/** @see Matrix::buildMatrixUnique */
		template< Descriptor descr = descriptors::no_operation, typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end ) {
#ifdef _DEBUG
			(void)printf( "buildMatrixUnique called with %zd nonzeroes.\n", cap );
			std::cout << "buildMatrixUnique: input is\n";
			for( auto it = _start; it != _end; ++it ) {
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

			// reset col_start array to zero, fused loop
			{
				// get minimum dimension
				size_t min_dim = static_cast< size_t >( m );
				size_t max_dim = static_cast< size_t >( n );
				if( min_dim > max_dim ) {
					std::swap( min_dim, max_dim );
				}
				// fill until minimum
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				#pragma omp parallel for schedule( \
		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
				for( size_t i = 0; i < min_dim; ++i ) {
					CRS.col_start[ i ] = 0;
					CCS.col_start[ i ] = 0;
				}
				// if the minimum dimension is the row dimension
				if( min_dim == static_cast< size_t >( m ) ) {
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					#pragma omp parallel for schedule( \
		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
					// then continue to fill column dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CCS.col_start[ i ] = 0;
					}
				} else {
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
					#pragma omp parallel for schedule( \
		dynamic, config::CACHE_LINE_SIZE::value() )
#endif
					// otherwise, continue to fill row dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CRS.col_start[ i ] = 0;
					}
				}
			}

			// perform counting sort and detect dimension mismatches
			// parallelise this loop -- internal issue #64
			for( fwd_iterator it = _start; it != _end; ++it ) {
				if( it.i() >= m ) {
					std::cerr << "Error: " << m << " times " << n
							  << " matrix nonzero ingestion encounters row "
								 "index at "
							  << it.i() << "\n";
					return MISMATCH;
				}
				if( it.j() >= n ) {
					std::cerr << "Error: " << m << " times " << n
							  << " matrix nonzero ingestion encounters column "
								 "input at "
							  << it.j() << "\n";
					return MISMATCH;
				}
				++( CRS.col_start[ it.i() ] );
				++( CCS.col_start[ it.j() ] );
				++nz;
			}

			// check if we can indeed store nz values
			if( nz >= static_cast< size_t >( std::numeric_limits< grb::config::NonzeroIndexType >::max() ) ) {
				return OVERFLW;
			}

			// put final entries
			CRS.col_start[ m ] = nz;
			CCS.col_start[ n ] = nz;

			// allocate enough space
			resize( nz );

			// parallelise below loops -- internal issue #192

			// make counting sort array cumulative
			for( size_t i = 1; i < m; ++i ) {
#ifdef _DEBUG
				(void)printf( "There are %ld nonzeroes at row %ld.\n", CRS.col_start[ i ], i );
#endif
				CRS.col_start[ i ] += CRS.col_start[ i - 1 ];
			}

			// make counting sort array cumulative
			for( size_t i = 1; i < n; ++i ) {
#ifdef _DEBUG
				(void)printf( "There are %ld nonzeroes at column %ld.\n", CCS.col_start[ i ], i );
#endif
				CCS.col_start[ i ] += CCS.col_start[ i - 1 ];
			}

			// perform counting sort
			fwd_iterator it = _start;
			for( size_t k = 0; it != _end; ++k, ++it ) {
				const size_t crs_pos = --( CRS.col_start[ it.i() ] );
				CRS.recordValue( crs_pos, false, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) is stored at CRS position " << static_cast< size_t >( crs_pos ) << ".\n"; // Disabled the following to support
				                                                                                                                                                    // pattern matrices: "Its stored
				                                                                                                                                                    // value is " << CRS.values[crs_pos]
				                                                                                                                                                    // << ", while the original input
				                                                                                                                                                    // was " << it.v() << ".\n";
#endif
				const size_t ccs_pos = --( CCS.col_start[ it.j() ] );
				CCS.recordValue( ccs_pos, true, it );
#ifdef _DEBUG
				std::cout << "Nonzero " << k << ", ( " << it.i() << ", " << it.j() << " ) is stored at CCS position " << static_cast< size_t >( ccs_pos ) << ".\n"; // Disabled the following to support
				                                                                                                                                                    // pattern matrices: ". Its stored
				                                                                                                                                                    // value is " << CCS.values[ccs_pos]
				                                                                                                                                                    // << ", while the original input
				                                                                                                                                                    // was " << it.v() << ".\n";
#endif
			}

#ifdef _DEBUG
			for( size_t i = 0; i <= m; ++i ) {
				(void)printf( "row_start[ %ld ] = %ld.\n", i, CRS.col_start[ i ] );
			}
			for( size_t i = 0; i <= n; ++i ) {
				(void)printf( "col_start[ %ld ] = %ld.\n", i, CCS.col_start[ i ] );
			}
#endif

			// done
			return SUCCESS;
		}

	public:
		/** @see Matrix::value_type */
		typedef D value_type;

		/** @see Matrix::Matrix() */
		Matrix( const size_t rows, const size_t columns ) : m( rows ), n( columns ), cap( 0 ), nz( 0 ) {
			if( rows >= static_cast< size_t >( std::numeric_limits< grb::config::RowIndexType >::max() ) ) {
				throw std::overflow_error( "Number of rows larger than "
										   "configured RowIndexType maximum!" );
			}
			if( columns >= static_cast< size_t >( std::numeric_limits< grb::config::ColIndexType >::max() ) ) {
				throw std::overflow_error( "Number of columns larger than "
										   "configured ColIndexType maximum!" );
			}
			if( m > 0 && n > 0 ) {
				coorArr[ 0 ] = new char[ internal::Coordinates< reference >::arraySize( m ) ];
				coorArr[ 1 ] = new char[ internal::Coordinates< reference >::arraySize( n ) ];
				coorBuf[ 0 ] = new char[ internal::Coordinates< reference >::bufferSize( m ) ];
				coorBuf[ 1 ] = new char[ internal::Coordinates< reference >::bufferSize( n ) ];
			} else {
				coorArr[ 0 ] = coorArr[ 1 ] = nullptr;
				coorBuf[ 0 ] = coorBuf[ 1 ] = nullptr;
			}
			size_t allocSize = m * internal::SizeOf< D >::value;
			if( allocSize > 0 ) {
				valbuf[ 0 ] = reinterpret_cast< D * >( new char[ allocSize ] );
			} else {
				valbuf[ 0 ] = NULL;
			}
			allocSize = n * internal::SizeOf< D >::value;
			if( allocSize > 0 ) {
				valbuf[ 1 ] = reinterpret_cast< D * >( new char[ allocSize ] );
			} else {
				valbuf[ 1 ] = NULL;
			}
			constexpr size_t globalBufferUnitSize = sizeof(typename config::RowIndexType) + sizeof(typename config::ColIndexType) + grb::utils::SizeOf< D >::value;
			static_assert( globalBufferUnitSize >= sizeof(typename config::NonzeroIndexType), "We hit here a configuration border case which the implementation does not handle at present. Please submit a bug report." );
			const bool hasNULL = coorArr[ 0 ] == NULL || coorArr[ 1 ] == NULL || coorBuf[ 0 ] == NULL || coorBuf[ 1 ] == NULL ||
				! internal::template ensureReferenceBufsize< char >( (std::max( m, n ) + 1) * globalBufferUnitSize );
			if( m > 0 && n > 0 && (hasNULL || ( allocCompressedStorage() != SUCCESS )) ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix construction" );
			}
		}

		/** @see Matrix::Matrix( const Matrix & ) */
		Matrix( const Matrix< D, reference > &other ) : Matrix( other.m, other.n ) {
			if( grb::resize( *this, nnz( other ) ) != SUCCESS ) {
				throw std::runtime_error( "Could not allocate memory during grb::Matrix copy-constructor" );
			}
			nz = other.nz;

			// if empty, return; otherwise copy
			if( nz == 0 ) { return; }

#ifdef _H_GRB_REFERENCE_OMP_MATRIX
			#pragma omp parallel
#endif
			{
				size_t range = CRS.copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				size_t start, end;
				config::OMP::localRange( start, end, 0, range );
#else
				const size_t start = 0;
				size_t end = range;
#endif
				CRS.copyFrom( other.CRS, nz, n, start, end );
				range = CCS.copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_MATRIX
				config::OMP::localRange( start, end, 0, range );
#else
				end = range;
#endif
				CCS.copyFrom( other.CCS, nz, m, start, end );
			}
		}

		/** @see Matrix::Matrix( Matrix&& ). */
		Matrix( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
		}

		/** * Move from temporary. */
		self_type& operator=( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
			return *this;
		}

		/** @see Matrix::~Matrix(). */
		~Matrix() {
#ifndef NDEBUG
			if( CRS.row_index == nullptr ) {
				assert( CCS.row_index == nullptr );
				assert( m == 0 || n == 0 || nz == 0 );
				assert( cap == 0 );
			}
#endif
			if( coorArr[ 0 ] != nullptr ) {
				delete [] coorArr[ 0 ];
			}
			if( coorArr[ 1 ] != nullptr ) {
				delete [] coorArr[ 1 ];
			}
			if( coorBuf[ 0 ] != nullptr ) {
				delete [] coorBuf[ 0 ];
			}
			if( coorBuf[ 1 ] != nullptr ) {
				delete [] coorBuf[ 1 ];
			}
			if( valbuf[ 0 ] != nullptr ) {
				delete [] reinterpret_cast< char * >( valbuf[ 0 ] );
			}
			if( valbuf[ 1 ] != nullptr ) {
				delete [] reinterpret_cast< char * >( valbuf[ 1 ] );
			}
		}

		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		begin( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution > IteratorType;
#ifdef _DEBUG
			std::cout << "In grb::Matrix<T,reference>::cbegin\n";
#endif
			return IteratorType( CRS, m, n, nz, false, s, P );
		}

		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		end( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
			assert( mode == PARALLEL );
			(void)mode;
			typedef typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution > IteratorType;
			return IteratorType( CRS, m, n, nz, true, s, P );
		}

		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
		cbegin( const IOMode mode = PARALLEL ) const {
			return begin< ActiveDistribution >( mode );
		}

		template< class ActiveDistribution = internal::Distribution< reference > >
		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution > cend( const IOMode mode = PARALLEL ) const {
			return end< ActiveDistribution >( mode );
		}
	};

	// template specialisation for GraphBLAS type traits
	template< typename D >
	struct is_container< Matrix< D, reference > > {
		/** A reference Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

} // namespace grb

// parse again for reference_omp backend
#ifdef _GRB_WITH_OMP
#ifndef _H_GRB_REFERENCE_OMP_MATRIX
#define _H_GRB_REFERENCE_OMP_MATRIX
#define reference reference_omp
#include "graphblas/reference/matrix.hpp"
#undef reference
#undef _H_GRB_REFERENCE_OMP_MATRIX
#endif
#endif

#endif // end ``_H_GRB_REFERENCE_MATRIX''
