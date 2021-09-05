
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
 * @author: A. N. Yzelman
 */

#if ! defined _H_GRB_BANSHEE_MATRIX
#define _H_GRB_BANSHEE_MATRIX

#ifdef _DEBUG
#include <cstdio>
#endif

#include <numeric> //std::accumulate

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/autodeleter.hpp>
#include <graphblas/utils/pattern.hpp> //for help with dealing with pattern matrix input
									   //#include <sstream> //std::stringstream
#include <assert.h>

#include "blas1.hpp"
#include "blas2.hpp"
#include "compressed_storage.hpp"

namespace grb {

	namespace internal {
		template< typename D >
		size_t & getNonzeroCapacity( grb::Matrix< D, banshee > & A ) noexcept {
			return A.cap;
		}

		template< typename D >
		size_t & getCurrentNonzeroes( grb::Matrix< D, banshee > & A ) noexcept {
			return A.nz;
		}

		template< typename D >
		void setCurrentNonzeroes( grb::Matrix< D, banshee > & A, const size_t nnz ) noexcept {
			A.nz = nnz;
		}

		template< Descriptor,
			bool,
			bool,
			bool,
			bool,
			template< typename >
			class One,
			typename IOType,
			class AdditiveMonoid,
			class Multiplication,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			typename RowColType,
			typename NonzeroType >
		void vxm_inner_kernel_scatter( RC & rc,
			internal::Coordinates< banshee >::Update &,
			Vector< IOType, banshee, Coords > &,
			IOType * __restrict__ const &,
			const size_t &,
			const Vector< InputType1, banshee, Coords > &,
			const InputType1 * __restrict__ const &,
			const size_t &,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > &,
			const Vector< InputType3, banshee, Coords > &,
			const InputType3 * __restrict__ const &,
			const AdditiveMonoid &,
			const Multiplication &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > & );

		template< Descriptor,
			bool,
			bool,
			bool,
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
		RC vxm_generic( Vector< IOType, banshee, Coords > &,
			const Vector< InputType3, banshee, Coords > &,
			const Vector< InputType1, banshee, Coords > &,
			const Vector< InputType4, banshee, Coords > &,
			const Matrix< InputType2, banshee > &,
			const AdditiveMonoid &,
			const Multiplication &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > & );
	} // namespace internal

	/**
	 * A GraphBLAS matrix, banshee implementation.
	 *
	 * Uses Compressed Column Storage (CCS) plus Compressed Row Storage (CRS).
	 *
	 * \warning This implementation prefers speed over memory efficiency.
	 *
	 * @tparam D  The type of a nonzero element.
	 */
	template< typename D >
	class Matrix< D, banshee > {

		static_assert( ! grb::is_object< D >::value, "Cannot create a GraphBLAS matrix of GraphBLAS objects!" );

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, banshee > & ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, banshee > & ) noexcept;

		template< typename DataType >
		friend size_t nnz( const Matrix< DataType, banshee > & ) noexcept;

		template< typename InputType >
		friend RC clear( Matrix< InputType, banshee > & ) noexcept;

		template< typename DataType >
		friend RC resize( Matrix< DataType, banshee > &, const size_t ) noexcept;

		template< typename Func, typename DataType >
		friend RC eWiseLambda( const Func, const Matrix< DataType, banshee > & );

		// template< bool, bool, bool, bool, bool, bool, Descriptor, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename RowColType, typename
		// NonzeroType >
		template< Descriptor,
			bool,
			bool,
			bool,
			bool,
			template< typename >
			class One,
			typename IOType,
			class AdditiveMonoid,
			class Multiplication,
			typename InputType1,
			typename InputType2,
			typename InputType3,
			typename Coords,
			typename RowColType,
			typename NonzeroType >
		friend void internal::vxm_inner_kernel_scatter( RC &,
			internal::Coordinates< banshee >::Update &,
			Vector< IOType, banshee, Coords > &,
			IOType * __restrict__ const &,
			const size_t &,
			const Vector< InputType1, banshee, Coords > &,
			const InputType1 * __restrict__ const &,
			const size_t &,
			const internal::Compressed_Storage< InputType2, RowColType, NonzeroType > &,
			const Vector< InputType3, banshee, Coords > &,
			const InputType3 * __restrict__ const &,
			const AdditiveMonoid &,
			const Multiplication &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > & );

		template< Descriptor,
			bool,
			bool,
			bool,
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
		friend RC vxm_generic( Vector< IOType, banshee, Coords > &,
			const Vector< InputType3, banshee, Coords > &,
			const Vector< InputType1, banshee, Coords > &,
			const Vector< InputType4, banshee, Coords > &,
			const Matrix< InputType2, banshee > &,
			const AdditiveMonoid &,
			const Multiplication &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > &,
			const std::function< size_t( size_t ) > & );

		/* ********************
		        IO friends
		   ******************** */

		template< Descriptor descr, typename InputType, typename fwd_iterator >
		friend RC buildMatrixUnique( Matrix< InputType, banshee > &, fwd_iterator, const fwd_iterator, const IOMode );

		friend internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & internal::getCRS<>( Matrix< D, banshee > & A ) noexcept;

		friend const internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & internal::getCRS<>( const Matrix< D, banshee > & A ) noexcept;

		friend internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & internal::getCCS<>( Matrix< D, banshee > & A ) noexcept;

		friend const internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & internal::getCCS<>( const Matrix< D, banshee > & A ) noexcept;

		template< typename InputType >
		friend size_t & internal::getNonzeroCapacity( grb::Matrix< InputType, banshee > & ) noexcept;

		template< typename InputType >
		friend size_t & internal::getCurrentNonzeroes( grb::Matrix< InputType, banshee > & ) noexcept;

		template< typename InputType >
		friend void internal::setCurrentNonzeroes( grb::Matrix< InputType, banshee > &, const size_t ) noexcept;

	private:
		/** Our own type. */
		typedef Matrix< D, banshee > self_type;

		/** The Row Compressed Storage */
		class internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > CRS;

		/** The Column Compressed Storage */
		class internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > CCS;

		/** The number of rows. */
		const size_t m;

		/** The number of columns. */
		const size_t n;

		/** The nonzero capacity (in elements). */
		size_t cap;

		/** The current number of nonzeroes. */
		size_t nz;

		/**
		 * Six utils::AutoDeleter objects to free matrix resources automatically
		 * once these go out of scope. We interpret each resource as a block of
		 * bytes, hence we choose \a char as datatype here. The amount of bytes
		 * is controlled by the internal::Compressed_Storage class.
		 */
		utils::AutoDeleter< char > _deleter[ 6 ];

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

			// allocate
			RC ret = utils::alloc( alloc[ 0 ], sizes[ 0 ], true, _deleter[ 0 ], alloc[ 1 ], sizes[ 1 ], true, _deleter[ 1 ] );

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

			// do allocation
			RC ret = utils::alloc(
				alloc[ 0 ], sizes[ 0 ], true, _deleter[ 2 ], alloc[ 1 ], sizes[ 1 ], true, _deleter[ 3 ], alloc[ 2 ], sizes[ 2 ], true, _deleter[ 4 ], alloc[ 3 ], sizes[ 3 ], true, _deleter[ 5 ] );

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
			printf( "buildMatrixUnique: input is\n" );
			for( auto it = _start; it != _end; ++it ) {
				printf( "\t %d, %d\n", (int)it.i(), (int)it.j() );
			}
			printf( "buildMatrixUnique: end input.\n" );
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
				for( size_t i = 0; i < min_dim; ++i ) {
					CRS.col_start[ i ] = 0;
					CCS.col_start[ i ] = 0;
				}
				// if the minimum dimension is the row dimension
				if( min_dim == static_cast< size_t >( m ) ) {
					// then continue to fill column dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CCS.col_start[ i ] = 0;
					}
				} else {
					// otherwise, continue to fill row dimension
					for( size_t i = min_dim; i < max_dim; ++i ) {
						CRS.col_start[ i ] = 0;
					}
				}
			}

			// perform counting sort and detect dimension mismatches
			for( fwd_iterator it = _start; it != _end; ++it ) {
				if( it.i() >= m ) {
					return MISMATCH;
				}
				if( it.j() >= n ) {
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

			// make counting sort array cumulative
			for( size_t i = 1; i < m; ++i ) {
#ifdef _DEBUG
				(void)printf( "There are %d nonzeroes at row %d.\n", CRS.col_start[ i ], i );
#endif
				CRS.col_start[ i ] += CRS.col_start[ i - 1 ];
			}

			// make counting sort array cumulative
			for( size_t i = 1; i < n; ++i ) {
#ifdef _DEBUG
				(void)printf( "There are %d nonzeroes at column %d.\n", CCS.col_start[ i ], i );
#endif
				CCS.col_start[ i ] += CCS.col_start[ i - 1 ];
			}

			// perform counting sort
			fwd_iterator it = _start;
			for( size_t k = 0; it != _end; ++k, ++it ) {
				const size_t crs_pos = --( CRS.col_start[ it.i() ] );
				CRS.recordValue( crs_pos, false, it );
#ifdef _DEBUG
				printf( "Nonzero %d, (%d, %d ) is stored at CRS position %d.\n", (int)k, (int)it.i(), (int)it.j(),
					(int)static_cast< size_t >( crs_pos ) ); // Disabled the following to support pattern matrices: "Its stored value is " << CRS.values[crs_pos] << ", while the original input was "
				                                             // << it.v() << ".\n";
#endif
				const size_t ccs_pos = --( CCS.col_start[ it.j() ] );
				CCS.recordValue( ccs_pos, true, it );
#ifdef _DEBUG
				printf( "Nonzero %d, (%d, %d ) is stored at CCS position %d \n", (int)k, (int)it.i(), (int)it.j(),
					(int)static_cast< size_t >( ccs_pos ) ); // Disabled the following to support pattern matrices: ". Its stored value is " << CCS.values[ccs_pos] << ", while the original input was "
				                                             // << it.v() << ".\n";
#endif
			}

#ifdef _DEBUG
			for( size_t i = 0; i <= m; ++i ) {
				(void)printf( "row_start[ %d ] = %d.\n", i, CRS.col_start[ i ] );
			}
			for( size_t i = 0; i <= n; ++i ) {
				(void)printf( "col_start[ %d ] = %d.\n", i, CCS.col_start[ i ] );
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
			//				if( rows >= static_cast< size_t >(std::numeric_limits< grb::config::RowIndexType >::max()) ) {
			//					throw std::overflow_error( "Number of rows larger than configured RowIndexType maximum!" );
			//				}
			//				if( columns >= static_cast< size_t >(std::numeric_limits< grb::config::ColIndexType >::max()) ) {
			//					throw std::overflow_error( "Number of columns larger than configured ColIndexType maximum!" );
			//				}
			allocCompressedStorage();
		}

		/** @see Matrix::Matrix( const Matrix & ) */
		Matrix( const Matrix< D, banshee > & other ) : Matrix( other.m, other.n, other.cap ) {
			// make explicit copy
			nz = other.nz;
			CRS.copyFrom( other.CRS, nz, n );
			CCS.copyFrom( other.CCS, nz, m );
		}

		/** @see Matrix::Matrix( Matrix&& ). */
		Matrix( self_type && other ) : CRS( other.CRS ), CCS( other.CCS ), m( other.m ), n( other.n ), cap( other.cap ) {
			other.m = 0;
			other.n = 0;
			other.cap = 0;
			other.nz = 0;
		}

		/** @see Matrix::~Matrix(). */
		~Matrix() {
#ifndef NDEBUG
			if( CRS.row_index == NULL ) {
				assert( CCS.row_index == NULL );
				assert( m == 0 || n == 0 || nz == 0 );
				assert( cap == 0 );
			}
#endif
		}
	};

	// template specialisation for GraphBLAS type traits
	template< typename D >
	struct is_container< Matrix< D, banshee > > {
		/** A banshee Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

} // namespace grb

#endif // end ``_H_GRB_BANSHEE_MATRIX''
