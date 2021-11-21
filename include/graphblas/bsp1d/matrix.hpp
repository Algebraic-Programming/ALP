
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
 * @date 16th of February, 2017
 */

#ifndef _H_GRB_BSP1D_MATRIX
#define _H_GRB_BSP1D_MATRIX

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/reference/matrix.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils.hpp>

#include "config.hpp"
#include "init.hpp"
#include "spmd.hpp"

namespace grb {

	/**
	 * A BSP1D Matrix. Uses a 1D block-cyclic distribution for A and A-transpose.
	 */
	template< typename D >
	class Matrix< D, BSP1D > {

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, BSP1D > & ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, BSP1D > & ) noexcept;

		template< typename DataType >
		friend size_t nnz( const Matrix< DataType, BSP1D > & ) noexcept;

		template< typename InputType, typename length_type >
		friend RC resize( Matrix< InputType, BSP1D > &, const length_type );

		template< Descriptor, bool, bool, bool, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename InputType4, typename Coords >
		friend RC internal::bsp1d_mxv( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Ring & );

		template< Descriptor descr, bool, bool, bool, class Ring, typename IOType, typename InputType1, typename InputType2, typename InputType3, typename InputType4, typename Coords >
		friend RC internal::bsp1d_vxm( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > &,
			const Ring & );

		template< Descriptor descr, typename InputType, typename fwd_iterator >
		friend RC buildMatrixUnique( Matrix< InputType, BSP1D > &, fwd_iterator, const fwd_iterator, const IOMode );

		template< typename IOType >
		friend Matrix< IOType, _GRB_BSP1D_BACKEND > & internal::getLocal( Matrix< IOType, BSP1D > & ) noexcept;

		template< typename IOType >
		friend const Matrix< IOType, _GRB_BSP1D_BACKEND > & internal::getLocal( const Matrix< IOType, BSP1D > & ) noexcept;

	private:
		/** The type of the sequential matrix implementation. */
		typedef Matrix< D, _GRB_BSP1D_BACKEND > LocalMatrix;

		/** My own type. */
		typedef Matrix< D, BSP1D > self_type;

		/**
		 * The global row-wise dimension of this matrix.
		 *
		 * \internal Not declared const to allow for elegant move construction
		 */
		size_t _m;

		/**
		 * The global column-wise dimension of this matrix.
		 *
		 * \internal Not declared const to allow for elegant move construction
		 */
		size_t _n;

		/** The actual matrix storage implementation. */
		LocalMatrix _local;

		/** Internal constructor. */
		Matrix( internal::BSP1D_Data & data, const size_t rows, const size_t columns ) :
			_m( rows ), _n( columns ), _local( internal::Distribution< BSP1D >::global_length_to_local( rows, data.s, data.P ), columns ) {
			if( _m > 0 && _n > 0 ) {
				// make sure we support an all-reduce on type D
				if( data.ensureBufferSize( data.P * utils::SizeOf< D >::value ) != SUCCESS ) {
					throw std::runtime_error( "Error during resizing of global GraphBLAS buffer" );
				}
			}
		}

		/** Implements move constructor and assign-from-temporary. */
		void moveFromOther( self_type &&other ) {
			// copy fields
			_m = other._m;
			_n = other._n;
			_local = std::move( other._local );

			// invalidate other
			other._m = 0;
			other._n = 0;
		}

	public:
		/** Base constructor. */
		Matrix( const size_t rows, const size_t columns ) : Matrix( internal::grb_BSP1D.load(), rows, columns ) {}

		/** Move constructor. */
		Matrix( self_type &&other ) noexcept : _m( other._m ), _n( other._n ), _local( std::move( other._local ) ) {
			other._m = 0;
			other._n = 0;
		}

		/** Assign-from-temporary. */
		self_type& operator=( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
			return *this;
		}

		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< internal::Distribution< BSP1D > > begin(
			const IOMode mode = PARALLEL ) const {
			return _local.template begin< internal::Distribution< BSP1D > >( mode, spmd< BSP1D >::pid(), spmd< BSP1D >::nprocs() );
		}

		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< internal::Distribution< BSP1D > > end(
			const IOMode mode = PARALLEL ) const {
			return _local.template end< internal::Distribution< BSP1D > >( mode, spmd< BSP1D >::pid(), spmd< BSP1D >::nprocs() );
		}

		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< internal::Distribution< BSP1D > > cbegin(
			const IOMode mode = PARALLEL ) const {
			return begin( mode );
		}

		typename internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< internal::Distribution< BSP1D > > cend(
			const IOMode mode = PARALLEL ) const {
			return end( mode );
		}
	};

	namespace internal {
		/** Gets the process-local matrix */
		template< typename D >
		Matrix< D, _GRB_BSP1D_BACKEND > & getLocal( Matrix< D, BSP1D > & A ) noexcept {
			return A._local;
		}
		/** Const variant */
		template< typename D >
		const Matrix< D, _GRB_BSP1D_BACKEND > & getLocal( const Matrix< D, BSP1D > & A ) noexcept {
			return A._local;
		}
	} // namespace internal

	// template specialisation for GraphBLAS type_traits
	template< typename D >
	struct is_container< Matrix< D, BSP1D > > {
		/** A BSP1D Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

} // namespace grb

#endif // end `_H_GRB_BSP1D_MATRIX'

