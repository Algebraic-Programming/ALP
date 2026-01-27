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

#include <sstream> //std::stringstream
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <iterator>
#include <cmath>

#include <assert.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/reference/compressed_storage.hpp>
#include <graphblas/reference/init.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/autodeleter.hpp>
#include <graphblas/utils/DMapper.hpp>
#include <graphblas/type_traits.hpp>

#include <graphblas/algorithms/hpcg/ndim_matrix_builders.hpp>
#include <graphblas/utils/iterators/utils.hpp>

#include <graphblas/reference/NonzeroWrapper.hpp>


namespace grb {

	namespace internal {

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		Matrix< DataType, reference, RIT, CIT, NIT >& getRefMatrix(
			Matrix< DataType, nonblocking, RIT, CIT, NIT > &A ) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		const Matrix< DataType, reference, RIT, CIT, NIT >& getRefMatrix(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A ) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getNonzeroCapacity(
			const grb::Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return A.cap;
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		const size_t & getCurrentNonzeroes(
			const grb::Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return A.nz;
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		void setCurrentNonzeroes(
			grb::Matrix< D, nonblocking, RIT, CIT, NIT > &A,
			const size_t nnz
		) noexcept {
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
		void getMatrixBuffers(
			char * &coorArr, char * &coorBuf, InputType * &valbuf,
			const unsigned int k,
			const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			assert( k < 2 );
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
		void vxm_inner_kernel_scatter(
			RC &rc,
			Vector< IOType, nonblocking, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, nonblocking, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, nonblocking, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &dst_global_to_local
		);

		template<
			Descriptor descr,
			bool masked, bool input_masked, bool left_handed,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords, typename RIT, typename CIT, typename NIT
		>
		RC vxm_generic(
			Vector< IOType, nonblocking, Coords > &u,
			const Vector< InputType3, nonblocking, Coords > &mask,
			const Vector< InputType1, nonblocking, Coords > &v,
			const Vector< InputType4, nonblocking, Coords > &v_mask,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

	} // namespace internal

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nrows(
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
	) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t ncols(
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
	) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t nnz(
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
	) noexcept;

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, nonblocking, RIT, CIT, NIT > & ) noexcept;

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	RC resize(
		Matrix< DataType, nonblocking, RIT, CIT, NIT > &,
		const size_t
	) noexcept;

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
	template<
		typename D,
		typename RowIndexType,
		typename ColIndexType,
		typename NonzeroIndexType
	>
	class Matrix< D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType > {

		static_assert( !grb::is_object< D >::value,
			"Cannot create an ALP matrix of ALP objects!" );

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend Matrix< DataType, reference, RIT, CIT, NIT > & internal::getRefMatrix(
			Matrix< DataType, nonblocking, RIT, CIT, NIT > &A
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend const Matrix< DataType, reference, RIT, CIT, NIT > &
		internal::getRefMatrix(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A
		) noexcept;


		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nrows(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t ncols(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		friend size_t nnz(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend RC clear(
			Matrix< InputType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename DataType, typename RIT, typename CIT, typename NIT  >
		friend RC resize(
			Matrix< DataType, nonblocking, RIT, CIT, NIT > &,
			const size_t
		) noexcept;

		template<
			Descriptor descr,
			bool input_dense, bool output_dense, bool masked, bool left_handed,
			template< typename > class One,
			typename IOType,
			class AdditiveMonoid, class Multiplication,
			typename InputType1, typename InputType2,
			typename InputType3,
			typename RowColType, typename NonzeroType,
			typename Coords
		>
		friend void internal::vxm_inner_kernel_scatter(
			RC &rc,
			Vector< IOType, nonblocking, Coords > &destination_vector,
			IOType * __restrict__ const &destination,
			const size_t &destination_range,
			const Vector< InputType1, nonblocking, Coords > &source_vector,
			const InputType1 * __restrict__ const &source,
			const size_t &source_index,
			const internal::Compressed_Storage<
				InputType2, RowColType, NonzeroType
			> &matrix,
			const Vector< InputType3, nonblocking, Coords > &mask_vector,
			const InputType3 * __restrict__ const &mask,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &src_local_to_global,
			const std::function< size_t( size_t ) > &dst_global_to_local
		);

		template<
			Descriptor descr,
			bool masked, bool input_masked, bool left_handed,
			template< typename > class One,
			class AdditiveMonoid, class Multiplication,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords, typename RIT, typename CIT, typename NIT
		>
		friend RC internal::vxm_generic(
			Vector< IOType, nonblocking, Coords > &u,
			const Vector< InputType3, nonblocking, Coords > &mask,
			const Vector< InputType1, nonblocking, Coords > &v,
			const Vector< InputType4, nonblocking, Coords > &v_mask,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > &A,
			const AdditiveMonoid &add,
			const Multiplication &mul,
			const std::function< size_t( size_t ) > &row_l2g,
			const std::function< size_t( size_t ) > &row_g2l,
			const std::function< size_t( size_t ) > &col_l2g,
			const std::function< size_t( size_t ) > &col_g2l
		);

		/* ********************
		        IO friends
		   ******************** */

		template<
			Descriptor descr, typename InputType,
			typename RIT, typename CIT, typename NIT,
			typename fwd_iterator
		>
		friend RC buildMatrixUnique(
			Matrix< InputType, nonblocking, RIT, CIT, NIT > &,
			fwd_iterator, const fwd_iterator,
			const IOMode
		);

		friend internal::Compressed_Storage< D, RowIndexType, NonzeroIndexType > &
		internal::getCRS<>(
			Matrix<
				D, nonblocking,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D,
			RowIndexType, NonzeroIndexType
		> & internal::getCRS<>(
			const Matrix<
				D, nonblocking,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend internal::Compressed_Storage< D, ColIndexType, NonzeroIndexType > &
		internal::getCCS<>(
			Matrix<
				D, nonblocking,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		friend const internal::Compressed_Storage<
			D, ColIndexType, NonzeroIndexType
		> & internal::getCCS<>(
			const Matrix<
				D, nonblocking,
				RowIndexType, ColIndexType, NonzeroIndexType
			> &A
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getNonzeroCapacity(
			const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend const size_t & internal::getCurrentNonzeroes(
			const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::setCurrentNonzeroes(
			grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &, const size_t
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend void internal::getMatrixBuffers(
			char *&, char *&, InputType *&,
			const unsigned int,
			const grb::Matrix< InputType, nonblocking, RIT, CIT, NIT > &
		) noexcept;

		template< typename InputType, typename RIT, typename CIT, typename NIT >
		friend uintptr_t getID(
			const Matrix< InputType, nonblocking, RIT, CIT, NIT > &
		);

		// Native interface friends

		friend const grb::Matrix<
			D, nonblocking,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, nonblocking >(
			const D *__restrict__ const,
			const ColIndexType *__restrict__ const,
			const NonzeroIndexType *__restrict__ const,
			const size_t, const size_t
		);

		friend grb::Matrix<
			D, nonblocking,
			ColIndexType, ColIndexType, NonzeroIndexType
		>
		internal::wrapCRSMatrix< D, ColIndexType, NonzeroIndexType, nonblocking >(
			D *__restrict__ const,
			ColIndexType *__restrict__ const,
			NonzeroIndexType *__restrict__ const,
			const size_t, const size_t, const size_t,
			char * const, char * const,
			D *__restrict__ const
		);


		private:

			Matrix< D, reference, RowIndexType, ColIndexType, NonzeroIndexType > ref;

			/** Our own type. */
			typedef Matrix<
				D, nonblocking,
				RowIndexType, ColIndexType, NonzeroIndexType
			> self_type;

			Matrix() : ref( )
			{}

			Matrix(
				const D *__restrict__ const _values,
				const ColIndexType *__restrict__ const _column_indices,
				const NonzeroIndexType *__restrict__ const _offset_array,
				const size_t _m, const size_t _n,
				const size_t _cap,
				char *__restrict__ const buf1 = nullptr,
				char *__restrict__ const buf2 = nullptr,
				D *__restrict__ const buf3 = nullptr
			) : ref(
				_values, _column_indices, _offset_array,
				_m, _n, _cap,
				buf1, buf2, buf3
			) {}

			void moveFromOther( self_type &&other ) {
				ref.moveFromOther( std::move( other.ref ) );
			}

			RC clear() {
				return ref.clear();
			}

			RC resize( const size_t nonzeroes ) {
				return ref.resize( nonzeroes );
			}

			template<
				Descriptor descr = descriptors::no_operation,
				typename fwd_iterator
			>
			RC buildMatrixUnique(
				const fwd_iterator &_start,
				const fwd_iterator &_end
			) {

				return ref.buildMatrixUnique( _start, _end );
			}


		public:

			/** @see Matrix::value_type */
			typedef D value_type;

			/** The iterator type over matrices of this type. */
			typedef typename internal::Compressed_Storage<
				D, RowIndexType, NonzeroIndexType
			>::template ConstIterator<
				internal::Distribution< reference >
			> const_iterator;

			Matrix(
				const size_t rows, const size_t columns, const size_t nz
			) : ref( rows, columns, nz )
			{}

			Matrix( const size_t rows, const size_t columns ) : ref( rows, columns )
			{}

			/**
			 * \internal
			 * \todo See below code comment
			 * \endinternal
			 */
			Matrix(
				const Matrix<
					D, nonblocking, RowIndexType, ColIndexType, NonzeroIndexType
				> &other ) : ref( other.ref )
			{
				//TODO: the pipeline should be executed once level-3 primitives are
				//      implemented. In the current implementation matrices may be used only
				//      as the input of SpMV
			}

			Matrix( self_type &&other ) noexcept : ref( std::move( other.ref ) ) {
				//TODO: the pipeline should be executed once level-3 primitives are
				//      implemented. In the current implementation matrices may be used only
				//      as the input of SpMV
			}

			self_type& operator=( self_type &&other ) noexcept {
				ref = std::move( other.ref );
				return *this;
			}

			~Matrix() {
				// the pipeline is executed before memory deallocation
				internal::le.execution( this );
			}

			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D, RowIndexType, NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > begin(
				const IOMode mode = PARALLEL,
				const size_t s = 0, const size_t P = 1
			) const {
				return ref.begin( mode, s, P );
			}

			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > end(
				const IOMode mode = PARALLEL,
				const size_t s = 0, const size_t P = 1
			) const {
				return ref.end( mode, s, P );
			}

			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > cbegin(
				const IOMode mode = PARALLEL
			) const {
				return ref.cbegin( mode );
			}

			template< class ActiveDistribution = internal::Distribution< reference > >
			typename internal::Compressed_Storage<
				D,
				RowIndexType,
				NonzeroIndexType
			>::template ConstIterator< ActiveDistribution > cend(
				const IOMode mode = PARALLEL
			) const {
				return ref.cend( mode );
			}

	};

	// template specialisation for GraphBLAS type traits
	template< typename D, typename RIT, typename CIT, typename NIT >
	struct is_container< Matrix< D, nonblocking, RIT, CIT, NIT > > {
		/** A nonblocking Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	//internal getters implementation
	namespace internal {

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		inline Matrix< DataType, reference, RIT, CIT, NIT >& getRefMatrix(
			Matrix< DataType, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return (A.ref);
		}

		template< typename DataType, typename RIT, typename CIT, typename NIT >
		inline const Matrix< DataType, reference, RIT, CIT, NIT >& getRefMatrix(
			const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return (A.ref);
		}

	} //end ``grb::internal'' namespace

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_MATRIX''

