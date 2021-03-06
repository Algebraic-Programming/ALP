
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

	// forward-declare internal getters
	namespace internal {

		template< typename D >
		Matrix< D, _GRB_BSP1D_BACKEND > & getLocal(
			Matrix< D, BSP1D > &
		) noexcept;

		template< typename D >
		const Matrix< D, _GRB_BSP1D_BACKEND > & getLocal(
			const Matrix< D, BSP1D > &
		) noexcept;

	} // namespace internal

	/**
	 * A BSP1D Matrix.
	 *
	 * \internal Uses a 1D block-cyclic distribution for A and A-transpose.
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

		template< typename DataType >
		friend size_t capacity( const Matrix< DataType, BSP1D > & ) noexcept;

		template< typename InputType >
		friend RC resize( Matrix< InputType, BSP1D > &, const size_t ) noexcept;

		template<
			Descriptor, bool, bool, bool, class Ring,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords
		>
		friend RC internal::bsp1d_mxv( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Ring &, const Phase & );

		template<
			Descriptor descr, bool, bool, bool, class Ring,
			typename IOType, typename InputType1, typename InputType2,
			typename InputType3, typename InputType4,
			typename Coords
		>
		friend RC internal::bsp1d_vxm( Vector< IOType, BSP1D, Coords > &,
			const Vector< InputType3, BSP1D, Coords > &,
			const Vector< InputType1, BSP1D, Coords > &,
			const Vector< InputType4, BSP1D, Coords > &,
			const Matrix< InputType2, BSP1D > &,
			const Ring &, const Phase & );

		template< Descriptor descr, typename InputType, typename fwd_iterator >
		friend RC buildMatrixUnique(
			Matrix< InputType, BSP1D > &,
			fwd_iterator, const fwd_iterator,
			const IOMode
		);

		template< typename IOType >
		friend Matrix< IOType, _GRB_BSP1D_BACKEND > & internal::getLocal(
			Matrix< IOType, BSP1D > &
		) noexcept;

		template< typename IOType >
		friend const Matrix< IOType, _GRB_BSP1D_BACKEND > & internal::getLocal(
			const Matrix< IOType, BSP1D > &
		) noexcept;

		template< typename IOType >
		friend uintptr_t getID( const Matrix< IOType, BSP1D > & );


	private:

		/** The type of the sequential matrix implementation. */
		typedef Matrix< D, _GRB_BSP1D_BACKEND > LocalMatrix;

		/** My own type. */
		typedef Matrix< D, BSP1D > self_type;

		/** The ID of this container. */
		uintptr_t _id;

		/** A pointer used to derive a unique ID from. */
		const char * _ptr;

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

		/**
		 * The global capacity of this matrix.
		 */
		size_t _cap;

		/** The actual matrix storage implementation. */
		LocalMatrix _local;

		/** Initializes this container. */
		void initialize( const size_t rows, const size_t cols, const size_t nz ) {
#ifdef _DEBUG
			std::cerr << "\t in initialize helper function (BSP1D matrix)\n";
#endif
			auto &data = internal::grb_BSP1D.load();
			_m = rows;
			_n = cols;
			if( _m > 0 && _n > 0 ) {
				// check capacity
				if( nz / _m > _n ||
					nz / _n / _m ||
					(nz / _m == _n && (nz % _m > 0)) ||
					(nz / _n == _m && (nz % _n > 0))
				) {
#ifdef _DEBUG
					std::cerr << "\t requested capacity is too large\n";
#endif
					throw std::runtime_error( toString( ILLEGAL ) );
				}

				// make sure we support an all-reduce on type D
				if( data.ensureBufferSize(
						data.P * utils::SizeOf< D >::value
					) != SUCCESS
				) {
					throw std::runtime_error( "Error during resizing of global buffer " );
				}

				// all OK, so assign ID
				_ptr = new char[1];
				_id = data.mapper.insert(
					reinterpret_cast< uintptr_t >(_ptr)
				);

				// derive local sizes
				const size_t local_m =
					internal::Distribution< BSP1D >::global_length_to_local(
						rows, data.s, data.P
					);
				const size_t local_n = cols;
#ifdef _DEBUG
				std::cerr << "\t\t will allocate local " << local_m << " by " << local_n
					<< " matrix and request a capacity of " << nz << "\n";
#endif

				// translate global capacity request into a local one
				size_t local_nz = nz;
				if( local_m == 0 ||
					local_n == 0 ||
					nz / local_m > local_n ||
					nz / local_n / local_m ||
					(nz / local_m == local_n && (nz % local_m > 0)) ||
					(nz / local_n == local_m && (nz % local_n > 0))
				) {
					local_nz = local_m * local_n;
#ifdef _DEBUG
					std::cerr << "\t\t will request a capacity of " << local_nz
						<< " instead of " << nz << "\n";
#endif
				}

				// delegate
				_local.initialize( &_id, local_m, local_n, local_nz );

				// sync global capacity
				size_t global_cap = capacity( _local );
				if( collectives< BSP1D >::allreduce(
						global_cap,
						operators::add< size_t >()
					) != SUCCESS
				) {
					std::cerr << "Fatal error while synchronising global capacity\n";
					throw PANIC;
				}
				_cap = global_cap;
			} else {
				_local.initialize( nullptr, 0, 0, 0 );
			}

		}

		/** Implements move constructor and assign-from-temporary. */
		void moveFromOther( self_type &&other ) {
			// copy fields
			_id = other._id;
			_ptr = other._ptr;
			_m = other._m;
			_n = other._n;
			_cap = other._cap;
			_local = std::move( other._local );

			// invalidate other
			other._id = std::numeric_limits< uintptr_t >::max();
			other._ptr = nullptr;
			other._m = 0;
			other._n = 0;
			other._cap = 0;
		}


	public:

		/**
		 * Matrix constructor.
		 *
		 * \parblock
		 * \par Performance semantics
		 *
		 * This constructor inherits the performance semantics of the grb::Matrix
		 * constructor of the underlying backend.
		 * The global work, intra-process data movement, and storage requirements are
		 * inherited from the underlying backend as \a P times what is required for
		 * \f$ \lceil m / P \rceil \times n \f$ process-local matrices with capacity
		 * \f$ \min\{ k, \lceil m / P \rceil n \} \f$.
		 *
		 * It additionally
		 *   -# incurs \f$ \Omega(\log P) + \mathcal{O}(P) \f$ work,
		 *   -# incurs \f$ \Omega(\log P) + \mathcal{O}(P) \f$ intra-process data
		 *      movement,
		 *   -# incurs \f$ \Omega(\log P) + \mathcal{O}( P ) \f$ inter-process data
		 *      movement,
		 *   -# one inter-process synchronisation step, and
		 *   -# dynamic memory allocations for \f$ \Theta( P ) \f$ memory with
		 *      corresponding system calls.
		 *
		 * Here, \f$ P \f$ is the number of user processes, while \f$ m, n, k \f$
		 * correspond to \a rows, \a columns, and \a nz, respectively.
		 * \endparblock
		 */
		Matrix( const size_t rows, const size_t columns, const size_t nz ) :
			_id( std::numeric_limits< uintptr_t >::max() ), _ptr( nullptr ),
			_m( 0 ), _n( 0 ), _cap( 0 )
		{
#ifdef _DEBUG
			std::cerr << "In grb::Matrix constructor (BSP1D, with requested initial "
				<< "capacity).\n\t Matrix size: " << rows << " by " << columns << ".\n"
				<< "\t Requested capacity: " << nz << "\n";
#endif
			initialize( rows, columns, nz );
		}

		/**
		 * Matrix constructor with default capacity argument.
		 *
		 * For performance semantics, see the above main constructor.
		 *
		 * \internal Computes the default capacity and then delegates to the main
		 *           constructor.
		 */
		Matrix( const size_t rows, const size_t columns ) :
			Matrix( rows, columns, std::max( rows, columns ) )
		{
#ifdef _DEBUG
			std::cerr << "In grb::Matrix constructor (BSP1D, default initial "
				<< "capacity).\n\t Matrix size: " << rows << " by " << columns << ".\n"
				<< "\t Default capacity: " << std::max( rows, columns ) << ".\n"
				<< "\t This constructor delegated to the constructor with explicitly "
				<< "requested initial capacity.\n";
#endif
		}

		/** Copy constructor */
		Matrix( const Matrix< D, BSP1D > &other ) :
			Matrix( other._m, other._n, other._cap )
		{
			assert( nnz( other ) <= capacity( *this ) );
			if( nnz( other ) > 0 ) {
				if( set( *this, other ) != SUCCESS ) {
					throw std::runtime_error( "Could not copy matrix" );
				}
			}
		}

		/** Move constructor. */
		Matrix( self_type &&other ) noexcept :
			_id( other._id ), _ptr( other._ptr ),
			_m( other._m ), _n( other._n ), _cap( other._cap ),
			_local( std::move( other._local )
		) {
			other._id = std::numeric_limits< uintptr_t >::max();
			other._ptr = nullptr;
			other._m = 0;
			other._n = 0;
		}

		/** Destructor. */
		~Matrix() {
#ifdef _DEBUG
			std::cerr << "In ~Matrix (BSP1D):\n"
				<< "\t matrix is " << _m << " by " << _n << "\n"
				<< "\t ID is " << _id << "\n";
#endif
			if( _m > 0 && _n > 0 ) {
#ifdef _DEBUG
				std::cerr << "\t removing ID...\n";
#endif
				assert( _ptr != nullptr );
				auto &data = internal::grb_BSP1D.load();
				assert( _id != std::numeric_limits< uintptr_t >::max() );
				data.mapper.remove( _id );
				delete [] _ptr;
				_ptr = nullptr;
			}
		}

		/** Assign-from-temporary. */
		self_type& operator=( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
			return *this;
		}

		/** Copy-assign. */
		self_type& operator=( const self_type &other ) {
			self_type replace( other );
			*this = std::move( replace );
			return *this;
		}

		typename internal::Compressed_Storage<
			D,
			grb::config::RowIndexType,
			grb::config::NonzeroIndexType
		>::template ConstIterator< internal::Distribution< BSP1D > > begin(
			const IOMode mode = PARALLEL
		) const {
			return _local.template begin< internal::Distribution< BSP1D > >(
				mode,
				spmd< BSP1D >::pid(),
				spmd< BSP1D >::nprocs()
			);
		}

		typename internal::Compressed_Storage<
			D,
			grb::config::RowIndexType,
			grb::config::NonzeroIndexType
		>::template ConstIterator< internal::Distribution< BSP1D > > end(
			const IOMode mode = PARALLEL
		) const {
			return _local.template end< internal::Distribution< BSP1D > >(
				mode,
				spmd< BSP1D >::pid(),
				spmd< BSP1D >::nprocs()
			);
		}

		typename internal::Compressed_Storage<
			D,
			grb::config::RowIndexType,
			grb::config::NonzeroIndexType
		>::template ConstIterator< internal::Distribution< BSP1D > > cbegin(
			const IOMode mode = PARALLEL
		) const {
			return begin( mode );
		}

		typename internal::Compressed_Storage<
			D,
			grb::config::RowIndexType,
			grb::config::NonzeroIndexType
		>::template ConstIterator< internal::Distribution< BSP1D > > cend(
			const IOMode mode = PARALLEL
		) const {
			return end( mode );
		}
	};

	namespace internal {

		/** Gets the process-local matrix */
		template< typename D >
		Matrix< D, _GRB_BSP1D_BACKEND > & getLocal(
			Matrix< D, BSP1D > &A
		) noexcept {
			return A._local;
		}
		/** Const variant */
		template< typename D >
		const Matrix< D, _GRB_BSP1D_BACKEND > & getLocal(
			const Matrix< D, BSP1D > &A
		) noexcept {
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

