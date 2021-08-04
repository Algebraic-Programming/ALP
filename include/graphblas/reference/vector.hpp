
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

#if ! defined _H_GRB_REFERENCE_VECTOR || defined _H_GRB_REFERENCE_OMP_VECTOR
#define _H_GRB_REFERENCE_VECTOR

#include <cstdlib>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <assert.h>
#include <errno.h>
#include <string.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/base/pinnedvector.hpp>
#include <graphblas/base/vector.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/config.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/distribution.hpp>
#include <graphblas/iomode.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/alloc.hpp>
#include <graphblas/utils/autodeleter.hpp>

#include "compressed_storage.hpp"
#include "coordinates.hpp"
#include "spmd.hpp"

#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | Provide a value of the same type as the first "    \
		"domain of the given accumulator.\n"                                   \
		"* Possible fix 3 | Provide a compatible accumulator where the first " \
		"domain is of the type of the given value in the template paramters "  \
		"of this call to " y ".\n"                                             \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

#define NO_MASKCAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"********************************************************************"     \
		"********************************************************************"     \
		"******************************\n"                                         \
		"*     ERROR      | " y " " z ".\n"                                        \
		"********************************************************************"     \
		"********************************************************************"     \
		"******************************\n"                                         \
		"* Possible fix 1 | Remove no_casting from the template parameters "       \
		"in this call to " y ".\n"                                                 \
		"* Possible fix 2 | Provide a vector of Booleans in this call to " y ".\n" \
		"********************************************************************"     \
		"********************************************************************"     \
		"******************************\n" );

namespace grb {

	// forward declaration of backend-local matrix specialization for vector's friends
	template< typename D >
	class Matrix< D, reference >;

	// forward-declare internal getters
	namespace internal {

		/** TODO documentation */
		template< typename D, typename C >
		inline C & getCoordinates( Vector< D, reference, C > & x ) noexcept;

		/** TODO documentation */
		template< typename D, typename C >
		inline const C & getCoordinates( const Vector< D, reference, C > & x ) noexcept;

		/** TODO documentation */
		template< typename D, typename C >
		inline D * getRaw( Vector< D, reference, C > & x ) noexcept;

		/** TODO documentation */
		template< typename D, typename C >
		inline const D * getRaw( const Vector< D, reference, C > & x ) noexcept;

		template< typename D >
		inline internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( Matrix< D, reference > & A ) noexcept;

		template< typename D >
		inline const internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( const Matrix< D, reference > & A ) noexcept;

		template< typename D >
		inline internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( Matrix< D, reference > & A ) noexcept;

		template< typename D >
		inline const internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( const Matrix< D, reference > & A ) noexcept;

	} // namespace internal

	/**
	 * The reference implementation of a GraphBLAS vector.
	 *
	 * @tparam D The type of an element of this vector. \a D shall not be a
	 *           GraphBLAS type.
	 *
	 * \warning Creating a grb::Vector of other GraphBLAS types is
	 *                <em>not allowed</em>.
	 *          Passing a GraphBLAS type as template parameter will lead to
	 *          undefined behaviour.
	 */
	template< typename D, typename MyCoordinates >
	class Vector< D, reference, MyCoordinates > {

		static_assert( ! grb::is_object< D >::value, "Cannot create a GraphBLAS vector of GraphBLAS objects!" );

		/* *********************
		     `Getter' friends
		   ********************* */

		friend MyCoordinates & internal::getCoordinates< D, MyCoordinates >( Vector< D, reference, MyCoordinates > & x ) noexcept;

		friend const MyCoordinates & internal::getCoordinates< D, MyCoordinates >( const Vector< D, reference, MyCoordinates > & x ) noexcept;

		friend D * internal::getRaw< D, MyCoordinates >( Vector< D, reference, MyCoordinates > & x ) noexcept;

		friend const D * internal::getRaw< D, MyCoordinates >( const Vector< D, reference, MyCoordinates > & x ) noexcept;

		/* *********************
		        IO friends
		   ********************* */

		template< Descriptor, typename InputType, typename fwd_iterator, typename Coords, class Dup >
		friend RC buildVector( Vector< InputType, reference, Coords > &, fwd_iterator, const fwd_iterator, const IOMode, const Dup & );

		template< Descriptor descr, typename InputType, typename fwd_iterator1, typename fwd_iterator2, typename Coords, class Dup >
		friend RC buildVector( Vector< InputType, reference, Coords > &, fwd_iterator1, const fwd_iterator1, fwd_iterator2, const fwd_iterator2, const IOMode, const Dup & );

		friend class PinnedVector< D, reference >;

		friend class PinnedVector< D, BSP1D >;

		/* *********************
		 Auxiliary backend friends
		   ********************* */
#ifdef _GRB_WITH_LPF
		friend class Vector< D, BSP1D, internal::DefaultCoordinates >;
#endif

	private:
		/** Pointer to the raw underlying array. */
		D * __restrict__ _raw;

		/** All (sparse) coordinate information. */
		MyCoordinates _coordinates;

		/**
		 * Will automatically free \a _raw, if initialised depending on how the
		 * vector was initialised and on whether the underlying data was pinned by
		 * the user.
		 */
		utils::AutoDeleter< D > _raw_deleter;

		/**
		 * Will automatically free the _assigned array in #_coordinates, depending
		 * on how the vector was initialised and on whether the underlying data was
		 * pinned by the user.
		 */
		utils::AutoDeleter< char > _assigned_deleter;

		/**
		 * Will automatically free the buffer area required by #_coordinates,
		 * depending on how the vector was initialised and on whether the
		 * underlying vector data was pinned by the user.
		 */
		utils::AutoDeleter< char > _buffer_deleter;

		/**
		 * Function to manually initialise this vector instance. This function is
		 * to be called by constructors only.
		 *
		 * @param[in] raw_in      The raw memory area this vector should wrap
		 *                        around. If \a NULL is passed, this function will
		 *                        allocate a new memory region to house \a cap_in
		 *                        vector elements. If \a NULL is passed, \a NULL
		 *                        must also be passed to \a assigned_in.
		 * @param[in] assigned_in The raw memory area this vector should wrap
		 *                        around. If \a NULL is passed, this function will
		 *                        allocate a new memory region to house a coordinate
		 *                        set of maximum size \a cap_in. If \a NULL is
		 *                        passed, \a NULL must also be passed to \a raw_in.
		 * @param[in] assigned_initialized Whether \a assigned_in was already
		 *                                 initialized, i.e., has all its bits set to
		 *                                 zero.
		 * @param[in] buffer_in   The raw memory area this instance should use as a
		 *                        buffer. If \a NULL is passed, this function will
		 *                        allocate a new memory region of appropriate size.
		 * @param[in] cap_in      The \em global size of the vector.
		 *
		 * @throws Out-of-memory When initialisation fails due to out-of-memory
		 *                       conditions.
		 * @throws Runtime error When the POSIX call to get an aligned memory area
		 *                       fails for any other reason.
		 */
		void initialize( D * const raw_in, void * const assigned_in, bool assigned_initialized, void * const buffer_in, const size_t cap_in ) {
			// set defaults
			_raw = NULL;
			_coordinates.set( NULL, false, NULL, 0 );

			// catch trivial case: zero capacity
			if( cap_in == 0 ) {
				return;
			}

			// catch trivial case: memory areas are passed explicitly
			if( raw_in != NULL || assigned_in != NULL || buffer_in != NULL ) {
				// raw_in and assigned_in must both be NULL or both be non-NULL in a call to
				// grb::Vector::initialize (reference or reference_omp).
				assert( ! ( raw_in == NULL || assigned_in == NULL || buffer_in == NULL ) );
				_raw = raw_in;
				_coordinates.set( assigned_in, assigned_initialized, buffer_in, cap_in );
				return;
			} else {
				assert( assigned_initialized == false );
			}

			// non-trivial case; we must allocate. First set defaults
			char * assigned = NULL;
			char * buffer = NULL;
			// now allocate in one go
			const RC rc = grb::utils::alloc( "grb::Vector< T, reference, "
											 "MyCoordinates > (constructor)",
				"", _raw, cap_in, true, _raw_deleter, // values array
				assigned, MyCoordinates::arraySize( cap_in ), true, _assigned_deleter, buffer, MyCoordinates::bufferSize( cap_in ), true, _buffer_deleter );

			// catch errors
			if( rc == OUTOFMEM ) {
				throw std::runtime_error( "Out-of-memory during reference "
										  "Vector memory allocation" );
			} else if( rc != SUCCESS ) {
				throw std::runtime_error( "Unhandled runtime error from Vector "
										  "memory allocation" );
			}

			// assign to _coordinates struct
			_coordinates.set( assigned, assigned_initialized, buffer, cap_in );

			// there should always be zero initial values
			assert( _coordinates.nonzeroes() == 0 );

			// done
			assert( rc == SUCCESS );
		}

		/**
		 * No implementation remarks.
		 * @see grb::buildVector for the user-level specfication.
		 */
		template< Descriptor descr = descriptors::no_operation, class Dup = typename operators::right_assign< D, D, D >, typename fwd_iterator = const D * __restrict__ >
		RC build( const Dup & dup, const fwd_iterator start, const fwd_iterator end, fwd_iterator & npos ) {
			// compile-time sanity checks
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D1, typename std::iterator_traits< fwd_iterator >::value_type >::value ), "Vector::assign",
				"called on a vector with a nonzero type that does not match the "
				"first domain of the given duplication-resolving operator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D2, D >::value ), "Vector::assign",
				"called on a vector with a nonzero type that does not match the "
				"second domain of the given duplication-resolving operator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D3, D >::value ), "Vector::assign",
				"called on a vector with a nonzero type that does not match the "
				"third domain of the given duplication-resolving operator" );

			// perform straight copy
			fwd_iterator it = start;
			for( size_t i = 0; start != end && i < _coordinates.size(); ++i ) {
				// flag coordinate as assigned
				if( _coordinates.assign( i ) ) {
					if( descr & descriptors::no_duplicates ) {
						return ILLEGAL;
					}
					// nonzero already existed, so fold into existing one
					foldl( _raw[ i ], *it++, dup );
				} else {
					// new nonzero, so overwrite
					_raw[ i ] = static_cast< D >( *it++ );
				}
			}

			// write back final position
			npos = it;

			// done
			return SUCCESS;
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		template< Descriptor descr = descriptors::no_operation, class Dup, typename ind_iterator = const size_t * __restrict__, typename nnz_iterator = const D * __restrict__ >
		RC build( const Dup & dup,
			const ind_iterator ind_start,
			const ind_iterator ind_end,
			const nnz_iterator nnz_start,
			const nnz_iterator nnz_end,
			const typename std::enable_if< grb::is_operator< Dup >::value, void >::type * const = NULL ) {
			// compile-time sanity checks
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D1, D >::value ), "Vector::build",
				"called with a duplicate operator whose left domain type does not "
				"match the vector element type" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D2, typename std::iterator_traits< nnz_iterator >::value_type >::value ), "Vector::build",
				"called with a duplicate operator whose right domain type does not "
				"match the input nonzero iterator value type" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Dup::D3, D >::value ), "Vector::build",
				"called with a duplicate operator whose output domain type does "
				"not match the vector element type" );

			// all OK, so perform copy
			nnz_iterator nnz = nnz_start;
			ind_iterator ind = ind_start;
			while( nnz != nnz_end || ind != ind_end ) {
				const size_t i = static_cast< size_t >( *ind++ );
				// sanity check
				if( i >= _coordinates.size() ) {
					return MISMATCH;
				}
				if( _coordinates.assign( i ) ) {
					if( descr & descriptors::no_duplicates ) {
						return ILLEGAL;
					}
					foldl( _raw[ i ], *nnz++, dup );
				} else {
					_raw[ i ] = static_cast< D >( *nnz++ );
				}
			}

			// done
			return SUCCESS;
		}

	public:
		/** @see Vector::value_type. */
		typedef D value_type;

		/**
		 * This implementation makes the simplest implementation choice and declares
		 * a lambda reference to be of the same type as a regular C++ reference. The
		 * restrictions as specified in Vector::lambda_reference, however, still
		 * apply.
		 *
		 * @see Vector::lambda_reference for the user-level specification.
		 */
		typedef D & lambda_reference;

		/**
		 * A standard iterator for the Vector< D, reference, MyCoordinates > class.
		 * @see Vector::const_iterator for the user-level specification.
		 */
		template< class ActiveDistribution >
		class ConstIterator : public std::iterator< std::forward_iterator_tag, std::pair< const size_t, const D >, size_t > {

			// Vector should be able to call ConstIterator's private constructor;
			// no-one else is allowed to.
			friend class Vector< D, reference, MyCoordinates >;

		private:
			/** Handle to the container to iterate on. */
			const Vector< D, reference, MyCoordinates > * container;

			/** The current iterator value. */
			std::pair< size_t, D > value;

			/** The current position in the container. */
			size_t position;

			/** The maximum value of #position. */
			size_t max;

			/** The local process ID. */
			const size_t s;

			/** The total number of processes. */
			const size_t P;

			/**
			 * The only way to construct a valid ConstIterator.
			 *
			 * \internal Can only be called from the grb::Vector< D, reference, MyCoordinates > class.
			 *
			 * @param[in] in      The container to iterate on.
			 * @param[in] initial The initial position of this iterator.
			 *
			 * If the initial position does not have an element assigned to it, the
			 * constructor will advance it to the first assigned value found in the
			 * container. If there are none, the iterator advances to the end
			 * position.
			 *
			 * The initial position must be smaller or equal to the capacity of the
			 * given container. If it is equal, this iterator will be set to its end
			 * position.
			 */
			ConstIterator( const Vector< D, reference, MyCoordinates > & in, size_t initial = 0, size_t processID = 0, size_t numProcesses = 1 ) noexcept :
				container( &in ), position( initial ), s( processID ), P( numProcesses ) {
				// make sure the initial value is valid;
				// if not, go to the next valid value:
				if( container->_coordinates.isEmpty() ) {
					max = 0;
				} else if( container->_coordinates.isDense() ) {
					max = container->_coordinates.size();
				} else {
					max = container->_coordinates.nonzeroes();
				}
				if( position < max ) {
					setValue();
				} else if( position > max ) {
					position = max;
				}
				assert( position <= max );
			}

			/**
			 * Checks whether two iterators are at the same position.
			 *
			 * @return false if the \a other iterator is derived from the same
			 *               container as this iterator, and the positions differ.
			 * @return true  if the \a other iterator is derived from the same
			 *               container as this iterator, and the positions match.
			 * \note If the \a other iterator is not derived from the same container
			 *       as this iterator, the result is undefined.
			 */
			bool equal( const ConstIterator & other ) const noexcept {
				return other.position == position;
			}

			/**
			 * Helper function that sets #value to the current #position. Should not
			 * be called if #position is out of range!
			 */
			void setValue() noexcept {
				size_t index;
				if( container->_coordinates.isDense() ) {
					index = position;
				} else {
					index = container->_coordinates.index( position );
				}
				assert( container->_coordinates.assigned( index ) );
				const size_t global_index = ActiveDistribution::local_index_to_global( index, size( *container ), s, P );
#ifdef _DEBUG
				std::cout << "\t ConstIterator at process " << s << " / " << P << " translated index " << index << " to " << global_index << "\n";
#endif
				value = std::make_pair( global_index, container->_raw[ index ] );
			}

		public:
			/** Default constructor. */
			ConstIterator() noexcept : container( NULL ), position( 0 ), max( 0 ), s( grb::spmd< reference >::pid() ), P( grb::spmd< reference >::nprocs() ) {}

			/** Copy constructor. */
			ConstIterator( const ConstIterator & other ) noexcept : container( other.container ), value( other.value ), position( other.position ), max( other.max ), s( other.s ), P( other.P ) {}

			/** Move constructor. */
			ConstIterator( ConstIterator && other ) : container( other.container ), s( other.s ), P( other.P ) {
				std::swap( value, other.value );
				std::swap( position, other.position );
				std::swap( max, other.max );
			}

			/** Copy assignment. */
			ConstIterator & operator=( const ConstIterator & other ) noexcept {
				container = other.container;
				value = other.value;
				position = other.position;
				max = other.max;
				assert( s == other.s );
				assert( P == other.P );
				return *this;
			}

			/** Move assignment. */
			ConstIterator & operator=( ConstIterator && other ) {
				container = other.container;
				std::swap( value, other.value );
				std::swap( position, other.position );
				std::swap( max, other.max );
				assert( s == other.s );
				assert( P == other.P );
				return *this;
			}

			/**
			 * @see Vector::ConstIterator::operator==
			 * @see equal().
			 */
			bool operator==( const ConstIterator & other ) const noexcept {
				return equal( other );
			}

			/**
			 * @see Vector::ConstIterator::operator!=
			 * @returns The negation of equal().
			 */
			bool operator!=( const ConstIterator & other ) const noexcept {
				return ! equal( other );
			}

			/** @see Vector::ConstIterator::operator* */
			std::pair< const size_t, const D > operator*() const noexcept {
				return value;
			}

			const std::pair< size_t, D > * operator->() const noexcept {
				return &value;
			}

			/** @see Vector::ConstIterator::operator++ */
			ConstIterator & operator++() noexcept {
				(void)++position;
				assert( position <= max );
				if( position < max ) {
					setValue();
				}
				return *this;
			}
		};

		/** Our const iterator type. */
		typedef ConstIterator< internal::Distribution< reference > > const_iterator;

		/**
		 * This constructor may throw exceptions.
		 *
		 * @see Vector for the user-level specfication.
		 */
		Vector( const size_t n ) : _raw( NULL ) {
			initialize( NULL, NULL, false, NULL, n );
		}

		/**
		 * The default constructor creates an empty vector and should never be
		 * used explicitly.
		 */
		Vector() : Vector( 0 ) {}

		/**
		 * Copy constructor.
		 *
		 * Incurs the same costs as the normal constructor, followed by a grb::set.
		 *
		 * @throws runtime_error If the call to grb::set fails, the error code is
		 *                       caught and thrown.
		 */
		Vector( const Vector< D, reference, MyCoordinates > & x ) {
			initialize( NULL, NULL, false, NULL, size( x ) );
			const auto rc = set( *this, x );
			if( rc != SUCCESS ) {
				throw std::runtime_error( "grb::set inside copy-constructor: " + toString( rc ) );
			}
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		Vector( Vector< D, reference, MyCoordinates > && x ) noexcept {
			// copy and move
			_raw = x._raw;
			_coordinates = std::move( x._coordinates );
			_raw_deleter = std::move( x._raw_deleter );
			_assigned_deleter = std::move( x._assigned_deleter );
			_buffer_deleter = std::move( x._buffer_deleter );

			// invalidate that which was not moved
			x._raw = NULL;
		}

		/** Assign-from-temporary. */
		Vector< D, reference, MyCoordinates > & operator=( Vector< D, reference, MyCoordinates > && x ) noexcept {
			_raw = x._raw;
			_coordinates = std::move( x._coordinates );
			_raw_deleter = std::move( x._raw_deleter );
			_assigned_deleter = std::move( x._assigned_deleter );
			_buffer_deleter = std::move( x._buffer_deleter );
			x._raw = NULL;
			return *this;
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		~Vector() {
			// all frees will be handled by
			// _raw_deleter,
			// _buffer_deleter, and
			// _assigned_deleter
		}

		/**
		 * This function simply translates to <code>return cbegin();</code>.
		 * @see Vector for the user-level specfication.
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		ConstIterator< ActiveDistribution > begin( const size_t s = 0, const size_t P = 1 ) const {
			return cbegin< ActiveDistribution >( s, P );
		}

		/**
		 * This function simply translates to <code>return cend();</code>.
		 * @see Vector for the user-level specfication.
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		ConstIterator< ActiveDistribution > end( const size_t s = 0, const size_t P = 1 ) const {
			return cend< ActiveDistribution >( s, P );
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		ConstIterator< ActiveDistribution > cbegin( const size_t s = 0, const size_t P = 1 ) const {
			return ConstIterator< ActiveDistribution >( *this, 0, s, P );
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		template< class ActiveDistribution = internal::Distribution< reference > >
		ConstIterator< ActiveDistribution > cend( const size_t s = 0, const size_t P = 1 ) const {
			return ConstIterator< ActiveDistribution >( *this, _coordinates.size(), s, P );
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector for the user-level specfication.
		 */
		template< Descriptor descr = descriptors::no_operation,
			typename mask_type,
			class Accum,
			typename ind_iterator = const size_t * __restrict__,
			typename nnz_iterator = const D * __restrict__,
			class Dup = operators::right_assign< D, typename nnz_iterator::value_type, D > >
		RC build( const Vector< mask_type, reference, MyCoordinates > mask,
			const Accum & accum,
			const ind_iterator ind_start,
			const ind_iterator ind_end,
			const nnz_iterator nnz_start,
			const nnz_iterator nnz_end,
			const Dup & dup = Dup() ) {
			(void)dup;

			// compile-time sanity checks
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::left_type, typename std::iterator_traits< nnz_iterator >::value_type >::value ), "Vector::build",
				"called with a value type that does not match the first domain of "
				"the given accumulator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_integral< typename std::iterator_traits< ind_iterator >::value_type >::value ), "Vector::build",
				"called with an index iterator value type that is not integral" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::right_type, D >::value ), "Vector::build",
				"called on a vector with a nonzero type that does not match the "
				"second domain of the given accumulator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::result_type, D >::value ), "Vector::build",
				"called on a vector with a nonzero type that does not match the "
				"third domain of the given accumulator" );
			NO_MASKCAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< bool, mask_type >::value ), "Vector::build",
				"called with non-boolean Vector mask while the no_casting "
				"descriptor was set" );
			static_assert( ( descr & descriptors::no_duplicates ) != 0, "This implementation does not support input of duplicate values." );

			// all OK, so perform copy
			nnz_iterator nnz = nnz_start;
			ind_iterator ind = ind_start;
			while( nnz != nnz_end || ind != ind_end ) {
				const size_t i = static_cast< size_t >( *ind++ );
				// sanity check
				if( i >= _coordinates.size() ) {
					return MISMATCH;
				}
				// only copy element when mask is true
				if( utils::interpretMask< descr >( mask._coordinates.assigned( i ), mask._raw + i ) ) {
					if( _coordinates.assign( i ) ) {
						foldl( _raw[ i ], *nnz++, accum );
					} else {
						_raw[ i ] = static_cast< D >( *nnz++ );
					}
				}
			}

			// done
			return SUCCESS;
		}

		/**
		 * No implementation remarks.
		 *
		 * @see Vector for the user-level specfication.
		 */
		template< Descriptor descr = descriptors::no_operation, class Accum = operators::right_assign< D, D, D >, typename T, typename mask_type = bool >
		RC assign( const T val, const Vector< mask_type, reference, MyCoordinates > mask, const Accum & accum = Accum() ) {
			// sanity checks
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::left_type, T >::value ), "Vector::assign (3)",
				"called with a value type that does not match the first domain of "
				"the given accumulator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::right_type, D >::value ), "Vector::assign (3)",
				"called on a vector with a nonzero type that does not match the "
				"second domain of the given accumulator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Accum::result_type, D >::value ), "Vector::assign (3)",
				"called on a vector with a nonzero type that does not match the "
				"third domain of the given accumulator" );
			NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< bool, mask_type >::value ), "Vector::assign (3)",
				"called with non-boolean Vector mask while the no_casting "
				"descriptor was set" );

			// fill capacity
			for( size_t i = 0; i < _coordinates.size(); ++i ) {
				// check mask
				if( utils::interpretMask< descr >( mask._coordinates.assigned( i ), mask._raw + i ) ) {
					// update _n if necessary
					if( _coordinates.assign( i ) ) {
						foldl( _raw[ i ], val, accum );
					} else {
						_raw[ i ] = static_cast< D >( val );
					}
				}
			}
			// return success
			return SUCCESS;
		}

		/**
		 * No implementation remarks.
		 * @see Vector for the user-level specfication.
		 */
		template< typename T >
		RC nnz( T & nnz ) const {
			nnz = _coordinates.nonzeroes();
			return SUCCESS;
		}

		/**
		 * Non-standard data accessor for debug purposes.
		 *
		 * \warning Do not use this fucntion.
		 *
		 * The user promises to never write to this data when GraphBLAS can operate
		 * on it. The user understands that data read out may be subject to incoming
		 * changes caused by preceding GraphBLAS calls.
		 *
		 * \warning This function is only defined for the reference backend--
		 *          thus switching backends may cause your code to not compile.
		 *
		 * @return A const reference to the raw data this vector contains.
		 *
		 * \note This function is used internally for testing purposes.
		 */
		D * raw() const {
			return _raw;
		}

		/**
		 * Returns a #lambda_reference to the i-th element of this vector. This
		 * reference may be modified.
		 *
		 * This implementation asserts only valid elements are requested.
		 *
		 * \note Compile with the \a NDEBUG flag set to disable checks for this
		 *       assertion.
		 *
		 * @see Vector::operator[] for the user-level specification.
		 */
		lambda_reference operator[]( const size_t i ) {
			// sanity checks
			assert( i < _coordinates.size() );
			assert( _coordinates.assigned( i ) );
			// directly return the reference
			return _raw[ i ];
		}

		/**
		 * Returns a #lambda_reference to the i-th element of this vector. This
		 * reference may \em not be modified.
		 *
		 * This implementation asserts only valid elements are requested.
		 *
		 * \note Compile with the \a NDEBUG flag set to disable checks for this
		 *       assertion.
		 *
		 * @see Vector::operator[] for the user-level specification.
		 */
		lambda_reference operator[]( const size_t i ) const {
			// sanity checks
			assert( i < _coordinates.size() );
			assert( _coordinates.assigned( i ) );
			// directly return the reference
			return _raw[ i ];
		}

		/**
		 * No implementation details.
		 *
		 * @see Vector::operator() for the user-level specification.
		 */
		/*template< class Monoid >
		    lambda_reference operator()( const size_t i, const Monoid &monoid = Monoid() ) {
		        //sanity checks
		        assert( i < _cap );
		        //check whether element exists
		        if( ! _assigned[ i ] ) {
		            //not yet, so create
		            _assigned[ i ] = true;
		            _raw[ i ] = monoid.template getIdentity< D >();
		            ++_n;
		        }
		        //and finally return the reference
		        return _raw[ i ];
		    }*/
	};

	// specialisation for GraphBLAS type_traits
	template< typename D, typename Coord >
	struct is_container< Vector< D, reference, Coord > > {
		/** A reference vector is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	// internal getters implementation
	namespace internal {

		template< typename D, typename C >
		inline C & getCoordinates( Vector< D, reference, C > & x ) noexcept {
			return x._coordinates;
		}

		template< typename D, typename C >
		inline const C & getCoordinates( const Vector< D, reference, C > & x ) noexcept {
			return x._coordinates;
		}

		template< typename D, typename C >
		inline D * getRaw( Vector< D, reference, C > & x ) noexcept {
			return x._raw;
		}

		template< typename D, typename C >
		inline const D * getRaw( const Vector< D, reference, C > & x ) noexcept {
			return x._raw;
		}

		template< typename D >
		inline internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( Matrix< D, reference > & A ) noexcept {
			return A.CRS;
		}

		template< typename D >
		inline const internal::Compressed_Storage< D, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( const Matrix< D, reference > & A ) noexcept {
			return A.CRS;
		}

		template< typename D >
		inline internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( Matrix< D, reference > & A ) noexcept {
			return A.CCS;
		}

		template< typename D >
		inline const internal::Compressed_Storage< D, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( const Matrix< D, reference > & A ) noexcept {
			return A.CCS;
		}

	} // namespace internal

} // namespace grb

#undef NO_CAST_ASSERT
#undef NO_MASKCAST_ASSERT

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
#ifndef _H_GRB_REFERENCE_OMP_VECTOR
#define _H_GRB_REFERENCE_OMP_VECTOR
#define reference reference_omp
#include "graphblas/reference/vector.hpp"
#undef reference
#undef _H_GRB_REFERENCE_OMP_VECTOR
#endif
#endif

#endif // end ``_H_GRB_REFERENCE_VECTOR''
