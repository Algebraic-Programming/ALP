
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
 * Provides the nonblocking vector.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_VECTOR
#define _H_GRB_NONBLOCKING_VECTOR

#include <cstdlib>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <assert.h>
#include <errno.h>
#include <string.h>

#include <graphblas/init.hpp>
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

#include <graphblas/reference/compressed_storage.hpp>

#include "coordinates.hpp"
#include "spmd.hpp"
#include "lazy_evaluation.hpp"

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

	namespace internal {

		extern LazyEvaluation le;

	}

}

namespace grb {

	// forward declaration of backend-local matrix specialization for vector's
	// friends
	template< typename D, typename RIT, typename CIT, typename NIT >
	class Matrix< D, nonblocking, RIT, CIT, NIT >;

	// forward-declare internal getters
	namespace internal {

		template< typename D, typename C >
		inline C & getCoordinates( Vector< D, nonblocking, C > &x ) noexcept;

		template< typename D, typename C >
		inline const C & getCoordinates(
			const Vector< D, nonblocking, C > &x
		) noexcept;

		template< typename D, typename C >
		inline D * getRaw( Vector< D, nonblocking, C > &x ) noexcept;

		template< typename D, typename C >
		inline const D * getRaw( const Vector< D, nonblocking, C > &x ) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline internal::Compressed_Storage< D, RIT, NIT > & getCRS(
			Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline const internal::Compressed_Storage< D, RIT, NIT > & getCRS(
			const Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline internal::Compressed_Storage< D, CIT, NIT > & getCCS(
			Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept;

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline const internal::Compressed_Storage< D, CIT, NIT > & getCCS(
			const Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept;

		template< typename D, typename C >
		inline Vector< D, reference, C >& getRefVector(
			Vector< D, nonblocking, C > &x ) noexcept;

		template< typename D, typename C >
		inline const Vector< D, reference, C >& getRefVector(
			const Vector< D, nonblocking, C > &x ) noexcept;

	} // namespace internal

	template< typename D, typename MyCoordinates >
	class Vector< D, nonblocking, MyCoordinates > {

		static_assert( !grb::is_object< D >::value, "Cannot create an ALP/GraphBLAS"
			"vector of ALP/GraphBLAS objects!" );

		/* *********************
		     `Getter' friends
		   ********************* */

		friend MyCoordinates & internal::getCoordinates< D, MyCoordinates >(
			Vector< D, nonblocking, MyCoordinates > & x ) noexcept;

		friend const MyCoordinates & internal::getCoordinates< D, MyCoordinates >(
			const Vector< D, nonblocking, MyCoordinates > & x ) noexcept;

		friend D * internal::getRaw< D, MyCoordinates >(
			Vector< D, nonblocking, MyCoordinates > & x ) noexcept;

		friend const D * internal::getRaw< D, MyCoordinates >(
			const Vector< D, nonblocking, MyCoordinates > & x ) noexcept;

		friend Vector< D, reference, MyCoordinates > & internal::getRefVector<>(
			Vector< D, nonblocking, MyCoordinates > &x ) noexcept;

		friend const Vector< D, reference, MyCoordinates > & internal::getRefVector<>(
			const Vector< D, nonblocking, MyCoordinates > &x ) noexcept;

		/* *********************
		        IO friends
		   ********************* */

		friend class PinnedVector< D, nonblocking >;


		private:
			Vector< D, reference, MyCoordinates > ref;


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

			typedef typename Vector< D, reference, MyCoordinates >::const_iterator
				const_iterator;


			Vector( const size_t n, const size_t nz ) : ref( n, nz ) {}

			Vector( const size_t n ) : Vector( n, n ) {

				// pipeline execution is not required here as this is a grb::Vector
				// declaration
#ifdef _DEBUG
				std::cerr << "In Vector< nonblocking >::Vector( size_t ) constructor\n";
#endif
			}

			Vector() : Vector( 0 ) {}

			Vector( const Vector< D, nonblocking, MyCoordinates > &x ) :
				ref( size( x.ref ), capacity( x.ref ) )
			{
				// full delegation to the copy constructor of the reference backend is
				// impossible since the pipeline must be executed before the copy
				// constructor
				// instead a parameterized constructor of the reference backend is invoked
				// to perform the necessary initialization as the initialize method is not
				// defined for the nonblocking backend
				if( internal::getCoordinates( x ).size() > 0 ) {
					internal::le.execution( &x );
				}


				// once the execution of any required pipeline is completed
				// the set primitive initializes the vector for this copy constructor
				if( size( x ) > 0 ) {
					const RC rc = set( *this, x );
					if( rc != SUCCESS ) {
						throw std::runtime_error( "grb::set inside copy-constructor: "
							+ toString( rc ) );
					}
				}
			}

			Vector( Vector< D, nonblocking, MyCoordinates > &&x ) noexcept {

				if( internal::getCoordinates( x ).size() > 0 ) {
					internal::le.execution( &x );
				}

				ref = std::move( x.ref );
			}

			Vector< D, nonblocking, MyCoordinates > & operator=(
				const Vector< D, nonblocking, MyCoordinates > &x
			) {
				const RC rc = set( *this, x );
				if( rc != grb::SUCCESS ) {
					throw std::runtime_error( grb::toString( rc ) );
				}
				return *this;
			}

			Vector< D, nonblocking, MyCoordinates > & operator=(
				Vector< D, nonblocking, MyCoordinates > &&x
			) noexcept {
				ref = std::move( x.ref );
				return *this;
			}

			~Vector() {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}
			}

			const_iterator begin(
				const size_t s = 0, const size_t P = 1
			) const {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}

				return ref.begin(s, P);
			}

			const_iterator end(
				const size_t s = 0, const size_t P = 1
			) const {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}

				return ref.end(s, P);
			}

			const_iterator cbegin(
				const size_t s = 0, const size_t P = 1
			) const {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}

				return ref.cbegin(s, P);
			}

			const_iterator cend(
				const size_t s = 0, const size_t P = 1
			) const {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}

				return ref.cend(s, P);
			}

			template< Descriptor descr = descriptors::no_operation,
				typename mask_type,
				class Accum,
				typename ind_iterator = const size_t * __restrict__,
				typename nnz_iterator = const D * __restrict__,
				class Dup = operators::right_assign<
					D, typename nnz_iterator::value_type, D
				>
			>
			RC build(
				const Vector< mask_type, nonblocking, MyCoordinates > &mask,
				const Accum &accum,
				const ind_iterator ind_start,
				const ind_iterator ind_end,
				const nnz_iterator nnz_start,
				const nnz_iterator nnz_end,
				const Dup &dup = Dup()
			) {
				return ref.build( mask.ref, accum, ind_start, ind_end, nnz_start, nnz_end,
					dup );
			}

			template<
				Descriptor descr = descriptors::no_operation,
				class Accum = operators::right_assign< D, D, D >,
				typename T, typename mask_type = bool
			>
			RC assign(
				const T &val,
				const Vector< mask_type, nonblocking, MyCoordinates > &mask,
				const Accum &accum = Accum()
			) {
				return ref.assign( val, mask.ref, accum );
			}

			template< typename T >
			RC nnz( T &nnz ) const {
				if( internal::getCoordinates( *this ).size() > 0 ) {
					internal::le.execution( this );
				}

				return ref.nnz( nnz );
			}

			D * raw() const {
				return ref.raw();
			}

			lambda_reference operator[]( const size_t i ) {
				return ref[ i ];
			}

			lambda_reference operator[]( const size_t i ) const {
				return ref[ i ];
			}

	};

	// specialisation for GraphBLAS type_traits
	template< typename D, typename Coord >
	struct is_container< Vector< D, nonblocking, Coord > > {
		/** A nonblocking vector is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	// internal getters implementation
	namespace internal {

		template< typename D, typename C >
		inline C & getCoordinates( Vector< D, nonblocking, C > &x ) noexcept {
			return internal::getCoordinates( x.ref );
		}

		template< typename D, typename C >
		inline const C & getCoordinates(
			const Vector< D, nonblocking, C > &x
		) noexcept {
			return internal::getCoordinates( x.ref );
		}

		template< typename D, typename C >
		inline D * getRaw( Vector< D, nonblocking, C > &x ) noexcept {
			return getRaw( x.ref );
		}

		template< typename D, typename C >
		inline const D * getRaw( const Vector< D, nonblocking, C > &x ) noexcept {
			return getRaw( x.ref );
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline internal::Compressed_Storage< D, RIT, NIT > & getCRS(
			Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return getCRS( A.ref );
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline const internal::Compressed_Storage< D, RIT, NIT > & getCRS(
			const Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return getCRS( A.ref );
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline internal::Compressed_Storage< D, CIT, NIT > & getCCS(
			Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return getCCS( A.ref );
		}

		template< typename D, typename RIT, typename CIT, typename NIT >
		inline const internal::Compressed_Storage< D, CIT, NIT > & getCCS(
			const Matrix< D, nonblocking, RIT, CIT, NIT > &A
		) noexcept {
			return getCCS( A.ref );
		}

		template< typename D, typename C >
		inline Vector< D, reference, C >& getRefVector(
			Vector< D, nonblocking, C > &x
		) noexcept {
			return x.ref;
		}

		template< typename D, typename C >
		inline const Vector< D, reference, C >& getRefVector(
			const Vector< D, nonblocking, C > &x
		) noexcept {
			return x.ref;
		}

	} // namespace internal

} // namespace grb

#undef NO_CAST_ASSERT
#undef NO_MASKCAST_ASSERT

#endif // end ``_H_GRB_NONBLOCKING_VECTOR''

