
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
 * @date 20th of January, 2017
 */

#ifndef _H_GRB_BSP1D_BLAS1
#define _H_GRB_BSP1D_BLAS1

#include <graphblas/blas0.hpp>
#include <graphblas/blas1.hpp>
#include <graphblas/bsp/collectives.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>

#include "distribution.hpp"
#include "vector.hpp"

#define NO_CAST_ASSERT( x, y, z )                                                  \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a value that matches the expected type.\n"     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );

namespace grb {

	namespace internal {

		template< typename DataType, typename Coords >
		Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getLocal( Vector< DataType, BSP1D, Coords > & x ) {
			return x._local;
		}

		template< typename DataType, typename Coords >
		const Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getLocal( const Vector< DataType, BSP1D, Coords > & x ) {
			return x._local;
		}

		template< typename DataType, typename Coords >
		Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getGlobal( Vector< DataType, BSP1D, Coords > & x ) {
			return x._global;
		}

		template< typename DataType, typename Coords >
		const Vector< DataType, _GRB_BSP1D_BACKEND, Coordinates< _GRB_BSP1D_BACKEND > > & getGlobal( const Vector< DataType, BSP1D, Coords > & x ) {
			return x._global;
		}

		template< typename DataType, typename Coords >
		RC updateNnz( Vector< DataType, BSP1D, Coords > & x ) {
			x._became_dense = false;
			x._cleared = false;
			x._nnz_is_dirty = true;
			return x.updateNnz();
		}

		template< typename DataType, typename Coords >
		void setDense( Vector< DataType, BSP1D, Coords > & x ) {
			x._became_dense = x._nnz < x._n;
			x._cleared = false;
			x._nnz_is_dirty = false;
			x._nnz = x._n;
		}

	} // namespace internal

	/** \internal Requires no inter-process communication. */
	template< Descriptor descr = descriptors::no_operation, typename DataType, typename Coords, typename T >
	RC set( Vector< DataType, BSP1D, Coords > & x, const T val, const typename std::enable_if< ! grb::is_object< T >::value, void >::type * const = NULL ) noexcept {
		const size_t old_nnz = nnz( x );
		RC ret = SUCCESS;
		if( descr & descriptors::use_index ) {
			const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
			const auto p = data.P;
			const auto s = data.s;
			const auto n = grb::size( x );
			if( old_nnz < size( x ) ) {
				internal::getCoordinates( internal::getLocal( x ) ).assignAll();
			}
			ret = eWiseLambda(
				[ &x, &n, &s, &p ]( const size_t i ) {
					x[ i ] = internal::Distribution< BSP1D >::local_index_to_global( i, n, s, p );
				},
				x );
		} else {
			ret = set< descr >( internal::getLocal( x ), val );
		}
		if( ret == SUCCESS ) {
			internal::setDense( x );
		}
		return ret;
	}

	/**
	 * \internal Delegates to underlying backend iff index-to-process translation
	 * indicates ownership.
	 */
	template< Descriptor descr = descriptors::no_operation, typename DataType, typename Coords, typename T >
	RC setElement( Vector< DataType, BSP1D, Coords > & x,
		const T val,
		const size_t i,
		const typename std::enable_if< ! grb::is_object< DataType >::value && ! grb::is_object< T >::value, void >::type * const = NULL ) {
		const size_t n = size( x );
		// sanity check
		if( i >= n ) {
			return MISMATCH;
		}

		// prepare return code and get access to BSP1D data
		RC ret = SUCCESS;
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();

		// check if local
		// if( (i / x._b) % data.P != data.s ) {
		if( internal::Distribution< BSP1D >::global_index_to_process_id( i, n, data.P ) == data.s ) {
			// local, so translate index and perform requested operation
			const size_t local_index = internal::Distribution< BSP1D >::global_index_to_local( i, n, data.P );
#ifdef _DEBUG
			std::cout << data.s << ", grb::setElement translates global index " << i << " to " << local_index << "\n";
#endif
			ret = setElement< descr >( internal::getLocal( x ), val, local_index );
		}

		// I cannot predict if a sibling process will change the total nnz, so flag nnz dirty
		if( ret == SUCCESS ) {
			internal::signalLocalChange( x );
		}
		// done
		return ret;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, BSP1D, Coords > &x,
		const Vector< InputType, BSP1D, Coords > &y
	) {
		// sanity check
		if( size( y ) != size( x ) ) {
			return MISMATCH;
		}

		// all OK, try to do assignment
		const RC ret = set< descr >( internal::getLocal( x ), internal::getLocal( y ) );

		// if successful, update nonzero count
		if( ret == SUCCESS ) {
			// reset nonzero count flags
			x._nnz = y._nnz;
			x._nnz_is_dirty = y._nnz_is_dirty;
			x._became_dense = y._became_dense;
			x._global_is_dirty = y._global_is_dirty;
		}

		// done
		return ret;
	}

	/** \internal Requires sync on nonzero structure. */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename MaskType, typename InputType, typename Coords >
	RC set( Vector< OutputType, BSP1D, Coords > & x, const Vector< MaskType, BSP1D, Coords > & mask, const Vector< InputType, BSP1D, Coords > & y ) {
		// sanity check
		if( grb::size( y ) != grb::size( x ) ) {
			return MISMATCH;
		}
		if( grb::size( mask ) != grb::size( x ) ) {
			return MISMATCH;
		}

		// all OK, try to do assignment
		const RC ret = set< descr >( internal::getLocal( x ), internal::getLocal( mask ), internal::getLocal( y ) );

		if( ret == SUCCESS ) {
			internal::signalLocalChange( x );
		}

		// done
		return ret;
	}

	/** \internal Requires sync on nonzero structure. */
	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename MaskType, typename InputType, typename Coords >
	RC set( Vector< OutputType, BSP1D, Coords > & x, const Vector< MaskType, BSP1D, Coords > & mask, const InputType & y ) {
		// sanity check
		if( grb::size( mask ) != grb::size( x ) ) {
			return MISMATCH;
		}

		// all OK, try to do assignment
		const RC ret = set< descr >( internal::getLocal( x ), internal::getLocal( mask ), y );

		if( ret == SUCCESS ) {
			internal::signalLocalChange( x );
		}

		// done
		return ret;
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename InputType, typename Coords, typename IOType >
	RC foldr( const Vector< InputType, BSP1D, Coords > & x,
		IOType & beta,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< IOType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		// cache local result
		IOType local = monoid.template getIdentity< IOType >();

		// do local foldr
		RC rc = foldl< descr >( local, internal::getLocal( x ), monoid );

		// do allreduce using \a op
		if( rc == SUCCESS ) {
			rc = collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );
		}

		// accumulate end result
		if( rc == SUCCESS ) {
			rc = foldr( local, beta, monoid.getOperator() );
		}

		// done
		return rc;
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename InputType, typename MaskType, typename Coords >
	RC foldl( IOType & alpha,
		const Vector< InputType, BSP1D, Coords > & y,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< MaskType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cout << "foldl: IOType <- [InputType] with a monoid called. Array has size "
				  << size( y ) << " with " << nnz( y ) << " nonzeroes. It has a mask of size "
				  << size( mask ) << " with " << nnz( mask ) << " nonzeroes." << std::endl;
#endif
		// static sanity checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< IOType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< InputType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< MaskType, bool >::value ), "grb::foldl",
			"called with a non-bool mask vector type while no_casting descriptor "
			"was set" );

		// dynamic sanity checks
		if( size( mask ) > 0 && size( mask ) != size( y ) ) {
			return MISMATCH;
		}
		if( size( y ) == 0 ) {
			return ILLEGAL;
		}

		// do local foldr
		IOType local = monoid.template getIdentity< IOType >();
		RC rc = foldl< descr >( local, internal::getLocal( y ), internal::getLocal( mask ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become " << local << ". Entering allreduce..." << std::endl;
		;
#endif

		// do allreduce using \a op
		if( rc == SUCCESS ) {
			rc = collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );
		}

		// accumulate end result
		if( rc == SUCCESS ) {
			rc = foldl( alpha, local, monoid.getOperator() );
		}

		// done
		return SUCCESS;
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename Coords, typename InputType >
	RC foldr( const InputType & alpha,
		Vector< IOType, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< InputType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		// simply delegate to reference implementation will yield correct result
		RC ret = foldr< descr >( alpha, internal::getLocal( y ), monoid );
		if( ret == SUCCESS ) {
			internal::getCoordinates( internal::getGlobal( y ) ).assignAll();
		}
		return ret;
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class Operator, typename IOType, typename InputType, typename Coords >
	RC foldr( const Vector< InputType, BSP1D, Coords > & x,
		Vector< IOType, BSP1D, Coords > & y,
		const Operator & op,
		const typename std::enable_if< grb::is_operator< Operator >::value, void >::type * const = NULL ) {
		// simply delegating will yield the correct result
		const size_t old_nnz = nnz( internal::getLocal( y ) );
		RC ret = foldr< descr >( internal::getLocal( x ), internal::getLocal( y ), op );
		if( ret == SUCCESS && old_nnz != nnz( internal::getLocal( y ) ) ) {
			internal::signalLocalChange( y );
		}
		return ret;
	}

	/** No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename IOType, typename Coords, typename InputType >
	RC foldl( Vector< IOType, BSP1D, Coords > & x,
		const InputType & beta,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< InputType >::value && grb::is_operator< OP >::value, void >::type * const = NULL ) {
		if( nnz( x ) < size( x ) ) {
			return ILLEGAL;
		}
		return foldl< descr >( internal::getLocal( x ), beta, op );
	}

	/** No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename Coords, typename InputType >
	RC foldl( Vector< IOType, BSP1D, Coords > & x,
		const InputType & beta,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< InputType >::value && grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		RC ret = foldl< descr >( internal::getLocal( x ), beta, monoid );
		if( ret == SUCCESS ) {
			internal::signalLocalChange( x );
		}
		return ret;
	}

	/**
	 * \internal Number of nonzeroes in \a x cannot change, hence no
	 * synchronisation required.
	 */
	template< Descriptor descr = descriptors::no_operation, class OP, typename IOType, typename InputType, typename Coords >
	RC foldl( Vector< IOType, BSP1D, Coords > & x,
		const Vector< InputType, BSP1D, Coords > & y,
		const OP & op,
		const typename std::enable_if< grb::is_operator< OP >::value, void >::type * const = NULL ) {
		const size_t n = size( x );

		// runtime sanity checks
		if( n != size( y ) ) {
			return MISMATCH;
		}

		// simply delegating will yield the correct result
		const RC ret = foldl< descr >( internal::getLocal( x ), internal::getLocal( y ), op );
		return ret;
	}

	/** \internal Requires synchronisation of output vector nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename IOType, typename InputType, typename Coords >
	RC foldl( Vector< IOType, BSP1D, Coords > & x,
		const Vector< InputType, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< grb::is_monoid< Monoid >::value, void >::type * const = NULL ) {
		const size_t n = size( x );

		// runtime sanity checks
		if( n != size( y ) ) {
			return MISMATCH;
		}
		// dense check
		if( ( descr | descriptors::dense ) || ( ( nnz( x ) == n ) && ( nnz( y ) == n ) ) ) {
			return foldl( x, y, monoid.getOperator() );
		}
		// simply delegating will yield the correct result
		RC ret = foldl< descr >( internal::getLocal( x ), internal::getLocal( y ), monoid );
		if( ret == SUCCESS ) {
			internal::updateNnz( x );
		}
		return ret;
	}

	/** \internal No communication necessary. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const InputType2 beta,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		const size_t n = size( z );
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
			return ILLEGAL;
		}

		RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( x ), beta, op );
		if( ret == SUCCESS ) {
			internal::setDense( z );
			return ret;
		} else {
			return ret;
		}
	}

	/** \internal No communication necessary. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & y,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		const size_t n = size( z );
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( nnz( y ) < n ) {
			return ILLEGAL;
		}
		const RC ret = eWiseApply< descr >( internal::getLocal( z ), alpha, internal::getLocal( y ), op );
		if( ret == SUCCESS ) {
			internal::setDense( z );
			return ret;
		} else {
			return ret;
		}
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const Vector< InputType2, BSP1D, Coords > & y,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		const size_t n = size( z );
		if( size( x ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because size(x) != size(z) -- "
					  << size( x ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( size( y ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because size(y) != size(z) -- "
					  << size( y ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because x is sparse -- nnz(x) = "
					  << nnz( x ) << "\n";
#endif
			return ILLEGAL;
		}
		if( nnz( y ) < n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because y is sparse -- nnz(y) = "
					  << nnz( y ) << "\n";
#endif
			return ILLEGAL;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( x ), internal::getLocal( y ), op );
		if( ret == SUCCESS ) {
			internal::setDense( z );
		}
		return ret;
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & y,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op );
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( nnz( y ) < n ) {
#ifdef _DEBUG
			std::cerr << "\t right-hand vector is sparse but using "
						 "operator-based eWiseApply\n";
#endif
			return ILLEGAL;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), alpha, internal::getLocal( y ), op );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const Vector< InputType1, BSP1D, Coords > & x,
		const InputType2 beta,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, beta, op );
		}
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
#ifdef _DEBUG
			std::cerr << "\t left-hand vector is sparse but using "
						 "operator-based eWiseApply\n";
#endif
			return ILLEGAL;
		}
		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), internal::getLocal( x ), beta, op );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to update global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class OP, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const Vector< InputType1, BSP1D, Coords > & x,
		const Vector< InputType2, BSP1D, Coords > & y,
		const OP & op,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_operator< OP >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, y, op );
		}
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
#ifdef _DEBUG
			std::cerr << "\t left-hand vector is sparse but using "
						 "operator-based eWiseApply\n";
#endif
			return ILLEGAL;
		}
		if( nnz( y ) < n ) {
#ifdef _DEBUG
			std::cerr << "\t right-hand vector is sparse but using "
						 "operator-based eWiseApply\n";
#endif
			return ILLEGAL;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), internal::getLocal( x ), internal::getLocal( y ), op );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		const size_t n = size( z );

		// check if can delegate to dense variant
		if( ( descr & descriptors::dense ) || nnz( x ) == n ) {
			return eWiseApply< descr >( z, x, beta, monoid.getOperator() );
		}

		// run-time checks
		if( size( x ) != n ) {
			return MISMATCH;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( x ), beta, monoid );
		if( ret == SUCCESS ) {
			internal::setDense( z );
		}
		return ret;
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		const size_t n = size( z );

		// check if can delegate to dense variant
		if( ( descr & descriptors::dense ) || nnz( y ) == n ) {
			return eWiseApply< descr >( z, alpha, y, monoid.getOperator() );
		}

		// run-time checks
		if( size( y ) != n ) {
			return MISMATCH;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), alpha, internal::getLocal( y ), monoid );
		if( ret == SUCCESS ) {
			internal::setDense( z );
		}
		return ret;
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		const size_t n = size( z );

		// check if we can delegate to dense variant
		if( ( descr & descriptors::dense ) || ( nnz( x ) == n && nnz( y ) == n ) ) {
			return eWiseApply< descr >( z, x, y, monoid.getOperator() );
		}

		// run-time checks
		if( size( x ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because size(x) != size(z) -- "
					  << size( x ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( size( y ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
						 "because size(y) != size(z) -- "
					  << size( y ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( x ), internal::getLocal( y ), monoid );

		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename Coords, typename InputType1, typename InputType2 >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, alpha, y, monoid );
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}

		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), alpha, internal::getLocal( y ), monoid );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename InputType1, typename Coords, typename InputType2 >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const Vector< InputType1, BSP1D, Coords > & x,
		const InputType2 beta,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, beta, monoid );
		}
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}

		RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), internal::getLocal( x ), beta, monoid );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Monoid, typename OutputType, typename MaskType, typename InputType1, typename InputType2, typename Coords >
	RC eWiseApply( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & mask,
		const Vector< InputType1, BSP1D, Coords > & x,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Monoid & monoid,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = NULL ) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, y, monoid );
		}
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		const RC ret = eWiseApply< descr >( internal::getLocal( z ), internal::getLocal( mask ), internal::getLocal( x ), internal::getLocal( y ), monoid );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & a,
		const Vector< InputType2, BSP1D, Coords > & x,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const bool sparse = grb::nnz( a ) != n || grb::nnz( x ) != n || grb::nnz( y ) != n;
		if( ! sparse ) {
			internal::setDense( z );
			return grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ), ring );
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename Coords, typename OutputType >
	RC eWiseAdd( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 & alpha,
		const Vector< InputType2, BSP1D, Coords > & x,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseAdd< descr >( internal::getLocal( z ), alpha, internal::getLocal( x ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 & alpha,
		const Vector< InputType2, BSP1D, Coords > & x,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha, internal::getLocal( x ), internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & a,
		const InputType2 chi,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( a ), chi, internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & a,
		const Vector< InputType2, BSP1D, Coords > & x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( a ), internal::getLocal( x ), gamma, ring );
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( a ), beta, gamma, ring );
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha, internal::getLocal( x ), gamma, ring );
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha, beta, internal::getLocal( y ), ring );
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha, beta, gamma, ring );
	}

	/** \internal Requires syncing of output nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseMul( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const RC ret = eWiseMul< descr >( internal::getLocal( z ), internal::getLocal( x ), internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			ret = internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires syncing of output nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseMul( Vector< OutputType, BSP1D, Coords > & z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const RC ret = eWiseMul< descr >( internal::getLocal( z ), alpha, internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires syncing of output nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename OutputType, typename Coords >
	RC eWiseMul( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< InputType1, BSP1D, Coords > & x,
		const InputType2 beta,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		const RC ret = eWiseMul< descr >( internal::getLocal( z ), internal::getLocal( x ), beta, ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires communication to sync global nonzero count. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const Vector< InputType1, BSP1D, Coords > & a,
		const Vector< InputType2, BSP1D, Coords > & x,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, a, x, y, ring );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Does not require communication. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const InputType1 & alpha,
		const Vector< InputType2, BSP1D, Coords > & x,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), alpha, internal::getLocal( x ), internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const Vector< InputType1, BSP1D, Coords > & a,
		const InputType2 chi,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, a, chi, y, ring );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), internal::getLocal( a ), chi, internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const Vector< InputType1, BSP1D, Coords > & a,
		const Vector< InputType2, BSP1D, Coords > & x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, a, x, gamma, ring );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), internal::getLocal( a ), internal::getLocal( x ), gamma, ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const Vector< InputType1, BSP1D, Coords > & a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, a, beta, gamma, ring );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), internal::getLocal( a ), beta, gamma, ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > & x,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, alpha, x, gamma, ring );
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), alpha, internal::getLocal( x ), gamma, ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, alpha, beta, y, ring );
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), alpha, beta, internal::getLocal( y ), ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** \internal Requires synchronisation of global number of nonzeroes. */
	template< Descriptor descr = descriptors::no_operation, class Ring, typename InputType1, typename InputType2, typename InputType3, typename Coords, typename OutputType, typename MaskType >
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > & z,
		const Vector< MaskType, BSP1D, Coords > & m,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring & ring = Ring(),
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< MaskType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
				! grb::is_object< InputType3 >::value && grb::is_semiring< Ring >::value,
			void >::type * const = NULL ) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || ( grb::nnz( m ) == n && ( descr & descriptors::structural ) && ! ( descr & descriptors::invert_mask ) ) ) {
			return eWiseMulAdd< descr >( z, alpha, beta, gamma, ring );
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		const RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), internal::getLocal( m ), alpha, beta, gamma, ring );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal
	 *
	 * BSP1D implementation of the \f$ \alpha = xy \f$ operation;
	 * the dot-product.
	 *
	 * @tparam descr      The descriptor used. If left unspecified,
	 *                    grb::descriptors::no_operation is used.
	 * @tparam Ring       The semiring to be used.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[out]  z  The output element \f$ \alpha \f$.
	 * @param[in]   x  The left-hand input vector.
	 * @param[in]   y  The right-hand input vector.
	 * @param[in] ring The semiring to perform the dot-product under. If left
	 *                 undefined, the default constructor of \a Ring will be used.
	 *
	 * @return grb::MISMATCH When the dimensions of \a x and \a y do not match. All
	 *                       input data containers are left untouched if this exit
	 *                       code is returned; it will be as though this call was
	 *                       never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
	 *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
	 *         \f$ p \f$ is the number of user processes. The constant factor
	 *         depends on the cost of evaluating the addition and multiplication
	 *         operators. A good implementation uses vectorised instructions
	 *         whenever the input domains, output domain, and the operators used
	 *         allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
	 *         by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
	 *         bytes of data movement.
	 *
	 *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
	 *         between two or more user processes.
	 *
	 *      -# A call to this function does result in any system calls.
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 *
	 * The vector distributions are block-cyclic and thus conforms to the work
	 * performance guarantee.
	 *
	 * This function performs a local dot product and then calls
	 * grb::collectives::allreduce(), and thus conforms to the bandwidth and
	 * synchornisation semantics defined above.
	 */
	template<
		Descriptor descr = grb::descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot( OutputType &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const AddMonoid &addMonoid,
		const AnyOp &anyOp,
		const typename std::enable_if< !grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value,
		void >::type * const = nullptr
	) {
		// sanity check
		if( size( y ) != size( x ) ) {
			return MISMATCH;
		}

		// get field for out-of-place dot
		OutputType oop = addMonoid.template getIdentity< OutputType >();

		// all OK, try to do assignment
		RC ret = grb::dot< descr >( oop, internal::getLocal( x ), internal::getLocal( y ), addMonoid, anyOp );
		ret = ret ? ret : collectives< BSP1D >::allreduce( oop, addMonoid.getOperator() );

		// fold out-of-place dot product into existing value and exit
		ret = ret ? ret : foldl( z, oop, addMonoid.getOperator() );
		return ret;
	}

	/** \internal No implementation notes. */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseMap( const Func f, const Vector< DataType, BSP1D, Coords > & x ) {
		return eWiseMap( f, internal::getLocal( x ) );
	}

	/**
	 * \internal
	 * We can simply delegates to the reference implementation because all vectors
	 * are distributed equally in this reference implementation. Length checking is
	 * also distributed which is correct, since all calls are collective there may
	 * never be a mismatch in globally known vector sizes.
	 */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseLambda( const Func f, const Vector< DataType, BSP1D, Coords > & x ) {
		// rely on local lambda
		return eWiseLambda( f, internal::getLocal( x ) );
		// note the sparsity structure will not change by the above call
	}

	/** \internal No implementation notes. */
	template< typename Func, typename DataType1, typename DataType2, typename Coords, typename... Args >
	RC eWiseLambda( const Func f, const Vector< DataType1, BSP1D, Coords > & x, const Vector< DataType2, BSP1D, Coords > & y, Args const &... args ) {
		// check dimension mismatch
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}
		// in this implementation, the distributions are equal so no need for any synchronisation
		return eWiseLambda( f, x, args... );
		// note the sparsity structure will not change by the above call
	}

	/** \internal No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, typename T, typename U, typename Coords >
	RC zip( Vector< std::pair< T, U >, BSP1D, Coords > & z,
		const Vector< T, BSP1D, Coords > & x,
		const Vector< U, BSP1D, Coords > & y,
		const typename std::enable_if< ! grb::is_object< T >::value && ! grb::is_object< U >::value, void >::type * const = NULL ) {
		const RC ret = zip( internal::getLocal( z ), internal::getLocal( x ), internal::getLocal( y ) );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** No implementation notes. */
	template< Descriptor descr = descriptors::no_operation, typename T, typename U, typename Coords >
	RC unzip( Vector< T, BSP1D, Coords > & x,
		Vector< U, BSP1D, Coords > & y,
		const Vector< std::pair< T, U >, BSP1D, Coords > & in,
		const typename std::enable_if< ! grb::is_object< T >::value && ! grb::is_object< U >::value, void >::type * const = NULL ) {
		RC ret = unzip( internal::getLocal( x ), internal::getLocal( y ), internal::getLocal( in ) );
		if( ret == SUCCESS ) {
			ret = internal::updateNnz( x );
		}
		if( ret == SUCCESS ) {
			ret = internal::updateNnz( y );
		}
		return ret;
	}

}; // namespace grb

#undef NO_CAST_ASSERT

#endif
