
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
 * Provides the I/O primitives for the nonblocking backend.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_IO
#define _H_GRB_NONBLOCKING_IO

#include <graphblas/base/io.hpp>
#include <graphblas/vector.hpp>
#include <graphblas/matrix.hpp>

#include "lazy_evaluation.hpp"
#include "boolean_dispatcher_io.hpp"

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
		"* Possible fix 2 | Provide a value input iterator with element "      \
		"types that match the output vector element type.\n"                   \
		"* Possible fix 3 | If applicable, provide an index input iterator "   \
		"with element types that are integral.\n"                              \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );


namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

}

namespace grb {

	/**
	 * \defgroup IO Data Ingestion -- nonblocking backend
	 * @{
	 */

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	uintptr_t getID( const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A ) {
		return getID( internal::getRefMatrix( A ) );
	}

	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, nonblocking, Coords > &x ) noexcept {
		return internal::getCoordinates( x ).size();
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nrows(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A
	) noexcept {
		return nrows( internal::getRefMatrix( A ) );
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t ncols(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A
	) noexcept {
		return ncols( internal::getRefMatrix( A ) );
	}

	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, nonblocking, Coords > &x ) noexcept {
		internal::le.execution( &x );
		return internal::getCoordinates( x ).nonzeroes();
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nnz(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A
	) noexcept {
		return nnz( internal::getRefMatrix( A ) );
	}

	template< typename DataType, typename Coords >
	size_t capacity( const Vector< DataType, nonblocking, Coords > &x ) noexcept {
		return internal::getCoordinates( x ).size();
	}

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t capacity(
		const Matrix< DataType, nonblocking, RIT, CIT, NIT > &A
	) noexcept {
		return capacity( internal::getRefMatrix( A ) );
	}

	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, nonblocking, Coords > &x ) noexcept {
		internal::le.execution( &x );
		internal::getCoordinates( x ).clear();
		return SUCCESS;
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear(
		Matrix< InputType, nonblocking, RIT, CIT, NIT > &A
	) noexcept {
		return clear( internal::getRefMatrix( A ) );
	}

	template<
		typename InputType,
		typename Coords
	>
	RC resize(
		Vector< InputType, nonblocking, Coords > &x,
		const size_t new_nz
	) noexcept {
		internal::le.execution( &x );
#ifdef _DEBUG
		std::cerr << "In grb::resize (vector, nonblocking)\n";
#endif
		// this cannot wait until after the below check, as the spec defines that
		// anything is OK for an empty vector
		if( new_nz == 0 ) {
			return grb::clear( x );
		}

		// check if we have a mismatch
		if( new_nz > grb::size( x ) ) {
#ifdef _DEBUG
			std::cerr << "\t requested capacity of " << new_nz << ", "
				<< "expected a value smaller than or equal to "
				<< size( x ) << "\n";
#endif
			return ILLEGAL;
		}

		// in the nonblocking implementation, vectors are of static size
		// so this function immediately succeeds. However, all existing contents
		// must be removed
		return grb::clear( x );
	}

	template<
		typename InputType,
		typename RIT,
		typename CIT,
		typename NIT
	>
	RC resize(
		Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		const size_t new_nz
	) noexcept {
		return resize( internal::getRefMatrix( A ), new_nz );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType,
		typename T,
		typename Coords
	>
	RC set(
		Vector< DataType, nonblocking, Coords > &x,
		const T val,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< DataType, T >::value
			), "grb::set (Vector, unmasked)",
			"called with a value type that does not match that of the given vector"
		);

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		// pre-cast value to be copied
		const DataType toCopy = static_cast< DataType >( val );
		DataType * const raw = internal::getRaw( x );
		const size_t n = internal::getCoordinates( x ).size();

		constexpr const bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&x, toCopy, raw] (
			internal::Pipeline &pipeline,
			size_t lower_bound, size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage set(x, val) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				bool already_dense_output = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( x ) );
				if( !already_dense_output ) {
#endif
					Coords local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
						upper_bound );

					local_x.local_assignAllNotAlreadyAssigned();
					assert( local_x.nonzeroes() == local_x.size() );

					internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			for( size_t i = lower_bound; i < upper_bound; i++ ) {
				raw[ i ] = internal::template ValueOrIndex<
						descr, DataType, DataType
					>::getFromScalar( toCopy, i );
			}

			return SUCCESS;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::IO_SET_SCALAR,
				n, sizeof( DataType ), dense_descr, true,
				&x, nullptr,
				&internal::getCoordinates( x ), nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr, nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: SET(x, val)" << std::endl;
#endif
		return ret;
	}

	namespace internal {

		template<
			Descriptor descr,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool loop_over_vector_length,
			bool already_dense_mask,
			bool mask_is_dense,
#endif
			typename DataType,
			typename MaskType,
			typename T,
			typename Coords
		>
		RC masked_set(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool loop_over_vector_length,
			bool already_dense_mask,
			bool mask_is_dense,
#endif
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			Vector< DataType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &m,
			const T val
		) {
			// pre-cast value to be copied
			const DataType toCopy = static_cast< DataType >( val );

			DataType * const raw = internal::getRaw( x );
			const MaskType * const m_p = internal::getRaw( m );

#ifdef _DEBUG
			if( loop_over_vector_length ) {
				std::cout << "\t using loop of size n (the vector length)\n";
			} else {
				std::cout << "\t using loop of size nz (the number of nonzeroes in the "
					<< "vector)\n";
			}
#endif

			const size_t local_n = upper_bound - lower_bound;
			const size_t local_mask_nz = already_dense_mask
				? local_n
				: local_mask.nonzeroes();

			const size_t local_size_n = loop_over_vector_length
				? local_x.size()
				: local_mask_nz;

			for( size_t k = 0; k < local_size_n; ++k ) {

				const size_t index = ( ( loop_over_vector_length || already_dense_mask )
					? k
					: local_mask.index( k ) ) + lower_bound;
				assert( index < internal::getCoordinates( x ).size() );
				if( already_dense_mask ) {
					if( !internal::getCoordinates( m ).template mask< descr >( index, m_p ) ) {
						continue;
					}
				} else {
					if( !local_mask.template mask< descr >(
						index - lower_bound, m_p + lower_bound
					) ) {
						continue;
					}
				}
				if( !mask_is_dense ) {
					(void) local_x.assign( index - lower_bound );
				}
				raw[ index ] = internal::ValueOrIndex<
						descr, DataType, DataType
					>::getFromScalar( toCopy, index );
			}

			return SUCCESS;
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType,
		typename MaskType,
		typename T,
		typename Coords
	>
	RC set(
		Vector< DataType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &m,
		const T val,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value && !grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::set (vector-to-value, masked)\n";
#endif
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< DataType, T >::value ), "grb::set (Vector to scalar, masked)",
			"called with a value type that does not match that of the given "
			"vector"
		);

		// catch empty mask
		if( size( m ) == 0 ) {
			return set< descr >( x, val, phase );
		}

		// dynamic sanity checks
		const size_t sizex = size( x );
		if( sizex != size( m ) ) {
			return MISMATCH;
		}

		// handle trivial resize
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask);

		// then source is a pattern vector, just copy its pattern
		internal::Pipeline::stage_type func = [&x, &m, val] (
			internal::Pipeline &pipeline,
			size_t lower_bound, size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage set(x, m, val) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			(void) pipeline;

			Coords local_mask, local_x;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_x_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_mask = true;

			const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			// for out-of-place operations with a mask and a scalar input, whether the
			// output is dense or not depends on the mask
			if( !mask_is_dense ) {
				local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
					upper_bound );
				local_x_nz = local_x.nonzeroes();
				if( dense_descr && local_x_nz < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( m ) );
				if( !already_dense_mask ) {
#else
				already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( m ).asyncSubset( lower_bound,
						upper_bound );
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( !mask_is_dense ) {
				local_x.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( x ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( x ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( x ) );
					}
				}
			}

			const bool loop_over_vector_length = ( descr & descriptors::invert_mask ) ||
				( 4 * local_mask.nonzeroes() > 3 * local_mask.size() );

#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_masked_set<
#else
			rc = internal::masked_set<
#endif
					descr, DataType, MaskType, T, Coords
				>(
					loop_over_vector_length,
					already_dense_mask, mask_is_dense,
					lower_bound, upper_bound,
					local_x, local_mask, x, m, val
				);

			if( !mask_is_dense ) {
				internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::IO_SET_MASKED_SCALAR,
				sizex, sizeof( DataType ),
				dense_descr, dense_mask,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&m, nullptr, nullptr, nullptr,
				&internal::getCoordinates( m ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: set(x, m, val)" << std::endl;
#endif
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType,
		typename T,
		typename Coords
	>
	RC setElement(
		Vector< DataType, nonblocking, Coords > &x,
		const T val,
		const size_t i,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
				!grb::is_object< DataType >::value &&
				!grb::is_object< T >::value, void
			>::type * const = nullptr
	) {
		internal::le.execution( &x );

		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< DataType, T >::value ),
			"grb::set (Vector, at index)",
			"called with a value type that does not match that of the given "
			"vector"
		);
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// dynamic sanity checks
		if( i >= size( x ) ) {
			return MISMATCH;
		}
		if( (descr & descriptors::dense) && nnz( x ) < size( x ) ) {
			return ILLEGAL;
		}

		// do set
		(void)internal::getCoordinates( x ).assign( i );
		internal::getRaw( x )[ i ] = static_cast< DataType >( val );

#ifdef _DEBUG
		std::cout << "setElement (nonblocking) set index " << i << " to value "
			<< internal::getRaw( x )[ i ] << "\n";
#endif
		return SUCCESS;
	}

	namespace internal {

		template<
			Descriptor descr,
			bool out_is_void,
			bool in_is_void,
			bool sparse,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool already_dense_vectors,
			bool already_dense_input,
#endif
			typename OutputType,
			typename InputType,
			typename Coords
		>
		RC set_generic(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool already_dense_vectors,
			bool already_dense_input,
#endif
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< InputType, nonblocking, Coords > &y
		) {
			const size_t local_n = upper_bound - lower_bound;
			const size_t local_y_nz = already_dense_input
				? local_n
				: local_y.nonzeroes();

			OutputType * __restrict__ const dst = internal::getRaw( x );
			const InputType * __restrict__ const src = internal::getRaw( y );

			if( sparse ) {
				if( src == nullptr && dst == nullptr ) {
					for( size_t i = 0; i < local_y_nz; ++i ) {
						const size_t index = ( already_dense_input ) ? i : local_y.index( i );
						if( !already_dense_vectors ) {
							(void) local_x.assign( index );
						}
					}
				} else {
#ifndef NDEBUG
					if( src == nullptr ) {
						assert( dst == nullptr );
					}
#endif
					for( size_t i = 0; i < local_y_nz; ++i ) {
						const size_t index = ( ( already_dense_input )
							? i
							: local_y.index( i ) ) + lower_bound;
						if( !already_dense_vectors ) {
							(void) local_x.assign( index - lower_bound );
						}
						if( !out_is_void && !in_is_void ) {
							dst[ index ] = internal::setIndexOrValue< descr, OutputType >( index,
								src[ index ] );
						}
					}
				}
			} else {
				if( !( src == nullptr && dst == nullptr ) ) {
#ifndef NDEBUG
					if( src == nullptr ) {
						assert( dst == nullptr );
					}
#endif
					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						if( !out_is_void && !in_is_void ) {
							dst[ i ] = src[ i ];
						}
					}
				}
			}

			return SUCCESS;
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, nonblocking, Coords > &x,
		const Vector< InputType, nonblocking, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, InputType >::value ),
			"grb::copy (Vector)",
			"called with vector parameters whose element data types do not match"
		);
		constexpr bool out_is_void = std::is_void< OutputType >::value;
		constexpr bool in_is_void = std::is_void< OutputType >::value;
		static_assert( !in_is_void || out_is_void,
			"grb::set (nonblocking, vector <- vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( !(descr & descriptors::use_index) || !out_is_void,
			"grb::set (nonblocking, vector <- vector, masked): "
			"use_index descriptor cannot be set if output vector is void" );

		//get length
		const size_t n = internal::getCoordinates( y ).size();
		// check contract
		if( n != size( x ) ) {
			return MISMATCH;
		}
		if( n == 0 ) {
			return SUCCESS;
		}
		if( getID( x ) == getID( y ) ) {
			return ILLEGAL;
		}

		// on resize
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// on execute
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr bool dense_descr = descr & descriptors::dense;

		internal::Pipeline::stage_type func = [&x, &y] (
			internal::Pipeline &pipeline,
			size_t lower_bound, size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage set(x, y) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_x, local_y;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_y_nz = local_n;
			bool sparse = false;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			(void) pipeline;
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_input = true;

			if( !already_dense_vectors ) {
				local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
					upper_bound );

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_input = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input ) {
#else
				already_dense_input = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
					if( local_y_nz < local_n ) {
						sparse = true;
					}
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( !already_dense_vectors ) {
				if( lower_bound == 0 ) {
					internal::getCoordinates( x ).reset_global_nnz_counter();
				}
			}

			if( sparse ) {
				// this primitive is out-of-place, thus make the output empty
				if( !already_dense_vectors ) {
					local_x.local_clear();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( x ) );
#endif
				}

#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_set_generic<
#else
				rc = internal::set_generic<
#endif
						descr, out_is_void, in_is_void, true
					>(
						already_dense_vectors, already_dense_input,
						lower_bound, upper_bound,
						local_x, local_y, x, y
					);
			} else {
				if( !already_dense_vectors ) {
					local_x.local_assignAll();
				}

#ifdef GRB_BOOLEAN_DISPATCHER
				rc = internal::boolean_dispatcher_set_generic<
#else
				rc = internal::set_generic<
#endif
						descr, out_is_void, in_is_void, false
					>(
						already_dense_vectors, already_dense_input,
						lower_bound, upper_bound,
						local_x, local_y, x, y
					);
			}

			if( !already_dense_vectors ) {
				internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::IO_SET_VECTOR,
				n, sizeof( OutputType ), dense_descr, true,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&y, nullptr, nullptr, nullptr,
				&internal::getCoordinates( y ), nullptr, nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: set(x, y)" << std::endl;
#endif
		return ret;
	}

	namespace internal {

		template<
			Descriptor descr,
			bool out_is_void,
			bool in_is_void,
#ifdef GRB_BOOLEAN_DISPATCHER
			bool loop_over_y,
			bool already_dense_input_y,
			bool already_dense_mask,
			bool mask_is_dense,
#endif
			typename OutputType,
			typename MaskType,
			typename InputType,
			typename Coords
		>
		RC masked_set(
#ifndef GRB_BOOLEAN_DISPATCHER
			bool loop_over_y,
			bool already_dense_input_y,
			bool already_dense_mask,
			bool mask_is_dense,
#endif
			const size_t lower_bound,
			const size_t upper_bound,
			Coords &local_x,
			const Coords &local_mask,
			const Coords &local_y,
			Vector< OutputType, nonblocking, Coords > &x,
			const Vector< MaskType, nonblocking, Coords > &mask,
			const Vector< InputType, nonblocking, Coords > &y
		) {
			const size_t local_n = upper_bound - lower_bound;
			const size_t local_y_nz = already_dense_input_y
				? local_n
				: local_y.nonzeroes();
			const size_t local_mask_nz = already_dense_mask
				? local_n
				: local_mask.nonzeroes();

			const size_t n = loop_over_y ? local_y_nz : local_mask_nz;

			for( size_t k = 0; k < n; ++k ) {
				const size_t i = ( loop_over_y
						? ( already_dense_input_y ? k : local_y.index( k ) )
						: ( already_dense_mask ? k : local_mask.index( k ) )
					) + lower_bound;
				if( already_dense_mask ) {
					if( !internal::getCoordinates( mask ).template mask< descr >(
						i, internal::getRaw( mask )
					) ) {
						continue;
					}
				} else {
					if( !local_mask.template mask< descr >(
						i - lower_bound, internal::getRaw( mask ) + lower_bound
					) ) {
						continue;
					}
				}
				if( loop_over_y || already_dense_input_y ||
					local_y.assigned( i - lower_bound )
				) {
					if( !out_is_void && !in_is_void ) {
						if( !mask_is_dense ) {
							(void) local_x.assign( i - lower_bound );
						}
						internal::getRaw( x )[ i ] = internal::ValueOrIndex<
								descr, OutputType, InputType
							>::getFromArray(
								internal::getRaw( y ),
								[] (const size_t i) {return i;},
								i
							);
					}
				}
			}

			return SUCCESS;
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType,
		typename MaskType,
		typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, nonblocking, Coords > &x,
		const Vector< MaskType, nonblocking, Coords > &mask,
		const Vector< InputType, nonblocking, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value,
		void >::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, InputType >::value ),
			"grb::set (Vector)",
			"called with vector parameters whose element data types do not match" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ),
			"grb::set (Vector)",
			"called with non-bool mask element types" );
		constexpr bool out_is_void = std::is_void< OutputType >::value;
		constexpr bool in_is_void = std::is_void< OutputType >::value;
		static_assert( !in_is_void || out_is_void,
			"grb::set (nonblocking, vector <- vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( !(descr & descriptors::use_index) || !out_is_void,
			"grb::set (nonblocking, vector <- vector, masked): "
			"use_index descriptor cannot be set if output vector is void" );

		// catch contract violations
		const size_t size = grb::size( y );
		if( size != grb::size( x ) ) {
			return MISMATCH;
		}
		if( size == 0 ) {
			return SUCCESS;
		}
		if( getID( x ) == getID( y ) ) {
			return ILLEGAL;
		}

		// delegate if possible
		if( grb::size( mask ) == 0 ) {
			return set( x, y );
		}

		// additional contract check
		if( size != grb::size( mask ) ) {
			return MISMATCH;
		}

		// on resize
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// on execute
		assert( phase == EXECUTE );

		RC ret = SUCCESS;

		constexpr const bool dense_descr = descr & descriptors::dense;
		constexpr const bool dense_mask = dense_descr &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask);

		internal::Pipeline::stage_type func = [&x, &mask, &y] (
			internal::Pipeline &pipeline,
			size_t lower_bound, size_t upper_bound
		) {
#ifdef _NONBLOCKING_DEBUG
			#pragma omp critical
			std::cout << "\t\tExecution of stage set(x, mask, y) in the range("
				<< lower_bound << ", " << upper_bound << ")" << std::endl;
#endif
			RC rc = SUCCESS;

			Coords local_mask, local_x, local_y;
			const size_t local_n = upper_bound - lower_bound;
			size_t local_mask_nz = local_n;
			size_t local_x_nz = local_n;
			size_t local_y_nz = local_n;

#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
			const bool already_dense_vectors = dense_descr ||
				pipeline.allAlreadyDenseVectors();
#else
			constexpr const bool already_dense_vectors = dense_descr;
#endif

			bool already_dense_mask = true;
			bool already_dense_input_y = true;

			// make the vector empty unless the dense descriptor is provided
			constexpr const bool mask_is_dense = (descr & descriptors::structural) &&
				!(descr & descriptors::invert_mask) && already_dense_vectors;

			if( !mask_is_dense ) {
				local_x = internal::getCoordinates( x ).asyncSubset( lower_bound,
					upper_bound );
				local_x_nz = local_x.nonzeroes();
				if( dense_descr && local_x_nz < local_n ) {
					return ILLEGAL;
				}
			}

			if( !already_dense_vectors ) {
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				already_dense_mask = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( mask ) );
				if( !already_dense_mask ) {
#else
					already_dense_mask = false;
#endif
					local_mask = internal::getCoordinates( mask ).asyncSubset( lower_bound,
						upper_bound );
					local_mask_nz = local_mask.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}

				already_dense_input_y = pipeline.containsAlreadyDenseVector(
					&internal::getCoordinates( y ) );
				if( !already_dense_input_y ) {
#else
				already_dense_input_y = false;
#endif
					local_y = internal::getCoordinates( y ).asyncSubset( lower_bound,
						upper_bound );
					local_y_nz = local_y.nonzeroes();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
				}
#endif
			}

			if( !mask_is_dense ) {
				local_x.local_clear();
				if( lower_bound == 0 ) {
					internal::getCoordinates( x ).reset_global_nnz_counter();
#ifdef GRB_ALREADY_DENSE_OPTIMIZATION
					pipeline.markMaybeSparseVector( &internal::getCoordinates( x ) );
#endif
					if( dense_descr ) {
						pipeline.markMaybeSparseDenseDescriptorVerification(
							&internal::getCoordinates( x ) );
					}
				}
			}

			// choose optimal loop size
			const bool loop_over_y = (descr & descriptors::invert_mask) ||
				( local_y_nz < local_mask_nz );

#ifdef GRB_BOOLEAN_DISPATCHER
			rc = internal::boolean_dispatcher_masked_set<
#else
			rc = internal::masked_set<
#endif
					descr, out_is_void, in_is_void
				>(
					loop_over_y,
					already_dense_input_y, already_dense_mask, mask_is_dense,
					lower_bound, upper_bound,
					local_x, local_mask, local_y,
					x, mask, y
				);

			if( !mask_is_dense ) {
				internal::getCoordinates( x ).asyncJoinSubset( local_x, lower_bound,
					upper_bound );
			}

			return rc;
		};

		ret = ret ? ret : internal::le.addStage(
				std::move( func ),
				internal::Opcode::IO_SET_MASKED_VECTOR,
				size, sizeof( OutputType ), dense_descr, dense_mask,
				&x, nullptr, &internal::getCoordinates( x ), nullptr,
				&mask, &y, nullptr, nullptr,
				&internal::getCoordinates( mask ), &internal::getCoordinates( y ),
				nullptr, nullptr,
				nullptr
			);

#ifdef _NONBLOCKING_DEBUG
		std::cout << "\t\tStage added to a pipeline: set(x, mask, y)" << std::endl;
#endif
		return ret;
	}

	namespace internal {

		template<
			bool A_is_mask,
			Descriptor descr,
			typename OutputType,
			typename InputType1, typename InputType2 = const OutputType,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2
		>
		RC set(
			Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
			const InputType2 * __restrict__ id = nullptr
		) noexcept {
			if( internal::NONBLOCKING::warn_if_not_native &&
				config::PIPELINE::warn_if_not_native
			) {
				std::cerr << "Warning: set (matrix copy, nonblocking) currently delegates "
					<< "to a blocking implementation.\n"
					<< "         Further similar such warnings will be suppressed.\n";
				internal::NONBLOCKING::warn_if_not_native = false;
			}

			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			grb::internal::le.execution();

			// second, delegate to the reference backend
			return set< A_is_mask, descr, OutputType, InputType1, InputType2 >(
				internal::getRefMatrix( C ), internal::getRefMatrix( A ), id );
		}

	} // end namespace internal::grb

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename RIT, typename CIT, typename NIT,
		typename ValueType = DataType
	>
	RC set(
		Matrix< DataType, nonblocking, RIT, CIT, NIT  > &C,
		const ValueType& val,
		const Phase &phase = EXECUTE
	) noexcept {
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-matrix, nonblocking)" << std::endl;
#endif
		// static checks
		static_assert(
			!std::is_void< DataType >::value,
			"grb::set (unmasked set to value): cannot have a pattern "
			"matrix as output"
		);
		static_assert(
			!std::is_void< ValueType >::value,
			"grb::set (unmasked set to value): cannot have a pattern "
			"matrix as output"
		);
		NO_CAST_ASSERT( (
			( !(descr & descriptors::no_casting)
				&& std::is_convertible< ValueType, DataType >::value
			) || std::is_same< DataType, ValueType >::value
			), "grb::set (unmasked set to value): ",
			"called with non-matching value types"
		);

		// dynamic checks
		assert( phase != TRY );

		// delegate
		if( internal::NONBLOCKING::warn_if_not_native &&
			config::PIPELINE::warn_if_not_native
		) {
			std::cerr << "Warning: set (matrix, value, nonblocking) currently delegates "
				<< "to a blocking implementation.\n"
				<< "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		grb::internal::le.execution();

		// second, delegate to the reference backend
		return set< descr >( internal::getRefMatrix( C ), val, phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType, nonblocking, RIT2, CIT2, NIT2 > &A,
		const Phase &phase = EXECUTE
	) noexcept {
		static_assert( std::is_same< OutputType, void >::value ||
			!std::is_same< InputType, void >::value,
			"grb::set cannot interpret an input pattern matrix without a "
			"semiring or a monoid. This interpretation is needed for "
			"writing the non-pattern matrix output. Possible solutions: 1) "
			"use a (monoid-based) foldl / foldr, 2) use a masked set, or "
			"3) change the output of grb::set to a pattern matrix also." );
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-matrix, nonblocking)" << std::endl;
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType, OutputType >::value
			), "grb::set",
			"called with non-matching value types" );

		// dynamic checks
		assert( phase != TRY );

		// delegate
		if( phase == RESIZE ) {
			return resize( C, nnz( A ) );
		} else {
			assert( phase == EXECUTE );
			return internal::set< false, descr >( C, A );
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< OutputType, nonblocking, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, nonblocking, RIT2, CIT2, NIT2 > &A,
		const InputType2 &val,
		const Phase &phase = EXECUTE
	) noexcept {
		static_assert( !std::is_same< OutputType, void >::value,
			"internal::grb::set (masked set to value): cannot have a pattern "
			"matrix as output" );
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-value-masked, nonblocking)\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, OutputType >::value
			), "grb::set",
			"called with non-matching value types"
		);

		// dynamic checks
		assert( phase != TRY );

		// delegate
		if( phase == RESIZE ) {
			return resize( C, nnz( A ) );
		} else {
			assert( phase == EXECUTE );
			if( std::is_same< OutputType, void >::value ) {
				return internal::set< false, descr >( C, A );
			} else {
				return internal::set< true, descr >( C, A, &val );
			}
		}
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator,
		typename Coords,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, nonblocking, Coords > &x,
		fwd_iterator start,
		const fwd_iterator end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		return buildVector< descr, InputType, fwd_iterator, Coords, Dup >(
			internal::getRefVector( x ), start, end, mode, dup );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1,
		typename fwd_iterator2,
		typename Coords,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, nonblocking, Coords > &x,
		fwd_iterator1 ind_start,
		const fwd_iterator1 ind_end,
		fwd_iterator2 val_start,
		const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		internal::le.execution( &x );
		return buildVector<
				descr, InputType, fwd_iterator1, fwd_iterator2, Coords, Dup
			>(
				internal::getRefVector( x ), ind_start, ind_end, val_start, val_end,
				mode, dup
			);
	}

	/** buildMatrixUnique is based on that of the reference backend */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename RIT,
		typename CIT,
		typename NIT,
		typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		fwd_iterator start,
		const fwd_iterator end,
		const IOMode mode
	) {
		return buildMatrixUnique<
				descr, InputType, RIT, CIT, NIT, fwd_iterator
			>( internal::getRefMatrix(A), start, end, mode );
	}

	template<
		typename InputType,
		typename Coords
	>
	uintptr_t getID( const Vector< InputType, nonblocking, Coords > &x ) {
		return getID( internal::getRefVector( x ) );
	}

	template<>
	RC wait< nonblocking >();

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType,
		typename Coords,
		typename ... Args
	>
	RC wait(
		const Vector< InputType, nonblocking, Coords > &x,
		const Args &... args
	) {
		RC ret = internal::le.execution( &x );
		if( ret != SUCCESS ) {
			return ret;
		}
		return wait( args... );
	}

	template<
		typename InputType,
		typename Coords
	>
	RC wait( const Vector< InputType, nonblocking, Coords > &x ) {
		return internal::le.execution( &x );
	}

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType,
		typename RIT, typename CIT, typename NIT,
		typename... Args
	>
	RC wait(
		const Matrix< InputType, nonblocking, RIT, CIT, NIT > &A,
		const Args &... args
	) {
		(void) A;
		//TODO: currently, matrices are read only and no action is required
		//		once the level-3 primitives are implemented
		//		the pipeline should be executed like for vectors
		return wait( args... );
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC wait( const Matrix< InputType, nonblocking > &A ) {
		(void) A;
		//TODO: currently, matrices are read only and no action is required
		//		once the level-3 primitives are implemented
		//		the pipeline should be executed like for vectors
		//return wait( args... );
		return SUCCESS;
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_NONBLOCKING_IO

