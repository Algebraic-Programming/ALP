
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
 * @date 5th of December 2016
 */

#if !defined _H_GRB_REFERENCE_IO || defined _H_GRB_REFERENCE_OMP_IO
#define _H_GRB_REFERENCE_IO

#include <graphblas/base/io.hpp>

#include <graphblas/vector.hpp>
#include <graphblas/matrix.hpp>

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

	/**
	 * \defgroup IO Data Ingestion -- reference backend
	 * @{
	 */

	/**
	 * \internal
	 *
	 * Uses pointers to internal buffer areas that are guaranteed to exist
	 * (except for empty matrices). The buffer areas reside in the internal
	 * compressed_storage class.
	 *
	 * \endinternal
	 */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	uintptr_t getID( const Matrix< InputType, reference, RIT, CIT, NIT > &A ) {
		assert( nrows(A) > 0 );
		assert( ncols(A) > 0 );
		return A.id;
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, reference, Coords > &x ) noexcept {
		return internal::getCoordinates( x ).size();
	}

	/** \internal No implementation notes. */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nrows(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A
	) noexcept {
		return A.m;
	}

	/** \internal No implementation notes. */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t ncols(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A
	) noexcept {
		return A.n;
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, reference, Coords > &x ) noexcept {
		return internal::getCoordinates( x ).nonzeroes();
	}

	/** \internal No implementation notes. */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nnz(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A
	) noexcept {
		return A.nz;
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename Coords >
	size_t capacity( const Vector< DataType, reference, Coords > &x ) noexcept {
		return internal::getCoordinates( x ).size();
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t capacity(
		const Matrix< DataType, reference, RIT, CIT, NIT > &A
	) noexcept {
		return internal::getNonzeroCapacity( A );
	}

	/**
	 * Clears the vector of all nonzeroes.
	 *
	 * \parblock
	 * \par Performance semantics
	 *  This primitive
	 *    -# contains \f$ \Theta( k ) \f$ work,
	 *    -# moves \f$ \Theta( k ) \f$ data within this user process,
	 *    -# leaves memory usage related to \a x untouched,
	 *    -# will not allocate nor free dynamic memory, nor will make any
	 *       other system calls.
	 * Here, \f$ k \f$ is equal to #grb::nnz( x ).
	 *
	 * Note that this is a single user process backend, and hence trivially no
	 * inter-process costs will occur.
	 *
	 * In the case of the #grb::reference_omp backend, the critical path length
	 * is \f$ \mathcal{O}( k / T + T ) \f$, where \f$ T \f$ is the number of OpenMP
	 * threads.
	 * \endparblock
	 */
	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, reference, Coords > &x ) noexcept {
		internal::getCoordinates( x ).clear();
		return SUCCESS;
	}

	/**
	 * Clears the matrix of all nonzeroes.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * This function
	 *   -# consitutes \f$ \Theta(m+n) \f$ work,
	 *   -# moves up to \f$ \Theta(m+n) \f$ bytes of memory within this user
	 *      process,
	 *   -# leaves memory usage related to \a A untouched,
	 *   -# will not allocate nor free dynamic memory, nor will make any
	 *       other system calls.
	 * Here, \f$ m \f$ and \f$ n \f$ are equal to #grb::nrows( A ) and
	 * #grb::ncols( A ), respectively.
	 *
	 * Note that this is a single user process backend, and hence trivially no
	 * inter-process costs will occur.
	 *
	 * In the case of the #grb::reference_omp backend, the critical path length
	 * is \f$ \mathcal{O}( (m+n) / T + T ) \f$, where \f$ T \f$ is the number of
	 * OpenMP threads.
	 * \endparblock
	 */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, reference, RIT, CIT, NIT > &A ) noexcept {
		// delegate
		return A.clear();
	}

	/**
	 * Resizes the capacity of a given vector. Any current elements in the vector
	 * are \em not retained.
	 *
	 * \parblock
	 * \par Performance semantics
	 *  This primitive
	 *    -# contains \f$ \mathcal{O}(n) \f$ work,
	 *    -# moves \f$ \mathcal{O}(n) \f$ data within this user process,
	 *    -# leaves memory usage related to \a x untouched,
	 *    -# will not allocate nor free dynamic memory, nor will make any
	 *       other system calls.
	 * Here, \f$ n \f$ is equal to #grb::size( x ).
	 *
	 * Note that this is a single user process backend, and hence trivially no
	 * inter-process costs will occur.
	 *
	 * In the case of the #grb::reference_omp backend, the critical path length
	 * is \f$ \mathcal{O}( n / T + T ) \f$, where \f$ T \f$ is the number of
	 * OpenMP threads.
	 * \endparblock
	 *
	 * \internal No implementation notes.
	 */
	template< typename InputType, typename Coords >
	RC resize(
		Vector< InputType, reference, Coords > &x,
		const size_t new_nz
	) noexcept {
#ifdef _DEBUG
		std::cerr << "In grb::resize (vector, reference)\n";
#endif
		// this cannot wait until after the below check, as the spec defines that
		// anything is OK for an empty vector
		if( new_nz == 0 ) { return grb::clear( x ); }

		// check if we have a mismatch
		if( new_nz > grb::size( x ) ) {
#ifdef _DEBUG
			std::cerr << "\t requested capacity of " << new_nz << ", "
				<< "expected a value smaller than or equal to "
				<< size( x ) << "\n";
#endif
			return ILLEGAL;
		}

		// in the reference implementation, vectors are of static size
		// so this function immediately succeeds. However, all existing contents
		// must be removed
		return grb::clear( x );
	}

	/**
	 * Resizes the nonzero capacity of this matrix. Any current contents of the
	 * matrix are \em not retained.
	 *
	 * \parblock
	 * \par Performance semantics
	 * This function
	 *   -# consitutes \f$ \mathcal{O}( \mathit{nz} ) \f$ work,
	 *   -# moves \f$ \mathcal{O}( \mathit{nz} ) \f$ of data within the current
	 *      user process,
	 *   -# the memory storage requirements, if \a new_nz is higher than
	 *      #grb::capacity( A ) and the call to this function is successful, will
	 *      be increased to \f$ \Theta( \mathit{nz} + m + n + 2 ) \f$.
	 *   -# allocates \f$ \mathcal{O}( \mathit{nz} ) \f$ bytes of dynamic
	 *      memory, and in so doing, may make system calls.
	 * Here, \f$ \mathit{nz} \f$ is \a new_nz, \f$ m \f$ equals #grb::nrows( A ),
	 * and \f$ n \f$ equals #grb::ncols( A ). This costing also assumes allocation
	 * proceeds in \f$ \mathcal{O}( \mathit{nz} ) \f$ time, although in reality
	 * it is likely non-deterministic.
	 *
	 * Note that this is a single user process backend, and hence trivially no
	 * inter-process costs will occur.
	 *
	 * In the case of the #grb::reference_omp backend, the critical path length
	 * is \f$ \mathcal{O}( \mathit{nz} / T + T ) \f$, where \f$ T \f$ is the
	 * number of OpenMP threads.
	 * \endparblock
	 *
	 * \internal No implementation notes.
	 */
	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC resize(
		Matrix< InputType, reference, RIT, CIT, NIT > &A,
		const size_t new_nz
	) noexcept {
#ifdef _DEBUG
		std::cerr << "In grb::resize (matrix, reference)\n"
			<< "\t matrix is " << nrows(A) << " by " << ncols(A) << "\n"
			<< "\t requested capacity is " << new_nz << "\n";
#endif
		RC ret = clear( A );
		if( ret != SUCCESS ) { return ret; }

		const size_t m = nrows( A );
		const size_t n = ncols( A );
		// catch trivial case
		if( m == 0 || n == 0 ) {
			return SUCCESS;
		}
		// catch illegal input
		if( new_nz / m > n ||
			new_nz / n > m ||
			(new_nz / m == n && (new_nz % m > 0)) ||
			(new_nz / n == m && (new_nz % n > 0))
		) {
#ifdef _DEBUG
			std::cerr << "\t requesting higher capacity than could be stored in a "
				<< "matrix of the current size\n";
#endif
			return ILLEGAL;
		}

		// delegate
		ret = A.resize( new_nz );

		// done
		return ret;
	}

	namespace internal {

#ifndef _H_GRB_REFERENCE_OMP_IO
		template< Descriptor descr,
			typename OutputType, typename IndexType, typename ValueType
		>
		OutputType setIndexOrValue( const IndexType &index, const ValueType &value,
			const typename std::enable_if<
				std::is_convertible< IndexType, OutputType >::value,
			void >::type * const = nullptr
		) {
			if( descr & grb::descriptors::use_index ) {
				return static_cast< OutputType >( index );
			} else {
				return static_cast< OutputType >( value );
			}
		}

		template< Descriptor descr,
			typename OutputType, typename IndexType, typename ValueType
		>
		OutputType setIndexOrValue( const IndexType &index, const ValueType &value,
			const typename std::enable_if<
				!std::is_convertible< IndexType, OutputType >::value,
			void >::type * const = nullptr
		) {
			(void)index;
			static_assert( !( descr & grb::descriptors::use_index ),
				"use_index descriptor passed while the index type cannot be cast "
				"to the output type" );
			return static_cast< OutputType >( value );
		}
#endif

	} // namespace internal

	/**
	 * Sets all elements of a vector to the given value.
	 *
	 * Unmasked variant.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function using the execute phase:
	 *   -# consists of \f$ \Theta(n) \f$ work;
	 *   -# moves \f$ \Theta(n) \f$ bytes of memory intra-process;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * Here, \f$ n \f$ is equal to #grb::size( x ).
	 *
	 * A call to this function using the try phase is as defined above, but with
	 * every big-Theta bound replaced by a big-Oh bound.
	 *
	 * A call to this function using the resize phase:
	 *   -# consists of \f$ \mathcal{O}(n) \f$ work;
	 *   -# moves \f$ \mathcal{O}(n) \f$ data intra-process;
	 *   -# may allocate and free dynamic memory, and thus may make the associated
	 *      system calls.
	 *
	 * Note that this is a single user process backend, and hence trivially no
	 * inter-process costs will occur.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename T,
		typename Coords
	>
	RC set(
		Vector< DataType, reference, Coords > &x,
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

		// dynamic checks
		const size_t n = size( x );
		if( (descr & descriptors::dense) && nnz( x ) < n ) {
			return ILLEGAL;
		}

		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// pre-cast value to be copied
		const DataType toCopy = static_cast< DataType >( val );

		// make vector dense if it was not already
		if( !(descr & descriptors::dense) ) {
			internal::getCoordinates( x ).assignAll();
		}
		DataType * const raw = internal::getRaw( x );

#ifdef _H_GRB_REFERENCE_OMP_IO
		#pragma omp parallel
		{
			size_t start, end;
			config::OMP::localRange( start, end, 0, n );
#else
			const size_t start = 0;
			const size_t end = n;
#endif
			for( size_t i = start; i < end; ++ i ) {
				raw[ i ] = internal::template ValueOrIndex< descr, DataType, DataType >::
					getFromScalar( toCopy, i );
			}
#ifdef _H_GRB_REFERENCE_OMP_IO
		}
#endif
		// sanity check
		assert( internal::getCoordinates( x ).nonzeroes() ==
			internal::getCoordinates( x ).size() );

		// done
		return SUCCESS;
	}

	/**
	 * Sets all elements of a vector to the given value.
	 *
	 * Masked variant.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta( nnz( m ) ) \f$ work;
	 *   -# moves \f$ \Theta( nnz( m ) ) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * If grb::descriptors::invert_mask is given, then \f$ nnz( m ) \f$ in the
	 * above shall be interpreted as \f$ size( m ) \f$ instead.
	 * \endparblock
	 *
	 * \todo Revise the above to account for different phases.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename MaskType, typename T,
		typename Coords
	>
	RC set(
		Vector< DataType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &m,
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

		// If the mask is empty: clear the vector
		if( size( m ) == 0 ) {
			return clear( x );
		}

		// dynamic sanity checks
		const size_t sizex = size( x );
		if( sizex != size( m ) ) {
			return MISMATCH;
		}
		if( (descr & descriptors::dense) &&
			(nnz( x ) < sizex || nnz( m ) < sizex)
		) {
			return ILLEGAL;
		}

		// handle trivial resize
		if( phase == RESIZE ) {
			return SUCCESS;
		}
		assert( phase == EXECUTE );

		// make the vector empty unless the dense descriptor is provided
		const bool mask_is_dense = (descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask) && (
				(descr & descriptors::dense) ||
				nnz( m ) == sizex
			);
		if( !((descr & descriptors::dense) && mask_is_dense) ) {
			internal::getCoordinates( x ).clear();
		} else if( mask_is_dense ) {
			// dispatch to faster variant if mask is structurally dense
			return set< descr >( x, val, phase );
		}

		// pre-cast value to be copied and get coordinate handles
		const DataType toCopy = static_cast< DataType >( val );
		DataType * const raw = internal::getRaw( x );
		auto &coors = internal::getCoordinates( x );
		const auto &m_coors = internal::getCoordinates( m );
		const MaskType * const m_p = internal::getRaw( m );

#ifdef _H_GRB_REFERENCE_OMP_IO
		#pragma omp parallel
		{
			auto localUpdate = coors.EMPTY_UPDATE();
			const size_t maxAsyncAssigns = coors.maxAsyncAssigns();
			size_t asyncAssigns = 0;
#endif
			const bool loop_over_vector_length = (descr & descriptors::invert_mask) ||
				( 4 * m_coors.nonzeroes() > 3 * m_coors.size() );
#ifdef _DEBUG
			if( loop_over_vector_length ) {
				std::cout << "\t using loop of size n (the vector length)\n";
			} else {
				std::cout << "\t using loop of size nz (the number of nonzeroes in the vector)\n";
			}
#endif
			const size_t n = loop_over_vector_length ?
				coors.size() :
				m_coors.nonzeroes();
#ifdef _H_GRB_REFERENCE_OMP_IO
			// since masks are irregularly structured, use dynamic schedule to ensure
			// load balance
			#pragma omp for schedule( dynamic,config::CACHE_LINE_SIZE::value() ) nowait
#endif
			for( size_t k = 0; k < n; ++k ) {
				const size_t index = loop_over_vector_length ? k : m_coors.index( k );
				if( !m_coors.template mask< descr >( index, m_p ) ) {
					continue;
				}
#ifdef _H_GRB_REFERENCE_OMP_IO
				if( !coors.asyncAssign( index, localUpdate ) ) {
					(void) ++asyncAssigns;
				}
				if( asyncAssigns == maxAsyncAssigns ) {
					(void) coors.joinUpdate( localUpdate );
					asyncAssigns = 0;
				}
#else
				(void) coors.assign( index );
#endif
				raw[ index ] = internal::ValueOrIndex<
						descr, DataType, DataType
					>::getFromScalar(
						toCopy, index
					);
			}
#ifdef _H_GRB_REFERENCE_OMP_IO
			while( !coors.joinUpdate( localUpdate ) ) {}
		} // end pragma omp parallel
#endif

		// done
		return SUCCESS;
	}

	/**
	 * Sets the element of a given vector at a given position to a given value.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta(1) \f$ work;
	 *   -# moves \f$ \Theta(1) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * \endparblock
	 *
	 * \todo Check and, if needed, revise performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename T, typename Coords
	>
	RC setElement(
		Vector< DataType, reference, Coords > &x,
		const T val,
		const size_t i,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< DataType >::value &&
			!grb::is_object< T >::value, void >::type * const = nullptr
	) {
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
		(void) internal::getCoordinates( x ).assign( i );
		internal::getRaw( x )[ i ] = static_cast< DataType >( val );

#ifdef _DEBUG
		std::cout << "setElement (reference) set index " << i << " to value "
			<< internal::getRaw( x )[ i ] << "\n";
#endif

		// done
		return SUCCESS;
	}

	/**
	 * Sets the content of a given vector \a x to be equal to that of
	 * another given vector \a y.
	 *
	 * Unmasked variant.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta(n) \f$ work;
	 *   -# moves \f$ \Theta(n) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * \endparblock
	 *
	 * \todo Check and, if needed, revise performance semantics.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType, typename Coords >
	RC set(
		Vector< OutputType, reference, Coords > &x,
		const Vector< InputType, reference, Coords > &y,
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
			"grb::set (reference, vector <- vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( !(descr & descriptors::use_index) || !out_is_void,
			"grb::set (reference, vector <- vector, masked): "
			"use_index descriptor cannot be set if output vector is void" );

		// check contract
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}
		// check trivial op
		// note: the below check cannot move after the check that uses getID
		if( n == 0 ) {
			return SUCCESS;
		}
		// continue contract checks
		if( getID( x ) == getID( y ) ) {
			return ILLEGAL;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) || nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
		}

		// on resize
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		// on execute
		assert( phase == EXECUTE );

		// get raw value arrays
		OutputType * __restrict__ const dst = internal::getRaw( x );
		const InputType * __restrict__ const src = internal::getRaw( y );

		// make the vector empty unless the dense descriptor is provided
		if( !(descr & descriptors::dense) ) {
			internal::getCoordinates( x ).clear();
		}

		// get #nonzeroes
		const size_t nz = nnz( y );
#ifdef _DEBUG
		std::cout << "grb::set called with source vector containing "
			<< nz << " nonzeroes." << std::endl;
#endif

#ifndef NDEBUG
		if( src == nullptr ) {
			assert( dst == nullptr );
		}
#endif
		// first copy contents
		if( src == nullptr && dst == nullptr ) {
			// if both source and destination are dense void vectors, this is a no-op
			if( (descr & descriptors::dense) || (
					nnz( x ) == size( x ) && nz == size( y )
				)
			) {
				return SUCCESS;
			}
			// otherwise, copy source nonzero pattern to destination:
#ifdef _H_GRB_REFERENCE_OMP_IO
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, nz );
#else
				const size_t start = 0;
				const size_t end = nz;
#endif
				for( size_t i = start; i < end; ++i ) {
					(void) internal::getCoordinates( x ).asyncCopy(
						internal::getCoordinates( y ), i );
				}
#ifdef _H_GRB_REFERENCE_OMP_IO
			}
#endif
		} else {
			// if the output is a void vector that is furthermore dense, then this is
			// actually also a no-op:
			if( (descr & descriptors::dense) && out_is_void ) {
				return SUCCESS;
			}
			// otherwise, the regular copy variant:
#ifdef _H_GRB_REFERENCE_OMP_IO
			#pragma omp parallel
			{
				size_t start, end;
				config::OMP::localRange( start, end, 0, nz );
#else
				const size_t start = 0;
				const size_t end = nz;
#endif
				for( size_t i = start; i < end; ++i ) {
					size_t index;
					if( !(descr & descriptors::dense) ) {
						index = internal::getCoordinates( x ).asyncCopy(
							internal::getCoordinates( y ), i );
					} else {
						index = i;
					}
					if( !out_is_void && !in_is_void ) {
						dst[ index ] = internal::setIndexOrValue< descr, OutputType >(
							index, src[ index ] );
					}
				}
#ifdef _H_GRB_REFERENCE_OMP_IO
			}
#endif
		}

		// set number of nonzeroes
		if( !(descr & descriptors::dense) ) {
			internal::getCoordinates( x ).joinCopy( internal::getCoordinates( y ) );
		}

		// done
		return SUCCESS;
	}

	/**
	 * Sets the content of a given vector \a x to be equal to that of
	 * another given vector \a y.
	 *
	 * Masked variant.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# consists of \f$ \Theta( \min\{ nnz( mask ), nnz( y ) \} ) \f$ work;
	 *   -# moves \f$ \Theta( \min\{ nnz( mask ), nnz( y ) \} ) \f$ bytes of memory;
	 *   -# does not allocate nor free any dynamic memory;
	 *   -# shall not make any system calls.
	 * If grb::descriptors::invert_mask is given, then \f$ nnz( mask ) \f$ in the
	 * above shall be considered equal to \f$ nnz( y ) \f$.
	 * \endparblock
	 *
	 * \todo Check and, if needed, revise performance semantics.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename MaskType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, reference, Coords > &x,
		const Vector< MaskType, reference, Coords > &mask,
		const Vector< InputType, reference, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
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
			"grb::set (reference, vector <- vector, masked): "
			"if input is void, then the output must be also" );
		static_assert( !(descr & descriptors::use_index) || !out_is_void,
			"grb::set (reference, vector <- vector, masked): "
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
		if( descr & descriptors::dense ) {
			if( nnz( x ) < grb::size( x ) ||
				nnz( y ) < grb::size( y ) ||
				nnz( mask ) < grb::size( mask )
			) {
				return ILLEGAL;
			}
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

		// handle non-trivial, fully masked vector copy
		const auto &m_coors = internal::getCoordinates( mask );
		const auto &y_coors = internal::getCoordinates( y );
		auto &x_coors = internal::getCoordinates( x );

		// make the vector empty unless the dense descriptor is provided
		const bool mask_is_dense = (descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask) && (
				(descr && descriptors::dense) ||
				nnz( mask ) == grb::size( mask )
			);
		if( !((descr & descriptors::dense) && mask_is_dense) ) {
			internal::getCoordinates( x ).clear();
		}

		// choose optimal loop size
		const bool loop_over_y = (descr & descriptors::invert_mask) ||
			( y_coors.nonzeroes() < m_coors.nonzeroes() );
		const size_t n = loop_over_y ? y_coors.nonzeroes() : m_coors.nonzeroes();

#ifdef _H_GRB_REFERENCE_OMP_IO
		// keeps track of updates of the sparsity pattern
		#pragma omp parallel
		{
			// keeps track of nonzeroes that the mask ignores
			internal::Coordinates< reference >::Update local_update =
				x_coors.EMPTY_UPDATE();
			const size_t maxAsyncAssigns = x_coors.maxAsyncAssigns();
			size_t asyncAssigns = 0;
			RC local_rc = SUCCESS;
			// since masks are irregularly structured, use dynamic schedule to ensure
			// load balance
			#pragma omp for schedule( dynamic, config::CACHE_LINE_SIZE::value() ) nowait
			for( size_t k = 0; k < n; ++k ) {
				const size_t i = loop_over_y ? y_coors.index( k ) : m_coors.index( k );
				// if not masked, continue
				if( !m_coors.template mask< descr >( i, internal::getRaw( mask ) ) ) {
					continue;
				}
				// if source has nonzero
				if( loop_over_y || y_coors.assigned( i ) ) {
					// get value
					if( !out_is_void && !in_is_void ) {
						internal::getRaw( x )[ i ] =
							internal::ValueOrIndex< descr, OutputType, InputType >::getFromArray(
								internal::getRaw( y ), [] (const size_t i) {return i;}, i
							);
					}
					// check if destination has nonzero
					if( !x_coors.asyncAssign( i, local_update ) ) {
						(void) ++asyncAssigns;
					}
				}
				if( asyncAssigns == maxAsyncAssigns ) {
					const bool was_empty = x_coors.joinUpdate( local_update );
#ifdef NDEBUG
					(void) was_empty;
#else
					assert( !was_empty );
#endif
					asyncAssigns = 0;
				}
			}
			while( !x_coors.joinUpdate( local_update ) ) {}
			if( local_rc != SUCCESS ) {
				ret = local_rc;
			}
		} // end omp parallel for
#else
		for( size_t k = 0; k < n; ++k ) {
			const size_t i = loop_over_y ? y_coors.index( k ) : m_coors.index( k );
			if( !m_coors.template mask< descr >( i, internal::getRaw( mask ) ) ) {
				continue;
			}
			if( loop_over_y || internal::getCoordinates( y ).assigned( i ) ) {
				if( !out_is_void && !in_is_void ) {
					// get value
					(void) x_coors.assign( i );
					internal::getRaw( x )[ i ] =
						internal::ValueOrIndex< descr, OutputType, InputType >::getFromArray(
							internal::getRaw( y ), [] (const size_t i) {return i;}, i
						);
				}
			}
		}
#endif

		// done
		return ret;
	}

	namespace internal {

		template<
			bool A_is_mask,
			Descriptor descr,
			typename OutputType, typename InputType1,
			typename InputType2 = const OutputType,
			typename RIT1, typename CIT1, typename NIT1,
			typename RIT2, typename CIT2, typename NIT2
		>
		RC set(
			Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
			const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
			const InputType2 * __restrict__ id = nullptr
		) noexcept {
#ifdef _DEBUG
			std::cout << "Called grb::set (matrices, reference), execute phase\n";
#endif
			// static checks
			NO_CAST_ASSERT(
				( !( descr & descriptors::no_casting ) ||
				( !A_is_mask && std::is_same< InputType1, OutputType >::value ) ),
				"internal::grb::set", "called with non-matching value types"
			);
			NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
				( A_is_mask && std::is_same< InputType2, OutputType >::value ) ),
				"internal::grb::set", "Called with non-matching value types"
			);

			// run-time checks
			const size_t m = nrows( A );
			const size_t n = ncols( A );
			if( nrows( C ) != m ) {
				return MISMATCH;
			}
			if( ncols( C ) != n ) {
				return MISMATCH;
			}
			if( A_is_mask ) {
				assert( id != nullptr );
			}

			// catch trivial cases
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}
			const size_t nz = nnz( A );
			if( nz == 0 ) {
#ifdef _DEBUG
				std::cout << "\t input matrix has no nonzeroes, "
					<< "simply clearing output matrix...\n";
#endif
				return clear( C );
			}
			if( nz > capacity( C ) ) {
#ifdef _DEBUG
				std::cout << "\t output matrix does not have sufficient capacity to "
					<< "complete requested operation\n";
#endif
				const RC clear_rc = clear( C );
				if( clear_rc != SUCCESS ) {
					return PANIC;
				} else {
					return FAILED;
				}
			}

#ifdef _H_GRB_REFERENCE_OMP_IO
			#pragma omp parallel
#endif
			{
				size_t range = internal::getCRS( C ).copyFromRange( nz, m );
#ifdef _H_GRB_REFERENCE_OMP_IO
				size_t start, end;
				config::OMP::localRange( start, end, 0, range );
#else
				const size_t start = 0;
				size_t end = range;
#endif
				if( A_is_mask ) {
					internal::getCRS( C ).template copyFrom< true >(
						internal::getCRS( A ), nz, m, start, end, id
					);
				} else {
					internal::getCRS( C ).template copyFrom< false >(
						internal::getCRS( A ), nz, m, start, end
					);
				}
				range = internal::getCCS( C ).copyFromRange( nz, n );
#ifdef _H_GRB_REFERENCE_OMP_IO
				config::OMP::localRange( start, end, 0, range );
#else
				end = range;
#endif
				if( A_is_mask ) {
					internal::getCCS( C ).template copyFrom< true >(
						internal::getCCS( A ), nz, n, start, end, id
					);
				} else {
					internal::getCCS( C ).template copyFrom< false >(
						internal::getCCS( A ), nz, n, start, end
					);
				}
			}
			internal::setCurrentNonzeroes( C, nz );

			// done
			return SUCCESS;
		}

	} // end namespace internal::grb

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType, reference, RIT2, CIT2, NIT2 > &A,
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
		std::cout << "Called grb::set (matrix-to-matrix, reference)" << std::endl;
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
		Matrix< OutputType, reference, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, reference, RIT2, CIT2, NIT2 > &A,
		const InputType2 &val,
		const Phase &phase = EXECUTE
	) noexcept {
		static_assert( !std::is_same< OutputType, void >::value,
			"internal::grb::set (masked set to value): cannot have a pattern "
			"matrix as output" );
#ifdef _DEBUG
		std::cout << "Called grb::set (matrix-to-value-masked, reference)\n";
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

	/**
	 * Ingests raw data into a GraphBLAS vector.
	 *
	 * This is the direct variant without iterator output position updates.
	 *
	 * The input is given by iterators. The \a start position will be assumed to
	 * contain a value to be added to this vector at index \a 0. The \a start
	 * position will be incremented up to \a n times.
	 *
	 * An element found at a position that has been incremented \a i times will
	 * be added to this vector at index \a i.
	 *
	 * If, when adding a value \a x to index \a i, an existing value at the same
	 * index position was found, then the given \a Dup will be used to
	 * combine the two values. \a Dup must be a binary operator; the old
	 * value will be used as the left-hand side input, the new value from the
	 * current iterator position as its right-hand side input. The result of
	 * applying the operator defines the new value at position \a i.
	 *
	 * \warning If there is no \a Dup type nor \a dup instance provided then
	 * grb::operators::right_assign will be assumed-- this means new values will
	 * simply overwrite old values.
	 *
	 * \warning If, on input, \a x is not empty, new values will be combined with
	 * old ones by use of \a Dup.
	 *
	 * \note To ensure all old values of \a x are deleted, simply preface a call
	 * to this function by one to grb::clear(x).
	 *
	 * If, after \a n increments of the \a start position, that incremented
	 * position is not found to equal the given \a end position, this function
	 * will return grb::MISMATCH. The \a n elements that were found, however,
	 * will have been added to the vector; the remaining items in the iterator
	 * range will simply be ignored.
	 * If \a start was incremented \a i times with \f$ i < n \f$ and is found to
	 * be equal to \a end, grb::MISMATCH will be returned as well. The \a i
	 * values that were extracted from \a start on will still have been added to
	 * the output vector \a x.
	 *
	 * Since this function lacks explicit input for the index of each vector
	 * element, IOMode::parallel is <em>not supported</em>.
	 *
	 * \warning If \a P is larger than one and \a IOMode is parallel, this
	 *          function will return grb::RC::ILLEGAL and will have no other
	 *          effect.
	 *
	 * @tparam descr        The descriptors passed to this function call.
	 * @tparam InputType    The type of the vector elements.
	 * @tparam Dup          The class of the operator used to resolve
	 *                      duplicated entries.
	 * @tparam fwd_iterator The type of the input forward iterator.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation for default behaviour;
	 *   -# grb::descriptors::no_casting which will cause compilation to fail
	 *      whenever \a InputType does not match \a fwd_iterator::value_type.
	 * \endparblock
	 *
	 * @param[in,out] x     The vector to update with new input values.
	 * @param[in,out] start On input:  the start position of the input forward
	 *                                 iterator.
	 *                      On output: the position after the last increment
	 *                                 performed while calling this function.
	 * @param[in]     end   The end position of the input forward iterator.
	 * @param[in]     dup   The operator to use for resolving write conflicts.
	 *
	 * \warning Use of this function, which is grb::IOMode::sequential, leads to
	 *          unscalable performance and should thus be used with care!
	 *
	 * @return grb::SUCCESS  Whenever \a n new elements from \a start to \a end
	 *                       were successfully added to \a x, where \a n is the
	 *                       size of this vector.
	 * @return grb::MISMATCH Whenever the number of elements between \a start to
	 *                       \a end does not equal \a n. When this is returned,
	 *                       the output vector \a x is still updated with
	 *                       whatever values that were successfully extracted
	 *                       from \a start. If this is not exected behaviour, the
	 *                       user could, for example, catch this error code and
	 *                       call grb::clear.
	 * @return grb::OUTOFMEM Whenever not enough capacity could be allocated to
	 *                       store the input from \a start to \a end. The output
	 *                       vector \a x is guaranteed to contain all values up to
	 *                       the returned position \a start.
	 * @return grb::ILLEGAL  Whenever \a mode is parallel while the number of user
	 *                       processes is larger than one.
	 * @return grb::PANIC    Whenever an un-mitigable error occurs. The state of
	 *                       the GraphBLAS library and all associated containers
	 *                       becomes undefined.
	 *
	 * \parblock
	 * \par Performance semantics:
	 * A call to this function
	 *   -# comprises \f$ \mathcal{O}( n ) \f$ work <em>per user process</em>,
	 *      where \a n is the vector size.
	 *   -# results in at most \f$ n \mathit{sizeof}( T ) \f$ bytes of data
	 *      movement, where \a T is the underlying data type of the input
	 *      iterator, <em>per user process</em>.
	 *   -# Results in at most \f$   n \mathit{sizeof}( \mathit{InputType} ) +
	 *                             2 n \mathit{sizeof}( \mathit{bool} ) \f$
	 *      bytes of data movement that may be distributed over multiple user
	 *      processes.
	 *   -# if the capacity of this vector is not large enough to hold \a n
	 *      elements, a call to this function may allocate \f$ \mathcal{O}( n ) \f$
	 *      new bytes of memory which may be distributed over multiple user
	 *      processes.
	 *   -# if the capacity of this vector is not large enough to hold \a n
	 *      elements, this function may make system calls at any of the user
	 *      processes.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator, typename Coords,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, reference, Coords > &x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode, const Dup & dup = Dup()
	) {
		// static sanity check
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< InputType, decltype( *std::declval< fwd_iterator >() ) >::value ),
			"grb::buildVector (reference implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// in the sequential reference implementation, the number of user processes always equals 1
		// therefore the sequential and parallel modes are equivalent
#ifndef NDEBUG
		assert( mode == SEQUENTIAL || mode == PARALLEL );
#else
		(void) mode;
#endif

		// declare temporary to meet delegate signature
		const fwd_iterator start_pos = start;

		// do delegate
		return x.template build< descr >( dup, start_pos, end, start );
	}

	/**
	 * Ingests raw data into a GraphBLAS vector. This is the coordinate-wise
	 * version.
	 *
	 * The input is given by iterators. The \a val_start position will be assumed
	 * to contain a value to be added to this vector at index pointed to by
	 * \a ind_start. The same remains true if both \a val_start and \a ind_start
	 * positions are incremented.
	 *
	 * When multiple iterator position pairs correspond to a new nonzero value
	 * at the same position \a i, then those values are combined using the given
	 * \a duplicate operator. \a Merger must be an \em associative binary
	 * operator.
	 *
	 * If, when adding a value \a x to index \a i an existing value at the same
	 * index position was found, then the given \a dup will be used to combine
	 * the two values. \a Dup must be a binary operator; the old value will be
	 * used as the left-hand side input, the new value from the current iterator
	 * position as its right-hand side input. The result of applying the operator
	 * defines the new value at position \a i.
	 *
	 * \warning If there is no \a Dup type nor \a dup instance provided then
	 * grb::operators::right_assign will be assumed-- this means new values will
	 * simply overwrite old values.
	 *
	 * \warning If, on input, \a x is not empty, new values will be combined with
	 * old ones by use of \a Dup.
	 *
	 * \note To ensure all old values of \a x are deleted, simply preface a call
	 * to this function by one to grb::clear(x).
	 *
	 * If, after \a n increments of the \a start position, that incremented
	 * position is not found to equal the given \a end position, this function
	 * will return grb::MISMATCH. The \a n elements that were found, however,
	 * will have been added to the vector; the remaining items in the iterator
	 * range will simply be ignored.
	 * If \a start was incremented \a i times with \f$ i < n \f$ and is found to
	 * be equal to \a end, grb::MISMATCH will be returned as well. The \a i
	 * values that were extracted from \a start on will still have been added to
	 * the output vector \a x.
	 *
	 * This function, like with all GraphBLAS I/O, has two modes as detailed in
	 * \a IOMode. In case of IOMode::sequential, all \a P user processes are
	 * expected to provide iterators with exactly the same context across all
	 * processes.
	 * In case of IOMode::parallel, the \a P user processes are expected to be
	 * provided disjoint parts of the input that make up the entire vector. The
	 * following two vectors \a x and \a y thus are equal:
	 * \code
	 * size_t s = ...; //let s be 0 or 1, the ID of this user process;
	 *                 //assume P=2 user processes total.
	 * double raw[8] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	 * size_t ind[8] = {0  , 1,   2,   3,   4,   5,   6,   7,   8,   9,   10};
	 *
	 * const double * const r = &(raw[0]); //get a standard C pointer to the data.
	 * const size_t * const i = &(ind[0]); //get a C pointer to the indices.
	 *
	 * grb::init( s, 2 );
	 * grb::Vector<double> x( 8 ), y( 8 );
	 * x.buildVector( x, i,       i + 8      , r,       r + 8,       sequential );
	 * y.buildVector( y, i + s*4, i + s*4 + 4, r + s*4, r + s*4 + 4, parallel );
	 * ...
	 *
	 * grb::finalize();
	 * \endcode
	 *
	 * \warning While the above is semantically equivalent, their performance
	 *          characteristics are not. Please see the below for details.
	 *
	 * @tparam descr         The descriptor to be used (descriptors::no_operation
	 *                       if left unspecified).
	 * @tparam Dup           The type of the operator used to resolve inputs to
	 *                       pre-existing vector contents. The default Dup simply
	 *                       overwrites pre-existing content.
	 * @tparam InputType     The type of values stored by the vector.
	 * @tparam fwd_iterator1 The type of the iterator to be used for index value
	 *                       input.
	 * @tparam fwd_iterator2 The type of the iterator to be used for nonzero
	 *                       value input.
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation for default behaviour;
	 *   -# grb::descriptors::no_casting which will cause compilation to fail
	 *      whenever 1) \a InputType does not match \a fwd_iterator2::value_type,
	 *      or whenever 2) \a fwd_iterator1::value_type is not an integral type.
	 * \endparblock
	 *
	 * @param[out]    x         Where the ingested data is to be added.
	 * @param[in,out] ind_start On input:  the start position of the indices
	 *                                     to be inserted. This iterator reports,
	 *                                     for every nonzero to be inserted, its
	 *                                     index value.
	 *                          On output: the position after the last increment
	 *                                     performed while calling this function.
	 * @param[in]  ind_end      The end iterator corresponding to \a ind_start.
	 * @param[in]  val_start    On input:  the start iterator of the auxiliary
	 *                                     data to be inserted. This iterator
	 *                                     reports, for every nonzero to be
	 *                                     inserted, its nonzero value.
	 *                          On output: the position after the last increment
	 *                                     performed while calling this function.
	 * @param[in]  val_end     The end iterator corresponding to \a val_start.
	 * @param[in]  mode        The IOMode of this call. By default this is set to
	 *                         IOMode::parallel.
	 * @param[in]  dup         The operator that resolves input to pre-existing
	 *                         vector entries.
	 *
	 * \warning Use of IOMode::sequential leads to unscalable performance and
	 *          should be used with care.
	 *
	 * @return grb::SUCCESS  Whenever \a n new elements from \a start to \a end
	 *                       were successfully added to \a x, where \a n is the
	 *                       size of this vector.
	 * @return grb::MISMATCH Whenever an element from \a ind_start is larger or
	 *                       equal to \a n. When this is returned, the output
	 *                       vector \a x is still updated with whatever values
	 *                       that were successfully extracted from \a ind_start
	 *                       and \a val_start.
	 *                       If this is not exected behaviour, the user could,
	 *                       for example, catch this error code and followed by
	 *                       a call to grb::clear.
	 * @return grb::OUTOFMEM Whenever not enough capacity could be allocated to
	 *                       store the input from \a start to \a end. The output
	 *                       vector \a x is guaranteed to contain all values up to
	 *                       the returned position \a start.
	 * @return grb::PANIC    Whenever an un-mitigable error occurs. The state of
	 *                       the GraphBLAS library and all associated containers
	 *                       becomes undefined.
	 *
	 * \parblock
	 * \par Performance semantics
	 * A call to this function
	 *   -# comprises \f$ \mathcal{O}( n ) \f$ work, where \a n is the number of
	 *      elements the given iterator pairs point to.
	 *   -# results in at most
	 *         \f$ n ( \mathit{sizeof}( T ) \mathit{sizeof}( U ) \f$
	 *      bytes of data movement, where \a T and \a U are the underlying data
	 *      types of each of the input iterators, <em>per user process</em>.
	 *   -# Results in at most \f$   n \mathit{sizeof}( \mathit{InputType} ) +
	 *                             2 n \mathit{sizeof}( \mathit{bool} ) \f$
	 *      bytes of data movement.
	 *   -# if the capacity of this vector is not large enough to hold \a n
	 *      elements, a call to this function may allocate
	 *         \f$ \mathcal{O}( n ) \f$
	 *      new bytes of memory.
	 *   -# no new dynamic memory shall be allocated.
	 *   -# no system calls shall be made.
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator1, typename fwd_iterator2,
		typename Coords, class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, reference, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		// static sanity check
		NO_CAST_ASSERT( ( !( descr & descriptors::no_casting ) ||
			std::is_same< InputType, decltype( *std::declval< fwd_iterator2 >() ) >::value ||
			std::is_integral< decltype( *std::declval< fwd_iterator1 >() ) >::value ),
			"grb::buildVector (reference implementation)",
			"At least one input iterator has incompatible value types while "
			"no_casting descriptor was set" );

		// in the sequential reference implementation, the number of user processes always equals 1
		// therefore the sequential and parallel modes are equivalent
#ifndef NDEBUG
		assert( mode == SEQUENTIAL || mode == PARALLEL );
#else
		(void)mode;
#endif

		// call the private member function that provides this functionality
		return x.template build< descr >( dup, ind_start, ind_end, val_start, val_end );
	}

	/*
	 * @see grb::buildMatrix.
	 *
	 * This function has only been implemented for descriptors::no_duplicates.
	 *
	 * @see grb::buildMatrixUnique calls this function when
	 *                             grb::descriptors::no_duplicates is passed.
	 *
	 * \todo Decide whether or not to keep this function. A reasonable alternative
	 *       may be to simply only support buildMatrixUnique...
	 *
	template<
	    Descriptor descr = descriptors::no_operation,
	    template< typename, typename, typename > class accum = operators::right_assign,
	    template< typename, typename, typename > class dup   = operators::add,
	    typename InputType,
	    typename fwd_iterator1 = const size_t *__restrict__,
	    typename fwd_iterator2 = const size_t *__restrict__,
	    typename fwd_iterator3 = const InputType  *__restrict__,
	    typename length_type = size_t
	>
	RC buildMatrix(
	    Matrix< InputType, reference > &A,
	    const fwd_iterator1 I,
	    const fwd_iterator2 J,
	    const fwd_iterator3 V,
	    const length_type nz,
	    const IOMode mode
	) {
	    //delegate in case of no duplicats
	    if( descr & descriptors::no_duplicates ) {
	        return buildMatrixUnique( A, I, J, V, nz, mode );
	    }
	    assert( false );
	    return PANIC;
	}*/

	/**
	 * Calls the other #buildMatrixUnique variant.
	 * @see grb::buildMatrixUnique for the user-level specification.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *
	 *        -# This function contains
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ amount of work.
	 *        -# This function may dynamically allocate
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of memory.
	 *        -# A call to this function will use \f$ \mathcal{O}(m+n) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function will copy each input forward iterator at most
	 *           \em once; the three input iterators \a I, \a J, and \a V thus
	 *           may have exactly one copyeach, meaning that all input may be
	 *           traversed only once.
	 *        -# Each of the at most three iterator copies will be incremented
	 *           at most \f$ \mathit{nz} \f$ times.
	 *        -# Each position of the each of the at most three iterator copies
	 *           will be dereferenced exactly once.
	 *        -# This function moves
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of data.
	 *        -# This function will likely make system calls.
	 *
	 * \endparblock
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename RIT, typename CIT, typename NIT,
		typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, reference, RIT, CIT, NIT > &A,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		// parallel or sequential mode are equivalent for reference implementation
		assert( mode == PARALLEL || mode == SEQUENTIAL );
#ifdef NDEBUG
		(void)mode;
#endif
#ifdef _DEBUG
		std::cout << "buildMatrixUnique (reference) called, delegating to matrix class\n";
#endif
		return A.template buildMatrixUnique< descr >( start, end, mode );
	}

	/**
	 * \internal
	 *
	 * Uses pointers to internal buffer areas that are guaranteed to exist
	 * (except for empty vectors). The buffer areas reside in the internal
	 * coordinates class.
	 *
	 * \endinternal
	 */
	template< typename InputType, typename Coords >
	uintptr_t getID( const Vector< InputType, reference, Coords > &x ) {
		assert( grb::size( x ) != 0 );
		const uintptr_t ret = x._id;
#ifdef _DEBUG
		std::cerr << "In grb::getID (reference, vector).\n"
			<< "\t returning deterministic ID " << ret << "\n";
#endif
		return ret;
	}

	template<>
	RC wait< reference >();

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType, typename Coords,
		typename ... Args
	>
	RC wait(
		const Vector< InputType, reference, Coords > &x,
		const Args &... args
	) {
		(void) x;
		return wait( args... );
	}

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType, typename RIT, typename CIT, typename NIT,
		 typename... Args
	>
	RC wait(
		const Matrix< InputType, reference, RIT, CIT, NIT > &A,
		const Args &... args
	) {
		(void) A;
		return wait( args... );
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

// parse again for reference_omp backend
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_IO
  #define _H_GRB_REFERENCE_OMP_IO
  #define reference reference_omp
  #include "graphblas/reference/io.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_IO
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_IO

