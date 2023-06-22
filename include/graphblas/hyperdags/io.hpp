
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
 * Provides the I/O primitives for the HyperDAGs backend
 *
 * @author A. Karanasiou
 * @date 3rd of March 2022
 */

#include <graphblas/config.hpp>

#include <array>


namespace grb {

	// input:

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator, typename Coords,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, hyperdags, Coords > &x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode, const Dup &dup = Dup()
	) {
		const RC ret = buildVector<descr>(
			internal::getVector(x), start, end, mode, dup
		);
		if( ret != SUCCESS ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&start
		);
		std::array< const void *, 1 > sourcesP{ &start };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator1, typename fwd_iterator2,
		typename Coords, class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, hyperdags, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		const RC ret = buildVector< descr >(
			internal::getVector(x), ind_start, ind_end, val_start, val_end, mode, dup
		);
		if( ret != SUCCESS ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&ind_start
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&val_start
		);
		std::array< const void *, 2 > sourcesP{ &ind_start, &val_start };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR_WITH_VALUES,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename RIT, typename CIT, typename NIT,
		typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		fwd_iterator start,
		const fwd_iterator end,
		const IOMode mode
	) {
		const RC ret = buildMatrixUnique< descr >(
			internal::getMatrix(A), start, end, mode
		);
		if( ret != SUCCESS ) { return ret; }
		if( ncols( A ) == 0 || nrows( A ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&start
		);
		std::array< const void *, 1 > sourcesP{ &start };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILDMATRIXUNIQUE_MATRIX_START_END_MODE,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType,
		typename T, typename Coords
	>
	RC setElement(
		Vector< DataType, hyperdags, Coords > &x,
		const T val,
		const size_t i,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		const RC ret = setElement< descr >(
			internal::getVector( x ), val, i, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		// x cannot be empty here or setElement would have failed-- no need to catch
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&i
		);
		std::array< const void *, 2 > sourcesP{ &val, &i };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_VECTOR_ELEMENT,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Coords,
		typename T
	>
	RC set(
		Vector< DataType, hyperdags, Coords > &x, const T val,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		const RC ret = set< descr >( internal::getVector( x ), val, phase );
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		if( !(descr & descriptors::use_index) ) {
			internal::hyperdags::generator.addSource(
				internal::hyperdags::SCALAR,
				&val
			);
			std::array< const void *, 1 > sourcesP{ &val };
			std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
			std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::SET_USING_VALUE,
				sourcesP.begin(), sourcesP.end(),
				sourcesC.begin(), sourcesC.end(),
				destinations.begin(), destinations.end()
			);
		} else {
			std::array< const void *, 0 > sourcesP{};
			std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
			std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::SET_USING_VALUE,
				sourcesP.begin(), sourcesP.end(),
				sourcesC.begin(), sourcesC.end(),
				destinations.begin(), destinations.end()
			);
		}
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename MaskType, typename T,
		typename Coords
	>
	RC set(
		Vector< DataType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &m,
		const T val,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value,
		void >::type * const = nullptr
	) {
		if( size( m ) == 0 ) { return set< descr >( x, val, phase ); }
		const RC ret = set< descr >(
			internal::getVector(x), internal::getVector(m),
			val, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		std::array< const void *, 1 > sourcesP{ &val };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(x) ),
			getID( internal::getVector(m) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_SCALAR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename MaskType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, hyperdags, Coords > &x,
		const Vector< MaskType, hyperdags, Coords > &mask,
		const Vector< InputType, hyperdags, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value,
		void >::type * const = nullptr
	) {
		if( size( mask ) == 0 ) { return set< descr >( x, y, phase ); }
		const RC ret = set< descr >(
			internal::getVector(x),
			internal::getVector(mask), internal::getVector(y),
			phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 3 > sourcesC{
			getID( internal::getVector(mask) ),
			getID( internal::getVector(y) ),
			getID( internal::getVector(x) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType, typename Coords
	>
	RC set(
		Vector< OutputType, hyperdags, Coords > &x,
		const Vector< InputType, hyperdags, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		const RC ret = set< descr >(
			internal::getVector(x), internal::getVector(y), phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getVector(y) ),
			getID( internal::getVector(x) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_FROM_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< OutputType, hyperdags, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType, hyperdags, RIT2, CIT2, NIT2 > &A,
		const Phase &phase = EXECUTE
	) {
		const RC ret = set< descr >(
			internal::getMatrix( C ), internal::getMatrix( A ), phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( C ) == 0 || ncols( C ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(C) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT1, typename CIT1, typename NIT1,
		typename RIT2, typename CIT2, typename NIT2
	>
	RC set(
		Matrix< OutputType, hyperdags, RIT1, CIT1, NIT1 > &C,
		const Matrix< InputType1, hyperdags, RIT2, CIT2, NIT2 > &A,
		const InputType2 &val,
		const Phase &phase = EXECUTE
	) {
		const RC ret = set< descr >(
			internal::getMatrix( C ), internal::getMatrix( A ),
			val, phase
		);
		if( ret != SUCCESS ) { return ret; }
		if( phase != EXECUTE ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		std::array< const void *, 1 > sourcesP{ &val };
		std::array< uintptr_t, 2 > sourcesC{
			getID( internal::getMatrix(A) ),
			getID( internal::getMatrix(C) )
		};
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(C) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX_INPUT2,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, hyperdags, Coords > &x ) {
		const RC ret = clear( internal::getVector( x ) );
		if( ret != SUCCESS ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CLEAR_VECTOR,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		const RC ret = clear( internal::getMatrix(A) );
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		std::array< const void *, 0 > sourcesP{};
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CLEAR_MATRIX,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	// getters:

	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, hyperdags, Coords > &x ) {
		return size (internal::getVector(x));
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nrows( const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		return nrows(internal::getMatrix(A));
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t ncols( const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		return ncols(internal::getMatrix(A));
	}

	template< typename DataType, typename Coords >
	size_t capacity( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
		return capacity(internal::getVector( x ));
	}

	template< typename DataType, typename RIT, typename CIT, typename NIT >
	size_t capacity( const Matrix< DataType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		return capacity(internal::getMatrix( A ));
	}

	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
		return nnz( internal::getVector( x ) );
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	size_t nnz( const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		return nnz(internal::getMatrix(A));
	}

	template< typename InputType, typename Coords >
	uintptr_t getID( const Vector< InputType, hyperdags, Coords > &x ) {
		return getID(internal::getVector( x ));
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	uintptr_t getID( const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) {
		return getID(internal::getMatrix( A ));
	}

	// resizers:

	template< typename InputType, typename Coords >
	RC resize(
		Vector< InputType, hyperdags, Coords > &x,
		const size_t new_nz
	) noexcept {
		const RC ret = resize( internal::getVector( x ), new_nz );
		if( ret != SUCCESS ) { return ret; }
		if( size( x ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&new_nz
		);
		std::array< const void *, 1 > sourcesP{ &new_nz };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getVector(x) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getVector(x) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::RESIZE,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC resize(
		Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		const size_t new_nz
	) noexcept {
		const RC ret = resize( internal::getMatrix(A), new_nz );
		if( ret != SUCCESS ) { return ret; }
		if( nrows( A ) == 0 || ncols( A ) == 0 ) { return ret; }
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&new_nz
		);
		std::array< const void *, 1 > sourcesP{ &new_nz };
		std::array< uintptr_t, 1 > sourcesC{ getID( internal::getMatrix(A) ) };
		std::array< uintptr_t, 1 > destinations{ getID( internal::getMatrix(A) ) };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::RESIZE_MATRIX,
			sourcesP.begin(), sourcesP.end(),
			sourcesC.begin(), sourcesC.end(),
			destinations.begin(), destinations.end()
		);
		return ret;
	}

	// nonblocking I/O:

	template<>
	RC wait< hyperdags >();

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType, typename Coords,
		typename ... Args
	>
	RC wait(
		const Vector< InputType, hyperdags, Coords > &x,
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
		const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		const Args &... args
	) {
		(void) A;
		return wait( args... );
	}

} // namespace grb

