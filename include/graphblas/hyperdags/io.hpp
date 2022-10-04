
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
 * @author A. Karanasiou
 * @date 3rd of March 2022
 */

#include <graphblas/config.hpp>


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
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&start
		);
		std::array< const void *, 2 > sources{ &start, &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return buildVector<descr>( internal::getVector(x), start, end, mode, dup );
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
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&ind_start
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&val_start
		);
		std::array< const void *, 3 > sources{ &x, &ind_start, &val_start };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILD_VECTOR_WITH_VALUES,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return buildVector< descr >(
			internal::getVector(x), ind_start, ind_end, val_start, val_end, mode, dup
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, hyperdags > &A,
		fwd_iterator start,
		const fwd_iterator end,
		const IOMode mode
	) {
		internal::hyperdags::generator.addSource(
			internal::hyperdags::ITERATOR,
			&start
		);
		std::array< const void *, 2 > sources{ &start, &A };
		std::array< const void *, 1 > destinations{ &A };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::BUILDMATRIXUNIQUE_MATRIX_START_END_MODE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return buildMatrixUnique< descr >( internal::getMatrix(A), start, end, mode );
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
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&i
		);
		std::array< const void *, 3 > sources{ &x, &val, &i };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_VECTOR_ELEMENT,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return setElement< descr >( internal::getVector( x ), val, i, phase );
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
		if( !(descr & descriptors::use_index) ) {
			internal::hyperdags::generator.addSource(
				internal::hyperdags::SCALAR,
				&val
			);
			std::array< const void *, 2 > sources{ &x, &val };
			std::array< const void *, 1 > destinations{ &x };
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::SET_USING_VALUE,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
			);
		} else {
			std::array< const void *, 1 > sources{ &x };
			std::array< const void *, 1 > destinations{ &x };
			internal::hyperdags::generator.addOperation(
				internal::hyperdags::SET_USING_VALUE,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
			);
		}
		return set< descr >( internal::getVector( x ), val, phase );
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
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		std::array< const void *, 3 > sources{ &x, &m, &val };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_SCALAR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set< descr >(
			internal::getVector(x), internal::getVector(m),
			val, phase
		);
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
		std::array< const void *, 3 > sources{ &mask, &y, &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_USING_MASK_AND_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set< descr >(
			internal::getVector(x),
			internal::getVector(mask), internal::getVector(y),
			phase
		);
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
		std::array< const void *, 2 > sources{ &y, &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_FROM_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set< descr >( internal::getVector(x), internal::getVector(y), phase );
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename RIT, typename CIT, typename NIT
	>
	RC set(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType, hyperdags, RIT, CIT, NIT > &A,
		const Phase &phase = EXECUTE
	) {
		std::array< const void *, 2 > sources{ &A, &C };
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set< descr >(
			internal::getMatrix( C ), internal::getMatrix( A ), phase
		);
	}

	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT
	>
	RC set(
		Matrix< OutputType, hyperdags, RIT, CIT, NIT > &C,
		const Matrix< InputType1, hyperdags, RIT, CIT, NIT > &A,
		const InputType2 &val,
		const Phase &phase = EXECUTE
	) {
		internal::hyperdags::generator.addSource(
			internal::hyperdags::SCALAR,
			&val
		);
		std::array< const void *, 3 > sources{ &A, &val, &C };
		std::array< const void *, 1 > destinations{ &C };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SET_MATRIX_MATRIX_INPUT2,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return set< descr >(
			internal::getMatrix( C ), internal::getMatrix( A ),
			val, phase
		);
	}

	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, hyperdags, Coords > &x ) {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CLEAR_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return clear( internal::getVector( x ) );
	}

	template< typename InputType, typename RIT, typename CIT, typename NIT >
	RC clear( Matrix< InputType, hyperdags, RIT, CIT, NIT > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 1 > destinations{ &A };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CLEAR_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		// delegate
		return clear( internal::getMatrix(A) );
	}

	// getters:

	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, hyperdags, Coords > &x ) {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations;
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::SIZE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return size (internal::getVector(x));
	}

	template< typename InputType >
	size_t nrows( const Matrix< InputType, hyperdags > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::NROWS,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return nrows(internal::getMatrix(A));
	}

	template< typename InputType >
	size_t ncols( const Matrix< InputType, hyperdags > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::NCOLS,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return ncols(internal::getMatrix(A));
	}

	template< typename DataType, typename Coords >
	size_t capacity( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CAPACITY_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return capacity(internal::getVector( x ));
	}

	template< typename DataType >
	size_t capacity( const Matrix< DataType, hyperdags > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::CAPACITY_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return capacity(internal::getMatrix( A ));
	}

	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::NNZ_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return nnz( internal::getVector( x ) );
	}

	template< typename InputType >
	size_t nnz( const Matrix< InputType, hyperdags > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::NNZ_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return nnz(internal::getMatrix(A));
	}

	template< typename InputType, typename Coords >
	uintptr_t getID( const Vector< InputType, hyperdags, Coords > &x ) {
		std::array< const void *, 1 > sources{ &x };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::GETID_VECTOR,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return getID(internal::getVector( x ));
	}

	template< typename InputType >
	uintptr_t getID( const Matrix< InputType, hyperdags > &A ) {
		std::array< const void *, 1 > sources{ &A };
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::GETID_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return getID(internal::getMatrix( A ));
	}

	// resizers:

	template< typename InputType, typename Coords >
	RC resize(
		Vector< InputType, hyperdags, Coords > &x,
		const size_t new_nz
	) noexcept {
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&new_nz
		);
		std::array< const void *, 2 > sources{ &x, &new_nz };
		std::array< const void *, 1 > destinations{ &x };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::RESIZE,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return resize(internal::getVector( x ), new_nz);
	}

	template< typename InputType >
	RC resize(
		Matrix< InputType, hyperdags > &A,
		const size_t new_nz
	) noexcept {
		internal::hyperdags::generator.addSource(
			internal::hyperdags::USER_INT,
			&new_nz
		);
		std::array< const void *, 2 > sources{ &A, &new_nz };
		std::array< const void *, 1 > destinations{ &A };
		internal::hyperdags::generator.addOperation(
			internal::hyperdags::RESIZE_MATRIX,
			sources.begin(), sources.end(),
			destinations.begin(), destinations.end()
		);
		return resize( internal::getMatrix(A), new_nz );
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
	template< typename InputType, typename... Args >
	RC wait(
		const Matrix< InputType, hyperdags > &A,
		const Args &... args
	) {
		(void) A;
		return wait( args... );
	}

} // namespace grb

