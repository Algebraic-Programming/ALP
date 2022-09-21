
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

	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator
	>
	RC buildMatrixUnique( Matrix< InputType, hyperdags > &A,
		fwd_iterator start, 
		const fwd_iterator end,
		const IOMode mode
	) {
		std::array< const void *, 1 > sources{ &A};
		std::array< const void *, 0 > destinations{  };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::BUILDMATRIXUNIQUE_MATRIX_START_END_MODE,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return buildMatrixUnique<descr>( internal::getMatrix(A), start, end, mode );
	}
	
	template< 
		typename DataType, typename Coords 
	>
	size_t capacity( const Vector< DataType, hyperdags, Coords > &x ) noexcept {
	
		std::array< const void *, 1 > sources{ &x};
		std::array< const void *, 0 > destinations{  };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::CAPACITY_VECTOR,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return capacity(internal::getVector( x ));
	}

	template< 
		typename DataType 
	>
	size_t capacity( const Matrix< DataType, hyperdags > &A ) noexcept {
		std::array< const void *, 1 > sources{ &A};
		std::array< const void *, 0 > destinations{  };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::CAPACITY_MATRIX,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return capacity(internal::getMatrix( A ));
	}
	
	template< 
		typename InputType, typename Coords 
	>
	RC resize( Vector< InputType, hyperdags, Coords > &x, 
		const size_t new_nz ) noexcept {
		
		std::array< const void *, 1 > sources{ &x};
		std::array< const void *, 0 > destinations{  };
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::RESIZE,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return resize(internal::getVector( x ), new_nz);
	}


	template< 
		typename InputType, typename Coords 
	>
	uintptr_t getID( const Vector< InputType, hyperdags, Coords > &x )
	{
	
		std::array< const void *, 1 > sources{ &x};
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::GETID_VECTOR,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
		return getID(internal::getVector( x ));
	}
	
	template< 
		typename InputType
	>
	uintptr_t getID( const Matrix< InputType, hyperdags > &A ) {
		std::array< const void *, 1 > sources{ &A};
		std::array< const void *, 0 > destinations{};
		internal::hyperdags::generator.addOperation(
				internal::hyperdags::GETID_MATRIX,
				sources.begin(), sources.end(),
				destinations.begin(), destinations.end()
		);
	
		return getID(internal::getMatrix( A ));
	}

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

