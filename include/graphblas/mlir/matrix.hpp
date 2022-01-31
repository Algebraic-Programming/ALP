
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
 * @author D. G. Spampinato
 * @date 16th of November, 2021
 */

#if ! defined _H_GRB_MLIR_MATRIX
#define _H_GRB_MLIR_MATRIX

#ifdef _DEBUG
#include <cstdio>
#endif

#include <algorithm> //std::copy
#include <numeric>   //std::accumulate
#include <sstream>   //std::stringstream
#include <type_traits>
#include <vector> //std::vector

#include <assert.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/iomode.hpp>
#include <graphblas/mlir/jitCtx.hpp>

namespace grb {

	namespace internal {

		template< typename D >
		class SizeOf {
		public:
			static constexpr size_t value = sizeof( D );
		};

		template<>
		class SizeOf< void > {
		public:
			static constexpr size_t value = 0;
		};

		/// Copy the memref to a std::vector.
		template< typename D >
		inline std::vector< D > getFull( Matrix< D, mlir > & A ) {
			grb::jit::JitContext & jitCtx = grb::jit::JitContext::getCurrentJitContext();
			jitCtx.buildAndExecute();
			std::vector< D > result;
			for( size_t i = 0; i < A.m; i++ )
				for( size_t j = 0; j < A.n; j++ )
					result.push_back( ( *A.storage )[ i ][ j ] );
			return result;
		}

		template< typename D >
		inline const std::vector< D > getFull( const Matrix< D, mlir > & A ) {
			grb::jit::JitContext & jitCtx = grb::jit::JitContext::getCurrentJitContext();
			jitCtx.buildAndExecute();
			std::vector< D > result;
			for( size_t i = 0; i < A.m; i++ )
				for( size_t j = 0; j < A.n; j++ )
					result.push_back( ( *A.storage )[ i ][ j ] );
			return result;
		}

	} // namespace internal

	template< typename InputType >
	RC clear( Matrix< InputType, mlir > & A ) noexcept {
		return A.clear();
	}

	template< typename DataType >
	size_t nrows( const Matrix< DataType, mlir > & A ) noexcept {
		return A.m;
	}

	template< typename DataType >
	size_t ncols( const Matrix< DataType, mlir > & A ) noexcept {
		return A.n;
	}

	/**
	 * A GraphBLAS matrix, mlir implementation.
	 *
	 * @tparam D  The element type.
	 */
	template< typename D >
	class Matrix< D, mlir > {

		static_assert( ! grb::is_object< D >::value, "Cannot create a GraphBLAS matrix of GraphBLAS objects!" );

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, mlir > & ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, mlir > & ) noexcept;

		template< typename InputType >
		friend RC clear( Matrix< InputType, mlir > & A ) noexcept;

		template< Descriptor descr, typename InputType, typename fwd_iterator >
		friend RC buildMatrixUnique( Matrix< InputType, mlir > &, fwd_iterator, const fwd_iterator, const IOMode );

		friend std::vector< D > internal::getFull<>( Matrix< D, mlir > & A );
		friend const std::vector< D > internal::getFull<>( const Matrix< D, mlir > & A );

	public:
		// Our own type.
		typedef Matrix< D, mlir > self_type;

		// The number of rows.
		const size_t m;

		// The number of columns.
		const size_t n;

		// The full storage.
		mlir::OwningMemRef< D, 2 > storage;

		RC clear() {
			// catch trivial case
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}

			// catch uninitialised case
			if( storage.empty() ) {
				// nothing to do
				return SUCCESS;
			}

			m = n = 0;
			storage.clear();

			return SUCCESS;
		}

		template< Descriptor descr = descriptors::no_operation, typename fwd_iterator >
		RC buildMatrixUnique( const fwd_iterator & _start, const fwd_iterator & _end ) {

			// detect trivial case
			if( _start == _end || m == 0 || n == 0 ) {
				return SUCCESS;
			}

			if( std::distance( _start, _end ) != m * n ) {
				return MISMATCH;
			}

#ifdef _DEBUG
			for( auto it = _start; it != _end; ++it ) {
				std::cout << *it << " ";
			}
#endif

			auto it = _start;
			// std::copy(_start, _end, full.begin());
			for( size_t i = 0; i < m; i++ )
				for( size_t j = 0; j < n; j++ ) {
					( *storage )[ i ][ j ] = *it;
					it++;
				}

#ifdef _DEBUG
			for( size_t i = 0; i < m; i++ ) {
				for( size_t j = 0; j < n; j++ ) {
					std::cout << ( *storage )[ i ][ j ] << " ";
				}
				std::cout << "\n";
			}
#endif

			// done
			return SUCCESS;
		}

	public:
		typedef D value_type;

		Matrix( const size_t rows, const size_t columns ) : m( rows ), n( columns ), storage( { rows, columns }, {} ) {
			if( rows >= static_cast< size_t >( std::numeric_limits< grb::config::RowIndexType >::max() ) ) {
				throw std::overflow_error( "Number of rows larger than "
										   "configured RowIndexType maximum!" );
			}
			if( columns >= static_cast< size_t >( std::numeric_limits< grb::config::ColIndexType >::max() ) ) {
				throw std::overflow_error( "Number of columns larger than "
										   "configured ColIndexType maximum!" );
			}
		}

		template< typename InputType >
		Matrix( const self_type & other ) : m( other.m ), n( other.n ), storage( std::move( other.storage ) ) {}

		Matrix( self_type && other ) : m( other.m ), n( other.n ), storage( std::move( other.storage ) ) {
			other.m = 0;
			other.n = 0;
		}

		~Matrix() {}
	};

	// template specialisation for GraphBLAS type traits
	template< typename D >
	struct is_container< Matrix< D, mlir > > {
		// A mlir Matrix is a GraphBLAS object.
		static const constexpr bool value = true;
	};

} // namespace grb

#endif // end ``_H_GRB_MLIR_MATRIX''
