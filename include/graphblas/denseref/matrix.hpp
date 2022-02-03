
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
 * @date 14th of January 2022
 */

#ifndef _H_GRB_DENSEREF_MATRIX
#define _H_GRB_DENSEREF_MATRIX

#include <graphblas/backends.hpp>
#include <graphblas/base/matrix.hpp>
#include <graphblas/config.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
//#include <graphblas/reference/blas1.hpp>
//#include <graphblas/reference/compressed_storage.hpp>
//#include <graphblas/reference/init.hpp>
#include <graphblas/type_traits.hpp>
#include <graphblas/utils/autodeleter.hpp>
//#include <graphblas/utils/pattern.hpp> //for help with dealing with pattern matrix input

namespace grb {


	template< typename T >
	T * getRaw( Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	const T * getRaw( const Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	size_t nrows( const Matrix< T, reference_dense > & ) noexcept;

	template< typename T >
	size_t ncols( const Matrix< T, reference_dense > & ) noexcept;

	/** \internal TODO */
	template< typename T >
	class Matrix< T, reference_dense > {

		/* *********************
		        BLAS2 friends
		   ********************* */

		template< typename DataType >
		friend size_t nrows( const Matrix< DataType, reference_dense > & m ) noexcept;

		template< typename DataType >
		friend size_t ncols( const Matrix< DataType, reference_dense > & m ) noexcept;

		/* *********************
		     `Getter' friends
		   ********************* */

		friend T * getRaw<T>( Matrix< T, reference_dense > &) noexcept;

		friend const T * getRaw<T>( const Matrix< T, reference_dense > & ) noexcept;

	private:
		/** Our own type. */
		typedef Matrix< T, reference_dense > self_type;

		/**
		* The number of rows.
		*
		* \internal Not declared const to be able to implement move in an elegant way.
		*/
		size_t m;

		/**
		* The number of columns.
		*
		* \internal Not declared const to be able to implement move in an elegant way.
		*/
		size_t n;

		/** The matrix data. */
		T *__restrict__ data;

		/** Whether the container presently is uninitialized. */
		bool initialized;

	public:
		/** @see Matrix::value_type */
		typedef T value_type;

		/** @see Matrix::Matrix() */
		Matrix( const size_t rows, const size_t columns ): m( rows ), n( columns ), initialized( false ) {

		}

		/** @see Matrix::Matrix( const Matrix & ) */
		Matrix( const Matrix< T, reference_dense > &other ) : Matrix( other.m, other.n ) {
			initialized = other.initialized;
		}

		/** @see Matrix::Matrix( Matrix&& ). */
		Matrix( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
		}

		/** * Move from temporary. */
		self_type& operator=( self_type &&other ) noexcept {
			moveFromOther( std::forward< self_type >(other) );
			return *this;
		}

		/** @see Matrix::~Matrix(). */
		~Matrix() {

		}

	};

	// template specialisation for GraphBLAS type traits
	template< typename T >
	struct is_container< Matrix< T, reference_dense > > {
		/** A reference_dense Matrix is a GraphBLAS object. */
		static const constexpr bool value = true;
	};

	template< typename T >
	T * getRaw( Matrix< T, reference_dense > &m ) noexcept {
		return m.data;
	}

	template< typename T >
	const T * getRaw( const Matrix< T, reference_dense > &m ) noexcept {
		return m.data;
	}

	template< typename T >
	size_t nrows( const Matrix< T, reference_dense > &m ) noexcept {
		return m.m;
	}

	template< typename T >
	size_t ncols( const Matrix< T, reference_dense > &m ) noexcept {
		return m.n;
	}

} // end namespace ``grb''

#endif // end ``_H_GRB_DENSEREF_MATRIX''

