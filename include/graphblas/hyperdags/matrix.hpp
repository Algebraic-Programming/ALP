
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
 * @date 3rd of March, 2022
 */

#ifndef _H_GRB_HYPERDAGS_MATRIX
#define _H_GRB_HYPERDAGS_MATRIX

#include <graphblas/config.hpp>


namespace grb {

	namespace internal {

		template< typename T >
		Matrix< T, _GRB_WITH_HYPERDAGS_USING > & getMatrix(
			Matrix< T, grb::hyperdags > &
		);

		template< typename T >
		const Matrix< T, _GRB_WITH_HYPERDAGS_USING> & getMatrix(
			const Matrix< T, grb::hyperdags > &x
		);
		

		template< typename T >
		inline internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( Matrix< T, grb::hyperdags > & A ) noexcept;

		template< typename T >
		inline const internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( const Matrix< T, grb::hyperdags > & A ) noexcept;

		template< typename T >
		inline internal::Compressed_Storage< T, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( Matrix< T, grb::hyperdags > & A ) noexcept;

		template< typename T >
		inline const internal::Compressed_Storage< T, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( const Matrix< T, grb::hyperdags > & A ) noexcept;


	}

	template< typename T >
	class Matrix< T, hyperdags > {


		template< typename A >
		friend Matrix< A, _GRB_WITH_HYPERDAGS_USING > & internal::getMatrix(
			Matrix< A, grb::hyperdags > &
		);

		template< typename A >
		friend const Matrix< A, _GRB_WITH_HYPERDAGS_USING > & internal::getMatrix(
			const Matrix< A, grb::hyperdags > &
		);


		private:

			/** \internal Simply use an underlying implementation */
			typedef Matrix< T, grb::_GRB_WITH_HYPERDAGS_USING > MyMatrixType;

			MyMatrixType matrix;


		public:
			Matrix( const size_t rows, const size_t columns ) : matrix( rows, columns ) {}
			Matrix( const size_t rows, const size_t columns, const size_t nz ) : matrix(rows, columns, nz) {}
			
			template< class ActiveDistribution = internal::Distribution< grb::_GRB_WITH_HYPERDAGS_USING > >
			typename internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
			begin( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
				return matrix.begin(mode, s, P);
			}

			template< class ActiveDistribution = internal::Distribution< grb::_GRB_WITH_HYPERDAGS_USING > >
			typename internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
			end( const IOMode mode = PARALLEL, const size_t s = 0, const size_t P = 1 ) const {
				return matrix.end(mode, s, P);
			}

			template< class ActiveDistribution = internal::Distribution< grb::_GRB_WITH_HYPERDAGS_USING > >
			typename internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
			cbegin( const IOMode mode = PARALLEL ) const {
				return matrix.cbegin(mode);
			}

			template< class ActiveDistribution = internal::Distribution< grb::_GRB_WITH_HYPERDAGS_USING > >
			typename internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType >::template ConstIterator< ActiveDistribution >
			cend( const IOMode mode = PARALLEL ) const {
				return matrix.cend(mode);
			}

	};

	template< typename D >
	struct is_container< Matrix< D, hyperdags > > {
		/** A hyperdags matrix is an ALP container. */
		static const constexpr bool value = true;
	};

	namespace internal {

		template< typename T >
		Matrix< T, _GRB_WITH_HYPERDAGS_USING > & getMatrix(
			Matrix< T, grb::hyperdags > &x
		) {
			return x.matrix;
		}

		template< typename T >
		const Matrix< T, _GRB_WITH_HYPERDAGS_USING > & getMatrix(
			const Matrix< T, grb::hyperdags > &x
		) {
			return x.matrix;
		}
		
		
		template< typename T >
		inline internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( Matrix< T, grb::hyperdags > & A ) noexcept {
			return getCRS( internal::getMatrix(A) );
		}

		template< typename T >
		inline const internal::Compressed_Storage< T, grb::config::RowIndexType, grb::config::NonzeroIndexType > & getCRS( const Matrix< T, grb::hyperdags > & A ) noexcept {
			return getCRS( internal::getMatrix(A) );
		}

		template< typename T >
		inline internal::Compressed_Storage< T, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( Matrix< T, grb::hyperdags > & A ) noexcept {
			return getCCS( internal::getMatrix(A) );
		}

		template< typename T >
		inline const internal::Compressed_Storage< T, grb::config::ColIndexType, grb::config::NonzeroIndexType > & getCCS( const Matrix< T, grb::hyperdags > & A ) noexcept {
			return getCCS( internal::getMatrix(A) );
		}

	}

}

#endif

