
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

#ifndef _H_ALP_AMF_BASED_FUNCTORBASEDMATRIX
#define _H_ALP_AMF_BASED_FUNCTORBASEDMATRIX

#include <functional>

#include <alp/backends.hpp>
#include <alp/base/matrix.hpp>
#include <alp/config.hpp>
#include <alp/type_traits.hpp>
#include <alp/utils.hpp>
#include <alp/imf.hpp>

namespace alp {

	namespace internal {

		/** Forward declaration */
		template< typename DerivedMatrix >
		class MatrixBase;

		/** Forward declaration */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix;

		/** Functor reference getter used by friend functions of specialized Matrix */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		const typename FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType >::functor_type &getFunctor( const FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > &A );

		/**
		 * Getter for the functor of a functor-based matrix.
		 *
		 * @tparam MatrixType  The type of input matrix.
		 *
		 * @param[in] A        Input matrix.
		 *
		 * @returns A constant reference to a functor object within the
		 *          provided functor-based matrix.
		 */
		template<
			typename MatrixType,
			std::enable_if_t<
				internal::is_functor_based< MatrixType >::value
			> * = nullptr
		>
		const typename MatrixType::functor_type &getFunctor( const MatrixType &A ) {
			return static_cast< const typename MatrixType::base_type & >( A ).getFunctor();
		}

		/**
		 * Specialization of MatrixReference with a lambda function as a target.
		 * Used as a result of low-rank operation to avoid the need for allocating a container.
		 * The data is produced lazily by invoking the lambda function stored as a part of this object.
		 *
		 * \note Views-over-lambda-functions types are used internally as results of low-rank operations and are not
		 *       directly exposed to users. From the users perspective, the use of objects of this type does not differ
		 *       from the use of other \a alp::Matrix types. The difference lies in a lazy implementation of the access
		 *       to matrix elements, which is not exposed to the user.
		 *
		 */
		template< typename T, typename ImfR, typename ImfC, typename DataLambdaType >
		class FunctorBasedMatrix : public MatrixBase< FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > > {
			public:

				/** Expose static properties */
				typedef T value_type;
				/** Type returned by access function */
				typedef T access_type;
				/** Type of the index used to access the physical storage */
				typedef std::pair< size_t, size_t > storage_index_type;

			protected:

				typedef FunctorBasedMatrix< T, ImfR, ImfC, DataLambdaType > self_type;
				friend MatrixBase< self_type >;

				typedef std::function< bool() > initialized_functor_type;
				const initialized_functor_type initialized_lambda;

				const ImfR imf_r;
				const ImfC imf_c;

				const DataLambdaType data_lambda;

				std::pair< size_t, size_t > dims() const noexcept {
					return std::make_pair( imf_r.n, imf_c.n );
				}

				const DataLambdaType &getFunctor() const noexcept {
					return data_lambda;
				}

				bool getInitialized() const noexcept {
					return initialized_lambda();
				}

				void setInitialized( const bool ) noexcept {
					static_assert( "Calling setInitialized on a FunctorBasedMatrix is not allowed." );
				}

				access_type access( const storage_index_type &storage_index ) const {
					T result = 0;
					data_lambda( result, imf_r.map( storage_index.first ), imf_c.map( storage_index.second ) );
					return static_cast< access_type >( result );
				}

				storage_index_type getStorageIndex( const size_t i, const size_t j, const size_t s, const size_t P ) const {
					(void)s;
					(void)P;
					return std::make_pair( i, j );
				}

			public:

				FunctorBasedMatrix(
					initialized_functor_type initialized_lambda,
					ImfR imf_r,
					ImfC imf_c,
					const DataLambdaType data_lambda
				) :
					initialized_lambda( initialized_lambda ),
					imf_r( imf_r ),
					imf_c( imf_c ),
					data_lambda( data_lambda ) {}

		}; // class FunctorBasedMatrix

	} // namespace internal

} // namespace alp

#endif // end ``_H_ALP_AMF_BASED_FUNCTORBASEDMATRIX''
