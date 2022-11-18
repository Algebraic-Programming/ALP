
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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
 * @file ndim_vector.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Definition of NDimVector.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_NDIM_VECTOR
#define _H_GRB_ALGORITHMS_MULTIGRID_NDIM_VECTOR

#include <utility>
#include <vector>
#include <type_traits>
#include <cstddef>
#include <algorithm>

#include "linearized_ndim_system.hpp"

namespace grb {
	namespace utils {
		namespace multigrid {

			/**
			 * Maps an N-dimensional vector to an array of data.
			 *
			 * The user constructs an object by passing the sizes (as an N-dimensional vector)
			 * of the iteration space and accesses the stored data via an N-dimensional vector of coordinates.
			 *
			 * Example: if the user constructs an NDimVector with 3D sizes \a [2,3,4], she can access data
			 * via a 3D coordinates vector of ranges \a [0-1]x[0-2]x[0-3] (here \a x denoting the cartesian product)
			 * by using the #at() method.
			 *
			 * This facility allows associating a value of type \p DataType to, for example,
			 * each element of an N-dimensional grid.
			 *
			 * @tparam DataType type of data stored in the array
			 * @tparam SizeType type for the components of the N-dimensional vector:
			 * 	the maximum number of stored data is thus \f$ std::numeric_limits<SizeType>::max()^N \f$
			 * @tparam InternalVectorType storage type of the internal N-dimensional vector
			 */
			template<
				typename DataType,
				typename SizeType,
				typename InternalVectorType
			> class NDimVector {
			public:
				static_assert( std::is_default_constructible< DataType >::value,
					"the stored type is not default constructible" );
				static_assert( std::is_integral< SizeType >::value, "SizeType must be integral" );

				using ConstDomainVectorReference =
					typename LinearizedNDimSystem< SizeType, InternalVectorType >::ConstVectorReference;
				using ConstDomainVectorStorageType = typename InternalVectorType::ConstVectorStorageType;
				using DomainIterator = typename LinearizedNDimSystem< SizeType, InternalVectorType >::Iterator;
				using Selftype = NDimVector< DataType, SizeType, InternalVectorType >;

				NDimVector() = delete;

				/**
				 * Construct a new NDimVector object with sizes read from the iteration range
				 * and number of dimensions equal to the range distance; the data values are
				 * \b not initialized.
				 */
				template< typename IterT > NDimVector( IterT begin, IterT end) :
					_linearizer( begin, end )
				{
					this->data = new DataType[ _linearizer.system_size() ];
				}

				/**
				 * Construct a new NDimVector object with sizes read from the \p _sizes
				 * and number of dimensions equal to \p _sizes.size(); the data values are
				 * \b not initialized.
				 */
				NDimVector( const std::vector< size_t > &_sizes ) :
					NDimVector( _sizes.cbegin(), _sizes.cend() ) {}

				NDimVector( const Selftype& original ):
					_linearizer( original._linearizer ),
				    data( new DataType[ original.data_size() ] )
				{
					std::copy_n( original.data, original.data_size(), this->data );
				}

				NDimVector( Selftype&& original ) noexcept:
					_linearizer( std::move( original._linearizer ) )
				{
					this->data = original.data;
					original.data = nullptr;
				}

				Selftype& operator=( const Selftype &original ) = delete;

				Selftype& operator=( Selftype &&original ) = delete;

				~NDimVector() {
					this->clean_mem();
				}

				/**
				 * Number of dimensions of the underlying geometrical space.
				 */
				size_t dimensions() const {
					return this->_linearizer.dimensions();
				}

				/**
				 * Size of the the underlying geometrical space, i.e. number of stored data elements.
				 */
				size_t data_size() const {
					return this->_linearizer.system_size();
				}

				/**
				 * Access the data element at N-dimension coordinate given by the iterable
				 * \p coordinates.
				 */
				inline DataType& at( ConstDomainVectorReference coordinates ) {
					return this->data[ this->get_coordinate( coordinates.storage() ) ];
				}

				/**
				 * Const-access the data element at N-dimension coordinate given by the iterable
				 * \p coordinates.
				 */
				inline const DataType& at( ConstDomainVectorReference coordinates ) const {
					return this->data[ this->get_coordinate( coordinates.storage() ) ];
				}

				/**
				 * Access the data element at N-dimension coordinate given by the vector
				 * storage object \p coordinates.
				 */
				inline DataType& at( ConstDomainVectorStorageType coordinates ) {
					return this->data[ this->get_coordinate( coordinates ) ];
				}

				/**
				 * Const-access the data element at N-dimension coordinate given by the vector
				 * storage object \p coordinates.
				 */
				inline const DataType& at( ConstDomainVectorStorageType coordinates ) const {
					return this->data[ this->get_coordinate( coordinates ) ];
				}

				/**
				 * Returns an iterator to the beginning of the N-dimensional underlyign space,
				 * i.e. a vector \a [0,0,0,...,0].
				 */
				DomainIterator domain_begin() const {
					return this->_linearizer.begin();
				}

				/**
				 * Returns an iterator to the end of the N-dimensional underlyign space.
				 * This iterator should not be referenced nor incremented.
				 */
				DomainIterator domain_end() const {
					return this->_linearizer.end();
				}

			private:
				const LinearizedNDimSystem< SizeType, InternalVectorType > _linearizer;
				DataType* data;

				inline size_t get_coordinate( ConstDomainVectorStorageType coordinates ) const {
					return this->_linearizer.ndim_to_linear( coordinates );
				}

				inline size_t get_coordinate( DomainIterator coordinates ) const {
					return this->_linearizer.ndim_to_linear( coordinates );
				}

				void clean_mem() {
					if ( this->data == nullptr ) {
						delete[] this->data;
					}
				}
			};

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_NDIM_VECTOR
