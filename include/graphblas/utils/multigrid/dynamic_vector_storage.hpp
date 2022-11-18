
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
 * @file dynamic_vector_storage.cpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Extension of a heap-allocated array exposing the underlying storage and iterators.
 */

#ifndef _H_GRB_ALGORITHMS_MULTIGRID_DYNAMIC_VECTOR_STORAGE
#define _H_GRB_ALGORITHMS_MULTIGRID_DYNAMIC_VECTOR_STORAGE

#include <cstddef>
#include <cstddef>
#include <algorithm>

namespace grb {
	namespace utils {
		namespace multigrid {

			/**
			 * Array with fixed size (i.e. decided at object creation) allocated on the heap with an interface compliant
			 * to what other classes in the geometry namespace expect, like storage() and dimensions() methods.
			 *
			 * It describes a vector of dimensions #dimensions().
			 *
			 * @tparam DataType the data type of the vector elements
			 */
			template< typename DataType > class DynamicVectorStorage {

				size_t _dimensions;
				DataType* _storage;

				void clean() {
					if( this->_storage != nullptr ) {
						delete[] this->_storage;
					}
				}

			public:
				// iterator fields
				using reference = DataType&;
				using const_reference = const DataType&;
				using iterator = DataType*;
				using const_iterator = const DataType*;
				using pointer = DataType*;
				using const_pointer = const DataType*;

				using VectorStorageType = DataType*;
				using ConstVectorStorageType = DataType*;
				using SelfType = DynamicVectorStorage< DataType >;

				DynamicVectorStorage( size_t __dimensions ):
					_dimensions( __dimensions ) {
					if( __dimensions == 0 ) {
						throw std::invalid_argument("dimensions cannot be 0");
					}
					this->_storage = new DataType[ __dimensions ];
				}

				DynamicVectorStorage() = delete;

				DynamicVectorStorage( const SelfType &o ):
					_dimensions( o._dimensions ),
					_storage( new DataType[ o._dimensions ] )
				{
					std::copy_n( o._storage, o._dimensions, this->_storage );
				}

				DynamicVectorStorage( SelfType &&o ) = delete;

				SelfType& operator=( const SelfType &original ) {
					if( original._dimensions != this->_dimensions ) {
						this->clean();
						this->_storage = new DataType[ original._dimensions];
					}
					this->_dimensions = original._dimensions;
					std::copy_n( original._storage, original._dimensions, this->_storage );
					return *this;
				}

				SelfType& operator=( SelfType &&original ) = delete;

				~DynamicVectorStorage() {
					this->clean();
				}

				size_t dimensions() const {
					return this->_dimensions;
				}

				inline iterator begin() {
					return this->_storage;
				}

				inline iterator end() {
					return this->_storage + this->_dimensions;
				}

				inline const_iterator begin() const {
					return this->_storage;
				}

				inline const_iterator end() const {
					return this->_storage + this->_dimensions;
				}

				inline const_iterator cbegin() const {
					return this->_storage;
				}

				inline const_iterator cend() const {
					return this->_storage + this->_dimensions;
				}

				inline VectorStorageType storage() {
					return this->_storage;
				}

				inline ConstVectorStorageType storage() const {
					return this->_storage;
				}

				inline reference operator[]( size_t pos ) {
					return *( this->_storage + pos);
				}

				inline const_reference operator[]( size_t pos ) const {
					return *( this->_storage + pos );
				}
			};

		} // namespace multigrid
	} // namespace utils
} // namespace grb

#endif // _H_GRB_ALGORITHMS_MULTIGRID_DYNAMIC_VECTOR_STORAGE
