
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
 * Provides internal implementation details for a transition-path sparse vector.
 *
 * This file was split from the preceding <tt>sparseblas.cpp</tt>, which
 * previously implemented both the SpBLAS interface as well as the SparseBLAS
 * one. Since both interfaces define extended functions that rely on the same
 * sparse vector extension, both library implementations must share a piece of
 * the same internals.
 *
 * @author A. N. Yzelman
 * @date 1/2/2024
 */


#ifndef _H_SPARSE_VECTOR_IMPL
#define _H_SPARSE_VECTOR_IMPL

#include <blas_sparse_vec.h>

#include <graphblas.hpp>

#include <assert.h>

#include <vector>


namespace native {

	/**
	 * A sparse vector that is either under construction or finalized as an
	 * ALP/GraphBLAS vector.
	 *
	 * This class simplifies I/O between ALP/GraphBLAS and native code. However,
	 * it should never be directly used -- it is instead intended to simplify the
	 * implementation of transition path libraries require such I/O.
	 */
	template< typename T >
	class SparseVector {

		public:

			int n;
			bool finalized;
			grb::Vector< T > * vector;
			typename grb::Vector< T >::const_iterator start, end;


		private:

			std::vector< T > uc_vals;
			std::vector< int > uc_inds;


		public:

			SparseVector( const int &_n ) :
				n( _n ), finalized( false ), vector( nullptr )
			{}

			~SparseVector() {
				if( finalized ) {
					assert( vector != nullptr );
					delete vector;
				} else {
					assert( vector == nullptr );
				}
			}

			void add( const T &val, const int &index ) {
				assert( !finalized );
				uc_vals.push_back( val );
				uc_inds.push_back( index );
			}

			void finalize() {
				assert( uc_vals.size() == uc_inds.size() );
				const size_t nz = uc_vals.size();
				vector = new grb::Vector< T >( n, nz );
				if( vector == nullptr ) {
					std::cerr << "Could not create ALP/GraphBLAS vector of size " << n
						<< " and capacity " << nz << "\n";
					throw std::runtime_error( "Could not create ALP/GraphBLAS vector" );
				}
				if( grb::capacity( *vector ) < nz ) {
					throw std::runtime_error( "ALP/GraphBLAS vector has insufficient "
						"capacity" );
				}
				const grb::RC rc = grb::buildVector(
					*vector,
					uc_inds.cbegin(), uc_inds.cend(),
					uc_vals.cbegin(), uc_vals.cend(),
					grb::SEQUENTIAL
				);
				if( rc != grb::SUCCESS ) {
					throw std::runtime_error( "Could not ingest nonzeroes into ALP/GraphBLAS "
						"vector" );
				}
				uc_vals.clear();
				uc_inds.clear();
				finalized = true;
			}

	};

}

#endif // end ifndef _H_SPARSE_VECTOR_IMPL

