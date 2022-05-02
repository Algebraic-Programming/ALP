
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

#include "nist.h"

#include <assert.h>

#include <graphblas.hpp>


/** \internal Internal namespace for the SparseBLAS implementation. */
namespace sparseblas {

	/** \internal Number of insertions in a single batch */
	constexpr const size_t BATCH_SIZE = 1000;
	static_assert( BATCH_SIZE > 0, "BATCH_SIZE must be positive" );

	/** \internal A single triplet for insertion */
	template< typename T >
	class Triplet {
		public:
			int row, col;
			T val;
	};

	template< typename T >
	class PartialTripletBatch {
		public:
			int ntriplets;
			Triplet< T > triplets[ BATCH_SIZE ];
			PartialTripletBatch() : ntriplets( 0 ) {}
	};

	template< typename T >
	class FullTripletBatch {
		public:
			Triplet< T > triplets[ BATCH_SIZE ];
			FullTripletBatch( PartialTripletBatch< T > &&other ) {
				assert( other.ntriplets == BATCH_SIZE );
				std::swap( triplets, other.triplets );
				other.ntriplets = 0;
			}
	};

	/** \internal A sparse matrix under construction. */
	template< typename T >
	class MatrixUC {

		private:

			std::vector< FullTripletBatch< T > > batches;
			PartialTripletBatch< T > last;

		public:

			class const_iterator {

				friend class MatrixUC< T >;

				private:

					const std::vector< FullTripletBatch< T > > &batches;
					const PartialTripletBatch< T > &last;
					size_t batch;
					size_t loc;


				protected:

					const_iterator( const MatrixUC< T > &x ) :
						batches( x.batches ), last( x.last ),
						batch( 0 ), loc( 0 )
					{}

					void setToEndPosition() {
						batch = batches.size();
						loc = last.ntriplets;
					}


				public:

					typedef Triplet< T > value_type;
					typedef value_type & reference;
					typedef value_type * pointer;
					typedef std::forward_iterator_tag iterator_category;
					typedef int row_coordinate_type;
					typedef int column_coordinate_type;
					typedef T nonzero_value_type;

					const_iterator( const const_iterator &other ) :
						batches( other.batches ), last( other.last ),
						batch( other.batch ), loc( other.loc )
					{}

					const_iterator( const_iterator &&other ) :
						batches( other.batches ), last( other.last ),
						batch( other.batch ), loc( other.loc )
					{
						other.batch = other.loc = 0;
					}

					bool operator==( const const_iterator &other ) const {
						return batch == other.batch && loc == other.loc;
					}

					bool operator!=( const const_iterator &other ) const {
						return !( operator==( other ) );
					}

					const_iterator operator++() {
						assert( batch <= batches.size() );
						if( batch == batches.size() ) {
							assert( loc < last.ntriplets );
							(void) ++loc;
						} else {
							(void) ++loc;
							assert( loc <= BATCH_SIZE );
							if( loc == BATCH_SIZE ) {
								(void) ++batch;
								assert( batch <= batches.size() );
							}
						}
						return *this;
					}

					const Triplet< T > & operator*() const {
						assert( batch <= batches.size() );
						if( batch == batches.size () ) {
							assert( loc < last.ntriplets );
							return last.triplets[ loc ];
						} else {
							assert( loc < BATCH_SIZE );
							return batches[ batch ].triplets[ loc ];
						}
					}

					const Triplet< T > * operator->() const {
						return &( operator*() );
					}

					int i() const {
						const Triplet< T > &triplet = this->operator*();
						return triplet.row;
					}

					int j() const {
						const Triplet< T > &triplet = this->operator*();
						return triplet.col;
					}

					double v() const {
						const Triplet< T > &triplet = this->operator*();
						return triplet.val;
					}

			};

			void add( const T& val, const int &row, const int &col ) {
				assert( last.ntriplets != BATCH_SIZE );
				auto &triplet = last.triplets[ last.ntriplets ];
				triplet.row = row;
				triplet.col = col;
				triplet.val = val;
				(void) ++( last.ntriplets );
				if( last.ntriplets == BATCH_SIZE ) {
					batches.push_back( std::move( last ) );
				}
				last = PartialTripletBatch< T >();
			}

			size_t nnz() const {
				size_t ret = batches.size() * BATCH_SIZE;
				ret += last.ntriplets;
				return ret;
			}

			const_iterator cbegin() const {
				return const_iterator( *this );
			}

			const_iterator cend() const {
				auto ret = const_iterator( *this );
				ret.setToEndPosition();
				return ret;
			}

	};

	template< typename T >
	class SparseMatrix {

		public:

			int m, n;

			bool finalized;

			MatrixUC< T > * ingest;

			grb::Matrix< T > * A;

			SparseMatrix( const int _m, const int _n ) :
				m( _m ), n( _n ),
				finalized( false ), A( nullptr )
			{
				ingest = new MatrixUC< T >();
			}

			SparseMatrix( grb::Matrix< T > &X ) :
				m( grb::nrows( X ) ), n( grb::ncols( X ) ),
				finalized( true ), ingest( nullptr ), A( &X )
			{}

			~SparseMatrix() {
				m = n = 0;
				if( ingest != nullptr ) {
					assert( !finalized );
					delete ingest;
					ingest = nullptr;
				} else {
					assert( A != nullptr );
					assert( finalized );
					delete A;
					A = nullptr;
				}
				finalized = false;
			}

			void finalize() {
				assert( !finalized );
				assert( A == nullptr );
				assert( ingest != nullptr );
				A = new grb::Matrix< T >( m, n, ingest->nnz() );
				const auto rc = grb::buildMatrixUnique(
					*A, ingest->cbegin(), ingest->cend(), grb::SEQUENTIAL );
				if( rc != grb::SUCCESS ) {
					throw std::runtime_error( "Could not ingest matrix into ALP/GraphBLAS"
						"during finalisation." );
				}
				delete ingest;
				ingest = nullptr;
				finalized = true;
			}

	};

	SparseMatrix< double > * getDoubleMatrix( blas_sparse_matrix A ) {
		return static_cast< SparseMatrix< double >* >( A );
	}

}

extern "C" {

	blas_sparse_matrix BLAS_duscr_begin( const int m, const int n ) {
		return new sparseblas::SparseMatrix< double >( m, n );
	}

	int BLAS_duscr_insert_entry(
		blas_sparse_matrix A,
		const double val, const int row, const int col
	) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized == false );
		assert( matrix->ingest != nullptr );
		try {
			matrix->ingest->add( val, row, col );
		} catch( ... ) {
			return 255;
		}
		return 0;
	}

	int BLAS_duscr_insert_entries(
		blas_sparse_matrix A,
		const int nnz,
		const double * vals, const int * rows, const int * cols
	) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized == false );
		assert( matrix->ingest != nullptr );
		try {
			for( size_t k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], rows[ k ], cols[ k ] );
			}
		} catch( ... ) {
			return 255;
		}
		return 0;
	}

	int BLAS_duscr_insert_col(
		blas_sparse_matrix A,
		const int j, const int nnz,
		const double * vals, const int * rows
	) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized == false );
		assert( matrix->ingest != nullptr );
		try {
			for( size_t k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], rows[ k ], j );
			}
		} catch( ... ) {
			return 255;
		}
		return 0;
	}

	int BLAS_duscr_insert_row(
		blas_sparse_matrix A,
		const int i, const int nnz,
		const double * vals, const int * cols
	) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized == false );
		assert( matrix->ingest != nullptr );
		try {
			for( size_t k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], i, cols[ k ] );
			}
		} catch( ... ) {
			return 255;
		}
		return 0;
	}

	int BLAS_duscr_end( blas_sparse_matrix A ) {
		try {
			sparseblas::getDoubleMatrix( A )->finalize();
		} catch( ... ) {
			return 255;
		}
		return 0;
	}

	int BLAS_usds( blas_sparse_matrix A ) {
		delete sparseblas::getDoubleMatrix( A );
		return 0;
	}

	// TODO implement additional functionality from here onwards

}

