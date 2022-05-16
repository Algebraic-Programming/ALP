
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

#include "blas_sparse.h"

#include <limits>
#include <vector>
#include <iostream>
#include <stdexcept>

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

	/**
	 * \internal A set of triplets, the number of which is than or equal to
	 *           #BATCH_SIZE triplets.
	 */
	template< typename T >
	class PartialTripletBatch {
		public:
			size_t ntriplets;
			Triplet< T > triplets[ BATCH_SIZE ];
			PartialTripletBatch() : ntriplets( 0 ) {}
	};

	/**
	 * \internal A set of #BATCH_SIZE triplets.
	 */
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

			/** \internal A series of full triplet batches. */
			std::vector< FullTripletBatch< T > > batches;

			/** \internal One partial batch of triplets. */
			PartialTripletBatch< T > last;


		public:

			/**
			 * \internal An iterator over the triplets contained herein. The iterator
			 *           adheres both to STL as well as ALP.
			 */
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
								loc = 0;
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

			/** \internal Adds a triplet. */
			void add( const T& val, const int &row, const int &col ) {
				assert( last.ntriplets != BATCH_SIZE );
				auto &triplet = last.triplets[ last.ntriplets ];
				triplet.row = row;
				triplet.col = col;
				triplet.val = val;
				(void) ++( last.ntriplets );
				if( last.ntriplets == BATCH_SIZE ) {
					FullTripletBatch< T > toAdd( std::move( last ) );
					batches.push_back( toAdd );
					assert( last.ntriplets == 0 );
				}
			}

			/** \internal Counts the number of triplets currently contained within. */
			size_t nnz() const {
				size_t ret = batches.size() * BATCH_SIZE;
				ret += last.ntriplets;
				return ret;
			}

			/** \internal Retrieves an iterator in start position. */
			const_iterator cbegin() const {
				return const_iterator( *this );
			}

			/** \internal Retrieves an iterator in end position. */
			const_iterator cend() const {
				auto ret = const_iterator( *this );
				ret.setToEndPosition();
				return ret;
			}

	};

	/**
	 * \internal A sparse vector that is either under construction, or finalized as
	 *           an ALP/GraphBLAS vector.
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

	/**
	 * \internal SparseBLAS allows a matrix to be under construction or finalized.
	 *           This class matches that concept -- for non-finalized matrices, it
	 *           is backed by MatrixUC, and otherwise by an ALP/GraphBLAS matrix.
	 */
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

			/**
			 * \internal Switches from a matrix under construction to a finalized
			 *           matrix.
			 */
			void finalize() {
				assert( !finalized );
				assert( A == nullptr );
				assert( ingest != nullptr );
				const size_t nnz = ingest->nnz();
				if( nnz > 0 ) {
					A = new grb::Matrix< T >( m, n, nnz );
					assert( grb::capacity( *A ) >= nnz );
					const grb::RC rc = grb::buildMatrixUnique(
						*A, ingest->cbegin(), ingest->cend(), grb::SEQUENTIAL );
					if( rc != grb::SUCCESS ) {
						throw std::runtime_error( "Could not ingest matrix into ALP/GraphBLAS"
							"during finalisation." );
					}
				} else {
					A = new grb::Matrix< T >( m, n );
				}
				delete ingest;
				ingest = nullptr;
				finalized = true;
			}

	};

	/**
	 * \internal Utility function that converts a #extblas_sparse_vector to a
	 *           sparseblas::SparseVector. This is for vectors of doubles.
	 */
	SparseVector< double > * getDoubleVector( extblas_sparse_vector x ) {
		return static_cast< SparseVector< double >* >( x );
	}

	/**
	 * \internal Utility function that converts a #blas_sparse_matrix to a
	 *           sparseblas::SparseMatrix. This is for matrices of doubles.
	 */
	SparseMatrix< double > * getDoubleMatrix( blas_sparse_matrix A ) {
		return static_cast< SparseMatrix< double >* >( A );
	}

}

// implementation of the SparseBLAS API follows

extern "C" {

	extblas_sparse_vector EXTBLAS_dusv_begin( const int n ) {
		return new sparseblas::SparseVector< double >( n );
	}

	int EXTBLAS_dusv_insert_entry(
		extblas_sparse_vector x,
		const double val,
		const int index
	) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( !(vector->finalized) );
		try {
			vector->add( val, index );
		} catch( ... ) {
			return 20;
		}
		return 0;
	}

	int EXTBLAS_dusv_end( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( !(vector->finalized) );
		try {
			vector->finalize();
		} catch( ... ) {
			return 30;
		}
		return 0;
	}

	int EXTBLAS_dusvds( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		delete vector;
		return 0;
	}

	int EXTBLAS_dusvnz( const extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		const size_t nnz = grb::nnz( *(vector->vector) );
		if( nnz > static_cast< size_t >( std::numeric_limits< int >::max() ) ) {
			std::cerr << "Number of nonzeroes is larger than what can be represented by "
				<< "a SparseBLAS int!\n";
			return 10;
		}
		return static_cast< int >(nnz);
	}

	int EXTBLAS_dusv_clear( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		const grb::RC rc = grb::clear( *(vector->vector) );
		if( rc != grb::SUCCESS ) {
			return 10;
		}
		return 0;
	}

	int EXTBLAS_dusv_open( const extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		try {
			vector->start = vector->vector->cbegin();
			vector->end = vector->vector->cend();
		} catch( ... ) {
			return 10;
		}
		return 0;
	}

	int EXTBLAS_dusv_get(
		const extblas_sparse_vector x,
		double * const val, int * const ind
	) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		assert( vector->start != vector->end );
		assert( val != nullptr );
		assert( ind != nullptr );
		*val = vector->start->second;
		*ind = vector->start->first;
		try {
			(void) ++(vector->start);
		} catch( ... ) {
			return 2;
		}
		if( vector->start == vector->end ) {
			return 0;
		} else {
			return 1;
		}
	}

	int EXTBLAS_dusv_close( const extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		vector->start = vector->end;
		return 0;
	}

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
			return 2;
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
			for( int k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], rows[ k ], cols[ k ] );
			}
		} catch( ... ) {
			return 3;
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
			for( int k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], rows[ k ], j );
			}
		} catch( ... ) {
			return 4;
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
			for( int k = 0; k < nnz; ++k ) {
				matrix->ingest->add( vals[ k ], i, cols[ k ] );
			}
		} catch( ... ) {
			return 5;
		}
		return 0;
	}

	int BLAS_duscr_end( blas_sparse_matrix A ) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized == false );
		assert( matrix->ingest != nullptr );
		try {
			matrix->finalize();
		} catch( const std::runtime_error &e ) {
			std::cerr << "Caught error: " << e.what() << "\n";
			return 1;
		}
		return 0;
	}

	int EXTBLAS_duscr_clear( blas_sparse_matrix A ) {
		auto matrix = sparseblas::getDoubleMatrix( A );
		assert( matrix->finalized );
		const grb::RC rc = grb::clear( *(matrix->A) );
		if( rc != grb::SUCCESS ) {
			return 10;
		}
		return 0;
	}

	int BLAS_usds( blas_sparse_matrix A ) {
		delete sparseblas::getDoubleMatrix( A );
		return 0;
	}

	int BLAS_dusmv(
		const enum blas_trans_type transa,
		const double alpha, const blas_sparse_matrix A,
		const double * x, int incx,
		double * const y, const int incy
	) {
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;
		auto matrix = sparseblas::getDoubleMatrix( A );
		if( alpha != 1.0 ) {
			grb::Vector< double > output = grb::internal::template
				wrapRawVector< double >( matrix->m, y );
			const grb::RC rc = grb::foldl< grb::descriptors::dense >(
				output, 1.0 / alpha, ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during pre-scaling during SpMV\n";
				return 50;
			}
		}
		if( incx != 1 || incy != 1 ) {
			// TODO: requires ALP views
			std::cerr << "Strided input and/or output vectors are not supported.\n";
			return 255;
		}
		if( !(matrix->finalized) ) {
			std::cerr << "Input matrix was not yet finalised; see BLAS_duscr_end.\n";
			return 100;
		}
		assert( matrix->finalized );
		if( transa == blas_no_trans ) {
			const grb::Vector< double > input = grb::internal::template
				wrapRawVector< double >( matrix->n, x );
			grb::Vector< double > output = grb::internal::template
				wrapRawVector< double >( matrix->m, y );
			const grb::RC rc = grb::mxv< grb::descriptors::dense >(
				output, *(matrix->A), input, ring
			);
			if( rc != grb::SUCCESS ) {
				std::cerr << "ALP/GraphBLAS returns error during SpMV: "
					<< grb::toString( rc ) << ".\n";
				return 200;
			}
		} else {
			const grb::Vector< double > input = grb::internal::template
				wrapRawVector< double >( matrix->m, x );
			grb::Vector< double > output = grb::internal::template
				wrapRawVector< double >( matrix->n, y );
			const grb::RC rc = grb::mxv<
				grb::descriptors::dense |
				grb::descriptors::transpose_matrix
			>(
				output, *(matrix->A), input, ring
			);
			if( rc != grb::SUCCESS ) {
				std::cerr << "ALP/GraphBLAS returns error during transposed SpMV: "
					<< grb::toString( rc ) << ".\n";
				return 200;
			}
		}
		if( alpha != 1.0 ) {
			grb::Vector< double > output = grb::internal::template
				wrapRawVector< double >( matrix->m, y );
			const grb::RC rc = grb::foldl< grb::descriptors::dense >(
				output, alpha, ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during post-scaling during SpMV\n";
				return 250;
			}
		}
		return 0;
	}

	void spblas_dcsrgemv(
		const char * transa,
		const int * m_p,
		const double * a, const int * ia, const int * ja,
		const double * x,
		double * y
	) {
		// declare algebraic structures
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;
		grb::Monoid<
			grb::operators::max< int >, grb::identities::negative_infinity
		> maxMonoid;

		// declare minimum necessary descriptors
		constexpr grb::Descriptor minDescr = grb::descriptors::dense |
			grb::descriptors::force_row_major;

		// determine matrix size
		const int m = *m_p;
		const grb::Vector< int > columnIndices =
			grb::internal::template wrapRawVector< int >( ia[ m ], ja );
		int n = 0;
		grb::RC rc = foldl( n, columnIndices, maxMonoid );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Could not determine matrix column size\n";
			assert( false );
			return;
		}

		// retrieve buffers (only when A needs to be output also)
		//char * const bitmask = sparseblas::getBitmask( n );
		//char * const stack = sparseblas::getStack( n );
		//double * const buffer = sparseblas::template getBuffer< double >( n );

		// retrieve necessary ALP/GraphBLAS container wrappers
		const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
			grb::internal::wrapCRSMatrix( a, ja, ia, m, n );
		const grb::Vector< double > input = grb::internal::template
			wrapRawVector< double >( n, x );
		grb::Vector< double > output = grb::internal::template
			wrapRawVector< double >( m, y );

		// set output vector to zero
		rc = grb::set( output, ring.template getZero< double >() );
		if( rc != grb::SUCCESS ) {
			std::cerr << "Could not set output vector to zero\n";
			assert( false );
			return;
		}

		// do either y=Ax or y=A^Tx
		if( transa[0] == 'N' ) {
			rc = grb::mxv< minDescr >(
				output, A, input, ring
			);
			if( rc != grb::SUCCESS ) {
				std::cerr << "ALP/GraphBLAS returns error during SpMV: "
					<< grb::toString( rc ) << ".\n";
				assert( false );
				return;
			}
		} else {
			// Hermitian is not supported
			assert( transa[0] == 'T' );
			rc = grb::mxv<
				minDescr |
				grb::descriptors::transpose_matrix
			>(
				output, A, input, ring
			);
			if( rc != grb::SUCCESS ) {
				std::cerr << "ALP/GraphBLAS returns error during transposed SpMV: "
					<< grb::toString( rc ) << ".\n";
				assert( false );
				return;
			}
		}

		// done
	}

	int BLAS_dusmm(
		const enum blas_order_type order,
		const enum blas_trans_type transa,
		const int nrhs,
		const double alpha, const blas_sparse_matrix A,
		const double * B, const int ldb,
		const double * C, const int ldc
	) {
		(void) order;
		(void) transa;
		(void) nrhs;
		(void) alpha;
		(void) A;
		(void) B;
		(void) ldb;
		(void) C;
		(void) ldc;
		// TODO requires dense ALP and mixed sparse/dense ALP operations
		std::cerr << "BLAS_dusmm (sparse matrix times dense matrix) has not yet "
			<< "been implemented.\n";
		assert( false );
		return 255;
	}

	int EXTBLAS_dusmsv(
		const enum blas_trans_type transa,
		const double alpha, const blas_sparse_matrix A,
		const extblas_sparse_vector x,
		extblas_sparse_vector y
	) {
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;
		auto matrix = sparseblas::getDoubleMatrix( A );
		auto input  = sparseblas::getDoubleVector( x );
		auto output = sparseblas::getDoubleVector( y );
		if( !(matrix->finalized) ) {
			std::cerr << "Uninitialised input matrix during SpMSpV\n";
			return 10;
		}
		if( !(input->finalized) ) {
			std::cerr << "Uninitialised input vector during SpMSpV\n";
			return 20;
		}
		if( !(output->finalized) ) {
			std::cerr << "Uninitialised output vector during SpMSpV\n";
			return 30;
		}
		grb::RC rc = grb::SUCCESS;
		if( alpha != 1.0 ) {
			rc = grb::foldl( *(output->vector), 1.0 / alpha,
				ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during pre-scaling of SpMSpV\n";
				return 40;
			}
		}
		if( transa == blas_no_trans ) {
			rc = grb::mxv( *(output->vector), *(matrix->A), *(input->vector), ring );
		} else {
			rc = grb::mxv< grb::descriptors::transpose_matrix >(
				*(output->vector), *(matrix->A), *(input->vector), ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Error during call to grb::mxv (SpMSpV)\n";
			return 50;
		}
		if( alpha != 1.0 ) {
			rc = grb::foldl( *(output->vector), alpha,
				ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during post-scaling of SpMSpV\n";
				return 60;
			}
		}
		return 0;
	}

	int EXTBLAS_dusmsm(
		const enum blas_trans_type transa,
		const double alpha, const blas_sparse_matrix A,
		const enum blas_trans_type transb, const blas_sparse_matrix B,
		blas_sparse_matrix C
	) {
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;
		auto matA = sparseblas::getDoubleMatrix( A );
		auto matB = sparseblas::getDoubleMatrix( B );
		auto matC = sparseblas::getDoubleMatrix( C );
		if( !(matA->finalized) ) {
			std::cerr << "Uninitialised left-hand input matrix during SpMSpM\n";
			return 10;
		}
		if( !(matB->finalized) ) {
			std::cerr << "Uninitialised right-hand input matrix during SpMSpM\n";
			return 20;
		}
		if( !(matC->finalized) ) {
			std::cerr << "Uninitialised output matrix during SpMSpM\n";
			return 30;
		}

		grb::RC rc = grb::SUCCESS;
		if( alpha != 1.0 ) {
			/*const grb::RC rc = grb::foldl( *(matC->A), 1.0 / alpha,
				ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during pre-scaling for SpMSpM\n";
				return 40;
			}*/
			// TODO requires level-3 fold in ALP/GraphBLAS
			std::cerr << "Any other alpha from 1.0 is currently not supported for "
				<< "SpMSpM multiplication\n";
			return 255;
		}

		// resize phase
		if( transa == blas_no_trans && transb == blas_no_trans ) {
			rc = grb::mxm( *(matC->A), *(matA->A), *(matB->A), ring, grb::RESIZE );
		} else if( transa != blas_no_trans && transb == blas_no_trans ) {
			rc = grb::mxm< grb::descriptors::transpose_left >(
				*(matC->A), *(matA->A), *(matB->A), ring, grb::RESIZE );
		} else if( transa == blas_no_trans && transb != blas_no_trans ) {
			rc = grb::mxm< grb::descriptors::transpose_right >(
				*(matC->A), *(matA->A), *(matB->A), ring, grb::RESIZE );
		} else {
			assert( transa != blas_no_trans );
			assert( transb != blas_no_trans );
			rc = grb::mxm<
				grb::descriptors::transpose_left |
				grb::descriptors::transpose_right
			>( *(matC->A), *(matA->A), *(matB->A), ring, grb::RESIZE );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Error during call to ALP/GraphBLAS mxm (RESIZE phase): "
				<< grb::toString( rc ) << "\n";
			return 50;
		}

		// execute phase
		if( transa == blas_no_trans && transb == blas_no_trans ) {
			rc = grb::mxm( *(matC->A), *(matA->A), *(matB->A), ring );
		} else if( transa != blas_no_trans && transb == blas_no_trans ) {
			rc = grb::mxm< grb::descriptors::transpose_left >(
				*(matC->A), *(matA->A), *(matB->A), ring );
		} else if( transa == blas_no_trans && transb != blas_no_trans ) {
			rc = grb::mxm< grb::descriptors::transpose_right >(
				*(matC->A), *(matA->A), *(matB->A), ring );
		} else {
			assert( transa != blas_no_trans );
			assert( transb != blas_no_trans );
			rc = grb::mxm<
				grb::descriptors::transpose_left |
				grb::descriptors::transpose_right
			>( *(matC->A), *(matA->A), *(matB->A), ring );
		}
		if( rc != grb::SUCCESS ) {
			std::cerr << "Error during call to ALP/GraphBLAS mxm (EXECUTE phase): \n"
				<< grb::toString( rc ) << "\n";
			return 60;
		}

		/*TODO see above
		if( alpha != 1.0 ) {
			rc = grb::foldl( *(matC->A), 1.0 / alpha,
				ring.getMultiplicativeOperator() );
			if( rc != grb::SUCCESS ) {
				std::cerr << "Error during post-scaling for SpMSpM\n";
				return 70;
			}
		}*/
		return 0;
	}

	int EXTBLAS_free() {
		const grb::RC rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			std::cerr << "Error during call to EXTBLAS_free\n";
			return 10;
		}
		return 0;
	}

	void extspblas_free() {
		(void) EXTBLAS_free();
	}

} // end extern "C"

