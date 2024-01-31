
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
#include <iterator>
#include <iostream>
#include <stdexcept>

#include <assert.h>

#include <graphblas.hpp>
#include <spblas.h>


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
	class MatrixUC;

	namespace internal {

		/**
		 * \internal An iterator over the triplets contained herein. The iterator
		 *           adheres both to STL as well as ALP.
		 */
		template< typename T >
		class MatrixUCIterator {

			friend class sparseblas::MatrixUC< T >;

			private:

				const std::vector< FullTripletBatch< T > > &batches;
				const PartialTripletBatch< T > &last;
				size_t batch;
				size_t loc;


			protected:

				MatrixUCIterator( const MatrixUC< T > &x ) :
					batches( x.batches ), last( x.last ),
					batch( 0 ), loc( 0 )
				{}

				void setToEndPosition() {
					batch = batches.size();
					loc = last.ntriplets;
				}


			public:

				typedef Triplet< T > value_type;
				typedef const value_type & reference_type;
				typedef const value_type * pointer_type;
				typedef std::forward_iterator_tag iterator_category;
				typedef size_t difference_type;
				typedef int RowIndexType;
				typedef int ColumnIndexType;
				typedef T ValueType;

				MatrixUCIterator( const MatrixUCIterator &other ) :
					batches( other.batches ), last( other.last ),
					batch( other.batch ), loc( other.loc )
				{}

				MatrixUCIterator( MatrixUCIterator &&other ) :
					batches( other.batches ), last( other.last ),
					batch( other.batch ), loc( other.loc )
				{
					other.batch = other.loc = 0;
				}

				bool operator==( const MatrixUCIterator &other ) const {
					return batch == other.batch && loc == other.loc;
				}

				bool operator!=( const MatrixUCIterator &other ) const {
					return !( operator==( other ) );
				}

				MatrixUCIterator operator++() {
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

	} // end namespace sparseblas::internal

	template< typename T >
	class MatrixUC {

		friend class internal::MatrixUCIterator< T >;

		private:

			/** \internal A series of full triplet batches. */
			std::vector< FullTripletBatch< T > > batches;

			/** \internal One partial batch of triplets. */
			PartialTripletBatch< T > last;


		public:

			/** Typedef our iterator */
			typedef typename internal::MatrixUCIterator< T > const_iterator;

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

			typename grb::Matrix< T >::const_iterator start, end;

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

	/**
	 * \internal Internal buffer used for output matrix containers.
	 */
	char * buffer = nullptr;

	/**
	 * \internal The size of #buffer.
	 */
	size_t buffer_size = 0;

	/**
	 * @returns false if and only if buffer allocation failed.
	 * @returns true on success.
	 */
	template< typename T >
	bool getBuffer(
		char * &bitmask, char * &stack, T * &valbuf,
		const size_t size
	) {
		typedef typename grb::internal::Coordinates< grb::config::default_backend >
			Coors;
		constexpr const size_t b = grb::config::CACHE_LINE_SIZE::value();

		// catch trivial case
		if( size == 0 ) {
			bitmask = stack = nullptr;
			valbuf = nullptr;
			return true;
		}

		// compute required size
		size_t reqSize = Coors::arraySize( size ) + Coors::stackSize( size ) +
			(size * sizeof(T)) + 3 * b;

		// ensure buffer is at least the required size
		if( buffer == nullptr ) {
			assert( buffer_size == 0 );
			buffer_size = reqSize;
			buffer = static_cast< char * >( malloc( buffer_size ) );
			if( buffer == nullptr ) {
				buffer_size = 0;
				return false;
			}
		} else if( buffer_size < reqSize ) {
			free( buffer );
			buffer_size = std::max( reqSize, 2 * buffer_size );
			buffer = static_cast< char * >( malloc( buffer_size ) );
			if( buffer == nullptr ) {
				buffer_size = 0;
				return false;
			}
		}

		// set buffers and make sure they are aligned
		char * walk = buffer;
		uintptr_t cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
		if( cur_mod > 0 ) {
			walk += (b - cur_mod);
		}
		bitmask = walk;
		walk += Coors::arraySize( size );
		cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
		if( cur_mod > 0 ) {
			walk += (b - cur_mod);
		}
		stack = walk;
		walk += Coors::stackSize( size );
		cur_mod = reinterpret_cast< uintptr_t >(walk) % b;
		if( cur_mod > 0 ) {
			walk += (b - cur_mod);
		}
		valbuf = reinterpret_cast< T * >( walk);

		// done
		return true;
	}

} // end namespace sparseblas

namespace std {

	/**
	 * Specialisation for the STL std::iterator_traits for the MatrixUC iterator.
	 */
	template< typename T >
	class iterator_traits< typename sparseblas::internal::MatrixUCIterator< T > > {

		private:

			typedef typename sparseblas::MatrixUC< T >::const_iterator SelfType;


		public:

			typedef typename SelfType::value_type value_type;
			typedef typename SelfType::pointer_type pointer_type;
			typedef typename SelfType::reference_type reference_type;
			typedef typename SelfType::iterator_category iterator_category;
			typedef typename SelfType::difference_type difference_type;

	};

} // end namespace std

// implementation of the SparseBLAS API follows

extern "C" {

	extblas_sparse_vector EXTBLAS_FUN( dusv_begin )( const int n ) {
		return new sparseblas::SparseVector< double >( n );
	}

	int EXTBLAS_FUN( dusv_insert_entry )(
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

	int EXTBLAS_FUN( dusv_end )( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( !(vector->finalized) );
		try {
			vector->finalize();
		} catch( ... ) {
			return 30;
		}
		return 0;
	}

	int EXTBLAS_FUN( dusvds )( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		delete vector;
		return 0;
	}

	int EXTBLAS_FUN( dusv_nz )( const extblas_sparse_vector x, int * const nz ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		const size_t nnz = grb::nnz( *(vector->vector) );
		if( nnz > static_cast< size_t >( std::numeric_limits< int >::max() ) ) {
			std::cerr << "Number of nonzeroes is larger than what can be represented by "
				<< "a SparseBLAS int!\n";
			return 10;
		}
		*nz = static_cast< int >(nnz);
		return 0;
	}

	int EXTBLAS_FUN( dusv_clear )( extblas_sparse_vector x ) {
		auto vector = sparseblas::getDoubleVector( x );
		assert( vector->finalized );
		const grb::RC rc = grb::clear( *(vector->vector) );
		if( rc != grb::SUCCESS ) {
			return 10;
		}
		return 0;
	}

	int EXTBLAS_FUN( dusv_open )( const extblas_sparse_vector x ) {
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

	int EXTBLAS_FUN( dusv_get )(
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

	int EXTBLAS_FUN( dusv_close )( const extblas_sparse_vector x ) {
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

	int EXTBLAS_dusm_clear( blas_sparse_matrix A ) {
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

	SPBLAS_RET_T SPBLAS_NAME( dcsrgemv )(
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

	SPBLAS_RET_T SPBLAS_NAME( dcsrmm )(
		const char * const transa,
		const int * m, const int * n, const int * k,
		const double * alpha,
		const char * matdescra, const double * val, const int * indx,
		const int * pntrb, const int * pntre,
		const double * b, const int * ldb,
		const double * beta,
		double * c, const int * ldc
	) {
		assert( transa[0] == 'N' || transa[0] == 'T' );
		assert( m != NULL );
		assert( n != NULL );
		assert( k != NULL );
		assert( alpha != NULL );
		// not sure yet what constraints if any on matdescra
		if( *m > 0 && *k > 0 ) {
			assert( pntrb != NULL );
			assert( pntre != NULL );
		}
		// val and indx could potentially be NULL if there are no nonzeroes
		assert( b != NULL );
		assert( ldb != NULL );
		assert( beta != NULL );
		assert( c != NULL );
		assert( ldc != NULL );
		(void) transa;
		(void) m; (void) n; (void) k;
		(void) alpha;
		(void) matdescra; (void) val; (void) indx; (void) pntrb; (void) pntre;
		(void) b; (void) ldb;
		(void) beta;
		(void) c; (void) ldc;
		// requires dense ALP and mixed sparse/dense operations
		assert( false );
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

	SPBLAS_RET_T EXT_SPBLAS_NAME( dcsrmultsv )(
		const char * trans, const int * request,
		const int * m, const int * n,
		const double * a, const int * ja, const int * ia,
		const extblas_sparse_vector x,
		extblas_sparse_vector y
	) {
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;
		const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
			grb::internal::wrapCRSMatrix( a, ja, ia, *m, *n );
		auto input  = sparseblas::getDoubleVector( x );
		auto output = sparseblas::getDoubleVector( y );
		if( !(input->finalized) ) {
			throw std::runtime_error( "Uninitialised input vector during SpMSpV\n" );
		}
		if( !(output->finalized) ) {
			throw std::runtime_error( "Uninitialised output vector during SpMSpV\n" );
		}
		if( request[ 0 ] != 0 && request[ 1 ] != 1 ) {
			throw std::runtime_error( "Illegal request during call to dcsrmultsv\n" );
		}
		grb::Phase phase = grb::EXECUTE;
		if( request[ 0 ] == 1 ) {
			phase = grb::RESIZE;
		}
		grb::RC rc;
		if( trans[0] == 'N' ) {
			rc = grb::mxv< grb::descriptors::force_row_major >( *(output->vector), A,
				*(input->vector), ring, phase );
		} else {
			if( trans[1] != 'T' ) {
				throw std::runtime_error( "Illegal trans argument to dcsrmultsv\n" );
			}
			rc = grb::mxv<
				grb::descriptors::force_row_major |
				grb::descriptors::transpose_matrix
			>( *(output->vector), A, *(input->vector), ring, phase );
		}
		if( rc != grb::SUCCESS ) {
			throw std::runtime_error( "ALP/GraphBLAS returns error during call to "
				"SpMSpV: " + grb::toString( rc ) );
		}
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

	SPBLAS_RET_T SPBLAS_NAME( dcsrmultcsr )(
		const char * trans, const int * request, const int * sort,
		const int * m_p, const int * n_p, const int * k_p,
		double * a, int * ja, int * ia,
		double * b, int * jb, int * ib,
		double * c, int * jc, int * ic,
		const int * nzmax, int * info
	) {
		assert( trans[0] == 'N' );
		assert( sort != NULL && sort[0] == 7 );
		assert( m_p != NULL );
		assert( n_p != NULL );
		assert( k_p != NULL );
		assert( a != NULL ); assert( ja != NULL ); assert( ia != NULL );
		assert( b != NULL ); assert( jb != NULL ); assert( ib != NULL );
		assert( c != NULL ); assert( jc != NULL ); assert( ic != NULL );
		assert( nzmax != NULL );
		assert( info != NULL );

		// declare algebraic structures
		grb::Semiring<
			grb::operators::add< double >, grb::operators::mul< double >,
			grb::identities::zero, grb::identities::one
		> ring;

		// check support
		if( trans[ 0 ] != 'N' ) {
			std::cerr << "ALP/SparseBLAS, error: illegal trans argument to dcsrmultcsr\n";
			*info = 4;
		}
		if( sort[ 0 ] != 7 ) {
			std::cerr << "ALP/SparseBLAS, error: illegal sort argument to dcsrmultcsr\n";
			*info = 5;
			return;
		}

		// declare minimum necessary descriptors
		constexpr const grb::Descriptor minDescr = grb::descriptors::dense |
			grb::descriptors::force_row_major;

		// determine matrix size
		const int m = *m_p;
		const int n = *n_p;
		const int k = *k_p;

		// retrieve buffers (only when A needs to be output also)
		char * bitmask = nullptr;
		char * stack = nullptr;
		double * valbuf = nullptr;
		if( sparseblas::template getBuffer< double >(
				bitmask, stack, valbuf, n
			) == false
		) {
			std::cerr << "ALP/SparseBLAS, error: could not allocate buffer for "
				<< "computations on an output matrix\n";
			*info = 10;
			return;
		}

		// retrieve necessary ALP/GraphBLAS container wrappers
		const grb::Matrix< double, grb::config::default_backend, int, int, int > A =
			grb::internal::wrapCRSMatrix( a, ja, ia, m, k );
		const grb::Matrix< double, grb::config::default_backend, int, int, int > B =
			grb::internal::wrapCRSMatrix( b, jb, ib, k, n );
		grb::Matrix< double, grb::config::default_backend, int, int, int > C =
			grb::internal::wrapCRSMatrix(
				c, jc, ic,
				m, n, *nzmax,
				bitmask, stack, valbuf
			);

		// set output vector to zero
		grb::RC rc = grb::clear( C );
		if( rc != grb::SUCCESS ) {
			std::cerr << "ALP/SparseBLAS, error: Could not clear output matrix\n";
			assert( false );
			*info = 20;
			return;
		}

		// do either C=AB or C=A^TB
		if( trans[0] == 'N' ) {
			if( *request == 1 ) {
				rc = grb::mxm< minDescr >( C, A, B, ring, grb::RESIZE );
			} else {
				assert( *request == 0 || *request == 2 );
				rc = grb::mxm< minDescr >( C, A, B, ring );
			}
			if( rc != grb::SUCCESS ) {
				std::cerr << "ALP/SparseBLAS, error during call to SpMSpM: "
					<< grb::toString( rc ) << ".\n";
				assert( false );
				*info = 30;
				return;
			}
		} else {
			// this case is not supported
			assert( false );
		}

		// done
		if( *request == 1 ) {
			*info = -1;
		} else {
			*info = 0;
		}
	}

	int EXTBLAS_dusm_nz( const blas_sparse_matrix A, int * nz ) {
		auto matA = sparseblas::getDoubleMatrix( A );
		if( !(matA->finalized) ) {
			std::cerr << "Uninitialised left-hand input matrix during dusm_nz\n";
			return 10;
		}
		const size_t grb_nz = grb::nnz( *(matA->A) );
		if( grb_nz > static_cast< size_t >(std::numeric_limits< int >::max()) ) {
			std::cerr << "Number of nonzeroes in given sparse matrix is larger than "
				<< "what can be represented by a SparseBLAS int\n";
			return 20;
		}
		*nz = static_cast< int >( grb_nz );
		return 0;
	}

	int EXTBLAS_dusm_open( const blas_sparse_matrix A ) {
		auto matA = sparseblas::getDoubleMatrix( A );
		if( !(matA->finalized) ) {
			std::cerr << "Uninitialised left-hand input matrix during dusm_nz\n";
			return 10;
		}
		try{
			matA->start = matA->A->cbegin();
			matA->end = matA->A->cend();
		} catch( ... ) {
			std::cerr << "Could not retrieve matrix iterators\n";
			return 20;
		}
		return 0;
	}

	int EXTBLAS_dusm_get(
		const blas_sparse_matrix A,
		double * value, int * row, int * col
	) {
		auto matA = sparseblas::getDoubleMatrix( A );
		if( !(matA->finalized) ) {
			std::cerr << "Uninitialised left-hand input matrix during dusm_nz\n";
			return 10;
		}
		assert( matA->start != matA->end );
		const auto &triplet = *(matA->start);
		*value = triplet.second;
		*row = triplet.first.first;
		*col = triplet.first.second;
		try {
			(void) ++(matA->start);
		} catch( ... ) {
			return 2;
		}
		if( matA->start == matA->end ) {
			return 0;
		} else {
			return 1;
		}
	}

	int EXTBLAS_dusm_close( const blas_sparse_matrix A ) {
		auto matA = sparseblas::getDoubleMatrix( A );
		if( !(matA->finalized) ) {
			std::cerr << "Uninitialised left-hand input matrix during dusm_nz\n";
			return 10;
		}
		matA->start = matA->end;
		return 0;
	}

	int EXTBLAS_free() {
		if( sparseblas::buffer != nullptr || sparseblas::buffer_size > 0 ) {
			assert( sparseblas::buffer != nullptr );
			assert( sparseblas::buffer_size > 0 );
			free( sparseblas::buffer );
			sparseblas::buffer_size = 0;
		}
		const grb::RC rc = grb::finalize();
		if( rc != grb::SUCCESS ) {
			std::cerr << "Error during call to EXTBLAS_free\n";
			return 10;
		}
		return 0;
	}

	SPBLAS_RET_T EXT_SPBLAS_NAME( free )() {
		(void) EXTBLAS_free();
	}

} // end extern "C"

