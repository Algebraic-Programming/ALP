
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
 * Implements the level-3 primitives for the nonblocking backend
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BLAS3
#define _H_GRB_NONBLOCKING_BLAS3

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits> //for std::enable_if

#include <graphblas/base/blas3.hpp>
#include <graphblas/nonblocking/analytic_model.hpp>
#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>

#include "io.hpp"
#include "matrix.hpp"

#include <omp.h>

#define NO_CAST_ASSERT( x, y, z )                                              \
	static_assert( x,                                                          \
		"\n\n"                                                                 \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"*     ERROR      | " y " " z ".\n"                                    \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n"                                     \
		"* Possible fix 1 | Remove no_casting from the template parameters "   \
		"in this call to " y ".\n"                                             \
		"* Possible fix 2 | For all mismatches in the domains of input "       \
		"parameters and the semiring domains, as specified in the "            \
		"documentation of the function " y ", supply a container argument of " \
		"the expected type instead.\n"                                         \
		"* Possible fix 3 | Provide a compatible semiring where all domains "  \
		"match those of the container arguments, as specified in the "         \
		"documentation of the function " y ".\n"                               \
		"********************************************************************" \
		"********************************************************************" \
		"******************************\n" );

namespace grb {

	namespace internal {

		extern LazyEvaluation le;

	}

} // namespace grb

namespace grb {

	namespace internal {

		/*
		//count the number of nonzeros in each tile of the output matrix C = AB
		template< typename OutputType, typename InputType1, typename InputType2, typename RIT, typename CIT, typename NIT >
		void count_NonZeros_Tile_MXM( Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
			const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
			const NIT tile_id,
			const NIT lower_bound,
			const NIT upper_bound ) {
			const auto & A_raw = internal::getCRS( A );
			const auto & B_raw = internal::getCRS( B );
			const size_t n = grb::ncols( B );

			// to store coordinates. These are local to each tile
			char arr[ n ];
			char buf[ n ];

			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n );
			size_t nnz_current_tile = 0;

			for( size_t i = lower_bound; i < upper_bound; ++i ) {
				coors.clear();
				for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					const size_t k_col = A_raw.row_index[ k ];
					for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( ! coors.assign( l_col ) ) {
							(void)++nnz_current_tile;
						}
					}
				}
			}

			// read vector that stores the number of nnz in each tile of matrix C
			auto & nnz_tiles_C = internal::getNonzerosTiles( C );

			// find tile_id and assign corresponding nnz
			for( size_t i = 0; i < nnz_tiles_C.size(); i++ ) {
				if( tile_id == nnz_tiles_C[ i ].first ) {
					nnz_tiles_C[ i ].second = nnz_current_tile;
				}
			}

			// std::cout << "number of nnz in tile " << nnz_current_tile << std::endl;
		}
		*/

		template< 
			Descriptor descr = descriptors::no_operation, 
			typename InputType, 
			typename RIT, typename CIT,
			typename NIT, 
			typename IOType,
			class Monoid 
		>
		RC foldl_unmasked_generic( IOType & x, Matrix< InputType, nonblocking, RIT, CIT, NIT > & A, const Monoid & monoid ) {
#ifdef _DEBUG
			std::cout << "In grb::internal::foldl_unmasked_generic\n";
#endif
			// nonblocking implementation
			RC ret = SUCCESS;
						
			// this stores the result of the reduction operation
			typename Monoid::D3 reduced = monoid.template getIdentity< typename Monoid::D3 >();
			
			//size_t reduced_size = sysconf( _SC_NPROCESSORS_ONLN ) * config::CACHE_LINE_SIZE::value();
			size_t reduced_size = grb::config::OMP::threads() * config::CACHE_LINE_SIZE::value();		
			
			// vector that stores the accumulated sum in each tile. This vector is used only by this primitive and then it
			// does not have to exist after executing this primitive
			typename Monoid::D3 array_reduced[ reduced_size ];

			for( size_t i = 0; i < reduced_size; i += config::CACHE_LINE_SIZE::value() ) {
				array_reduced[ i ] = monoid.template getIdentity< typename Monoid::D3 >();
			}

			/*
			// implementation accumulating using tiles ID			
			typename Monoid::D3 reduced = monoid.template getIdentity< typename Monoid::D3 >();
			size_t reduced_size = internal::getNumTilesNonblocking(A);
			typename Monoid::D3 array_reduced[ reduced_size ];
			for( size_t i = 0; i < reduced_size; ++i ) {
			    array_reduced[ i ] = monoid.template getIdentity< typename Monoid::D3 >();
			}	
			*/					

			// lambda  function to count the nnz in each tile
			internal::Pipeline::count_nnz_local_type func_count_nonzeros = []( const size_t lower_bound, const size_t upper_bound ) {
				(void)lower_bound;
				(void)upper_bound;
				return SUCCESS;
			};

			// lambda function to compute the prefix sum of local nnz
			internal::Pipeline::prefix_sum_nnz_mxm_type func_prefix_sum = []( ) {			
				return SUCCESS;
			};

			// lambda where computation are performed
			internal::Pipeline::stage_type func = [ &A, &monoid, &array_reduced ]( internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
				(void)pipeline;
				(void)upper_bound;

				const auto & A_raw = internal::getCRS( A );
				const auto & prefix_sum_tiles_A = internal::getPrefixSumTiles( A );
				
				// TODO: analytic model for tile size
				const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
				const size_t tile_id = lower_bound / tile_size;

				unsigned int thread_id = omp_get_thread_num() * config::CACHE_LINE_SIZE::value();
				
				size_t previous_nnz;
				size_t current_nnz;

				// special case for first tile
				if( 0 == tile_id ) {
					previous_nnz = 0;
					current_nnz = prefix_sum_tiles_A[ tile_id ];
				} else {
					previous_nnz = prefix_sum_tiles_A[ tile_id - 1 ];
					current_nnz = prefix_sum_tiles_A[ tile_id ];
				}
			
				// compute sum using the CRS format of A.
				for( size_t i = previous_nnz; i < current_nnz; ++i ) {
					const InputType a = A_raw.getValue( i, monoid.template getIdentity< typename Monoid::D3 >() );
					foldl( array_reduced[ thread_id ], a, monoid.getOperator() );					
				}

				/*
				#pragma omp critical
				{
				    std::cout << "tile_id = " << tile_id << ", accumulated sum = " << local_result << std::endl;
				}
				*/

				return SUCCESS;
			};

			/*
			// lambda where computation are performed
			internal::Pipeline::stage_type func = [ &A_raw, &nnz_accumulated, &monoid ]( internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
			    (void)pipeline;

			    // TODO: analytic model for tile size
			    const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
			    const size_t tile_id = lower_bound / tile_size;

			    const auto & identity = monoid.template getIdentity< typename Monoid::D3 >();
			    const auto & op = monoid.getOperator();
			    RC local_rc = SUCCESS;
			    auto local_x = identity;

			    size_t start, end;
			    start = A_raw.col_start[ lower_bound ];
			    end = A_raw.col_start[ upper_bound ];

			    for( size_t k = start; k < end; ++k ) {
			        const InputType a = A_raw.getValue( k, identity );
			        local_rc = local_rc ? local_rc : grb::foldl( local_x, a, op );
			    }

			    nnz_accumulated[ tile_id ] = local_x;

			    return SUCCESS;
			};
			*/

			ret = ret ? ret :
						internal::le.addStageLevel3( std::move( func ),
							// name of operation
							internal::Opcode::BLAS3_SCALAR_REDUCTION,
							// size of output matrix
							grb::nrows( A ),
							// size of data type in matrix C
							sizeof( InputType ),
							// dense_descr
							true,
							// dense_mask
							true,
							// matrices for mxm
							&A, nullptr, nullptr, nullptr, std::move( func_count_nonzeros ), std::move( func_prefix_sum ) );

			// compute final accumulated result computed by each tile.
			// we can do this since by this point the pipeline has been executed and array_reduced holds all its results
									
			for( size_t i = 0; i < reduced_size; i += config::CACHE_LINE_SIZE::value() ) {
				(void)grb::foldl( reduced, array_reduced[ i ], monoid.getOperator() );
			}
		
			// write back result
			x = static_cast< InputType >( reduced );

			return ret;
		}		

		template< Descriptor descr = descriptors::no_operation, typename InputType, typename RIT, typename CIT, typename NIT, typename IOType, class Monoid >
		// template< typename D, typename T, typename Func >
		RC matrixReduce( Matrix< InputType, nonblocking, RIT, CIT, NIT > & A, IOType & result, const Monoid & monoid ) {
			// nonblocking implementation
			RC ret = SUCCESS;
			const auto & A_raw = internal::getCRS( A );

			// std::cout << "number of tiles = " << internal::getNumTilesNonblocking( A ) << std::endl;
			auto & prefix_sum_tiles_A = internal::getPrefixSumTiles( A );

			// this stores the result of the reduction operation
			typename Monoid::D3 reduced = monoid.template getIdentity< typename Monoid::D3 >();

			size_t reduced_size = sysconf( _SC_NPROCESSORS_ONLN ) * config::CACHE_LINE_SIZE::value();

			

			// vector that stores the accumulated sum in each tile. This vector is used only by this primitive and then it
			// does not have to exist after executing this primitive
			typename Monoid::D3 array_reduced[ reduced_size ];

			for( size_t i = 0; i < reduced_size; i += config::CACHE_LINE_SIZE::value() ) {
				array_reduced[ i ] = monoid.template getIdentity< typename Monoid::D3 >();
			}

			// lambda  function to count the nnz in each tile
			internal::Pipeline::count_nnz_local_type func_count_nonzeros = []( const size_t lower_bound, const size_t upper_bound ) {
				(void)lower_bound;
				(void)upper_bound;
				return SUCCESS;
			};

			// lambda function to compute the prefix sum of local nnz
			internal::Pipeline::prefix_sum_nnz_mxm_type func_prefix_sum = []() {
				std::cout << "execute prefix from FOLDL " << std::endl;
				return SUCCESS;
			};

			// lambda where computation are performed
			internal::Pipeline::stage_type func = [ &prefix_sum_tiles_A, &A_raw, &monoid, &array_reduced ]( internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
				(void)pipeline;
				(void)upper_bound;

				// TODO: analytic model for tile size
				const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
				const size_t tile_id = lower_bound / tile_size;

				unsigned int thread_id = omp_get_thread_num() * config::CACHE_LINE_SIZE::value();

				size_t previous_nnz;
				size_t current_nnz;

				// special case for first tile
				if( 0 == tile_id ) {
					previous_nnz = 0;
					current_nnz = prefix_sum_tiles_A[ tile_id ];
				} else {
					previous_nnz = prefix_sum_tiles_A[ tile_id - 1 ];
					current_nnz = prefix_sum_tiles_A[ tile_id ];
				}

				// compute sum using the CRS format of A.
				for( size_t i = previous_nnz; i < current_nnz; ++i ) {
					const InputType a = A_raw.getValue( i, monoid.template getIdentity< typename Monoid::D3 >() );
					grb::foldl( array_reduced[ thread_id ], a, monoid.getOperator() );
				}

				/*
				#pragma omp critical
				{
				    std::cout << "tile_id = " << tile_id << ", accumulated sum = " << local_result << std::endl;
				}
				*/

				return SUCCESS;
			};

			ret = ret ? ret :
						internal::le.addStageLevel3( std::move( func ),
							// name of operation
							internal::Opcode::BLAS3_SCALAR_REDUCTION,
							// size of output matrix
							grb::nrows( A ),
							// size of data type in matrix C
							sizeof( InputType ),
							// dense_descr
							true,
							// dense_mask
							true,
							// matrices for mxm
							&A, nullptr, nullptr, nullptr, std::move( func_count_nonzeros ), std::move( func_prefix_sum ) );

			// compute final accumulated result computed by each tile.
			// we can do this since by this point the pipeline has been executed and array_reduced holds all its results
			for( size_t i = 0; i < reduced_size; i += config::CACHE_LINE_SIZE::value() ) {
				(void)grb::foldl( reduced, array_reduced[ i ], monoid.getOperator() );
			}

			// write back result
			result = static_cast< InputType >( reduced );

			return ret;
		}

		/*
		template< typename D, typename T >
		RC matrixSumReduce( grb::Matrix< D > & A, T & result ) {
		    return matrixReduce( A, result, []( T a, D b ) -> T {
		        return a + b;
		    } );
		}
		*/
			
		/*implementation of masked mxm*/
		template< bool allow_void,
			Descriptor descr,
			bool output_masked,
			class MulMonoid,
			typename OutputType, typename InputType1, typename InputType2,
			typename RIT, typename CIT, typename NIT,
			typename maskType,
			class Operator,
			class Monoid>
		RC mxm_masked_generic( Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
			const Matrix< maskType, nonblocking, RIT, CIT, NIT > & C_mask,
			const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
			const Operator & oper,
			const Monoid & monoid,
			const MulMonoid & mulMonoid,
			const Phase & phase,
			const typename std::enable_if< 
				! grb::is_object< OutputType >::value && 
				! grb::is_object< InputType1 >::value && 
				! grb::is_object< InputType2 >::value &&
				! grb::is_object< maskType >::value &&
				grb::is_operator< Operator >::value && 
				grb::is_monoid< Monoid >::value,
				void >::type * const = nullptr ) {

			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)" );

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_masked_generic (reference, masked)\n";
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// get whether we are required to stick to CRS
			constexpr bool crs_only = descr & descriptors::force_row_major;

			// static checks
			static_assert( ! ( crs_only && trans_left ),
				"Cannot (presently) transpose A "
				"and force the use of CRS" );
			static_assert( ! ( crs_only && trans_right ),
				"Cannot (presently) transpose B "
				"and force the use of CRS" );

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = ! trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t k = ! trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t k_B = ! trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = ! trans_right ? grb::ncols( B ) : grb::nrows( B );

			const size_t m_C_mask = grb::nrows( C_mask );
			const size_t n_C_mask = grb::ncols( C_mask);

			assert( phase != TRY );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}			

			//check that mask of C has the same dimensions of C
			if(m != m_C_mask || n != n_C_mask ){
				return MISMATCH;
			}			

			// read data from matrices 
			const auto &A_raw = !trans_left
				? internal::getCRS( A )
				: internal::getCCS( A );
			const auto &B_raw = !trans_right
				? internal::getCRS( B )
				: internal::getCCS( B );
			auto &C_raw = internal::getCRS( C );					

			char * arr = nullptr;
			char * buf = nullptr;
			OutputType * valbuf = nullptr;
			internal::getMatrixBuffers( arr, buf, valbuf, 1, C );
	
			// initialisations
			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n );

			//read data from C_mask
			const auto &C_mask_raw = !trans_left
				? internal::getCRS( C_mask )
				: internal::getCCS( C_mask );

			char * arr_mask = nullptr;
			char * buf_mask = nullptr;
			OutputType * valbuf_mask = nullptr;
			internal::getMatrixBuffers( arr_mask, buf_mask, valbuf_mask, 1, C_mask );
			internal::Coordinates< reference > coors_mask;
			coors_mask.set( arr_mask, false, buf_mask, n );
			
			// end initialisations
			
			// symbolic phase (counting sort, step 1)
			size_t nzc = 0; // output nonzero count
			//size_t nzc_mask = 0;

			//if( crs_only && phase == RESIZE ) {
				// we are using an auxialiary CRS that we cannot resize ourselves
				// instead, we update the offset array only
			//	C_raw.col_start[ 0 ] = 0;
			//}
			
			// if crs_only, then the below implements its resize phase
			// if not crs_only, then the below is both crucial for the resize phase,
			// as well as for enabling the insertions of output values in the output CCS

			/*
			if( (crs_only && phase == RESIZE) || !crs_only ) {				
				for( size_t i = 0; i < m; ++i ) {

					// we traverse C_mask to find column indices of nonzero elements
					// we know that the total number of nonzeros in the output matrix is the number of 
					// nonzeros in the mask	
					coors_mask.clear();				
					for( auto k = C_mask_raw.col_start[ i ]; k < C_mask_raw.col_start[ i + 1 ]; ++k ) {						
						const size_t k_col = C_mask_raw.row_index[ k ];						
						if( ! coors_mask.assign( k_col ) ) {
							(void)++nzc;
						}
					}														
				}
			}
			*/
			
			if( phase == RESIZE ) {
				if( !crs_only ) {
					// do final resize					
					std::cout << "value of nzc to pass to resize = " << internal::getCurrentNonzeroes(C_mask) << std::endl;
					// this will update cap of C to nzc
					const RC ret = grb::resize( C, internal::getCurrentNonzeroes(C_mask) );	

#ifndef NDEBUG
					const size_t old_nzc = internal::getCurrentNonzeroes(C_mask);
#endif				
					// set nzc to zero
					nzc= 0;

					C_raw.col_start[ 0 ] = 0;

					for( size_t i = 0; i < m; ++i ) {

						// we traverse C_mask to find column indices of nonzero elements
						
						coors_mask.clear();				
						for( auto k = C_mask_raw.col_start[ i ]; k < C_mask_raw.col_start[ i + 1 ]; ++k ) {						
							const size_t k_col = C_mask_raw.row_index[ k ];
							coors_mask.assign( k_col );							
						}
						// read column indices of nonzeros in coors_mask and copy them into nonzero_indices_mask
						unsigned int nonzero_indices_mask[coors_mask.nonzeroes()]; 						
						coors_mask.packValues( nonzero_indices_mask, 0, nullptr,nullptr );

						//sort of nonzero_indices_mask
						std::sort(nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes());						

						// check column indices of nonzeros int current row i of C = AB
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for(
								auto l = B_raw.col_start[ k_col ];
								l < B_raw.col_start[ k_col + 1 ];
								++l
							) {							
								const size_t l_col = B_raw.row_index[ l ];

								//search column indices that are common to the mask and to C
								// use binary search on sorted nonzero_indices_mask. STL implementation																												
								if( std::binary_search( nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes(), l_col, []( const int & a, const int & b ) { return a < b;}) ) {
									coors.assign( l_col );
								}
								
								/*
								if(i == l_col)																																
								{
									coors.assign( l_col );
								}
								*/
							}						
						}
						for( size_t k = 0; k < coors.nonzeroes(); k++ ) {
							assert( nzc < old_nzc );
							const size_t j = coors.index( k );
							// update CRS -> row_index
							C_raw.row_index[ nzc ] = j;
							(void)++nzc;
						}
						// update CRS -> col_start
						C_raw.col_start[ i + 1 ] = nzc;
					}					

					return ret;
				} else {
					// we are using an auxiliary CRS that we cannot resize
					// instead, we updated the offset array in the above and can now exit
					return SUCCESS;
				}
			}

			RC ret = SUCCESS;

			if( phase == EXECUTE ) {
				// lambda  function to count the nnz in each tile
				internal::Pipeline::count_nnz_local_type func_count_nonzeros = [ &A, &B, &C, &C_mask ]( const size_t lower_bound, const size_t upper_bound ) {
					
					// output matrix C sizes
					const size_t n = grb::ncols( C );					
					const auto & A_raw = internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B ) ;					
					const auto & C_mask_raw = internal::getCRS( C_mask ) ;
					auto & nnz_tiles_C = internal::getNonzerosTiles( C );	

					// we retrieve information about the tiles
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;
								
					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();

					/*
					std::vector< char > coorArr;
					std::vector< char > coorBuf;
					std::vector< OutputType > valbuf;
					internal::getCoordinatesTiles( coorArr, coorBuf, valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( coorArr.data() ), false, static_cast< void * >( coorBuf.data() ), n );
					
					// coordinates for mask
					std::vector< char > coorArr_mask;
					std::vector< char > coorBuf_mask;
					std::vector< maskType > valbuf_mask;
					internal::getCoordinatesTiles( coorArr_mask, coorBuf_mask, valbuf_mask, coordinates_id, C_mask );
					internal::Coordinates< reference > coors_mask;
					coors_mask.set( static_cast< void * >( coorArr_mask.data() ), false, static_cast< void * >( coorBuf_mask.data() ), n );
					*/
					
					std::vector< char >* ptr_coorArr;
					std::vector< char >* ptr_coorBuf;
					std::vector< OutputType >* ptr_valbuf;

					internal::getThreadsBuffers( ptr_coorArr, ptr_coorBuf, ptr_valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( ptr_coorArr->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );	

					// coordinates for mask
					std::vector< char >* ptr_coorArr_mask;
					std::vector< char >* ptr_coorBuf_mask;
					std::vector< OutputType >* ptr_valbuf_mask;

					internal::getThreadsBuffers( ptr_coorArr_mask, ptr_coorBuf_mask, ptr_valbuf_mask, coordinates_id, C_mask );
					internal::Coordinates< reference > coors_mask;
					coors_mask.set( static_cast< void * >( ptr_coorArr_mask->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );
													
					size_t nnz_current_tile = 0;
					
					for( size_t i = lower_bound; i < upper_bound; ++i ) {
					
						// we traverse C_mask to find column indices of nonzero elements
						coors_mask.clear();				
						for( auto k = C_mask_raw.col_start[ i ]; k < C_mask_raw.col_start[ i + 1 ]; ++k ) {						
							const size_t k_col = C_mask_raw.row_index[ k ];
							coors_mask.assign( k_col );													
						}
						// read column indices of nonzeros in coors_mask and copy them into nonzero_indices_mask
						unsigned int nonzero_indices_mask[coors_mask.nonzeroes()]; 						
						coors_mask.packValues( nonzero_indices_mask, 0, nullptr,nullptr );

						//sort of nonzero_indices_mask
						std::sort(nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes());						

						// check column indices of nonzeros int current row i of C = AB
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for(
								auto l = B_raw.col_start[ k_col ];
								l < B_raw.col_start[ k_col + 1 ];
								++l
							) {							
								const size_t l_col = B_raw.row_index[ l ];

								//search column indices that are common to the mask and to C
								// use binary search on sorted nonzero_indices_mask. STL implementation								
								if( std::binary_search( nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes(), l_col, []( const int & a, const int & b ) { return a < b;}) ) {
									coors.assign( l_col );									
								}								
						
								/*
								if(i == l_col)																																
								{
									coors.assign( l_col );
								}
								*/																																
							}						
						}
						nnz_current_tile += coors.nonzeroes();
					}

					// assign corresponding element tile_id of nnz_tiles_C					
					nnz_tiles_C[ tile_id ] = nnz_current_tile;																				

					return SUCCESS;
				};

				internal::Pipeline::prefix_sum_nnz_mxm_type func_prefix_sum = [ &C ]() {
					const auto & nnz_tiles_C = internal::getNonzerosTiles( C );
					auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );					

					prefix_sum_tiles_C[ 0 ] = nnz_tiles_C[ 0 ];
					// TODO: parallel prefix sum
					for( size_t i = 1; i < prefix_sum_tiles_C.size(); i++ ) {
						prefix_sum_tiles_C[ i ] = prefix_sum_tiles_C[ i - 1 ] + nnz_tiles_C[ i ];
					}
					// update nnz of C
					size_t total_nnz = 0;
					for( size_t i = 0; i < nnz_tiles_C.size(); i++ ) {
						total_nnz += nnz_tiles_C[ i ];
					}				

					
					if( grb::capacity( C ) < total_nnz ) {
#ifdef _DEBUG
						std::cerr << "\t not enough capacity to execute requested operation\n";
#endif
						const RC clear_rc = grb::clear( C );
						if( clear_rc != SUCCESS ) {
							return PANIC;
						} else {
							return FAILED;
						}
					}

					// check that the total number of zeros is equal to the capacity of C
					assert( total_nnz == internal::capacity( C ) );

					// this sets nz of C
					internal::setCurrentNonzeroes( C, total_nnz );

					return SUCCESS;
				};

				internal::Pipeline::stage_type func = [ &A, &B, &C, &C_mask, &oper, &monoid, &mulMonoid ]( internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
					(void)pipeline;
					// output matrix C sizes
					const size_t n = grb::ncols( C );
					
					const auto & A_raw = internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B ) ;
					auto & C_raw = internal::getCRS( C );			
					const auto & C_mask_raw = internal::getCRS( C_mask ) ;				
				
					auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );

					// TODO: analytic model for tile size
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;

					size_t previous_nnz;
					size_t current_nnz;

					// special case for first tile
					if( 0 == tile_id ) {
						previous_nnz = 0;
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					} else {
						previous_nnz = prefix_sum_tiles_C[ tile_id - 1 ];
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					}
					(void)current_nnz;

#ifndef NDEBUG
					const size_t nnz_local_old = current_nnz - previous_nnz;
#endif
					size_t nnz_local = previous_nnz;					
					
					//const size_t coordinates_id = grb::config::OMP::current_thread_ID();
					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();		
					
					/*
					std::vector< char > coorArr;
					std::vector< char > coorBuf;
					std::vector< OutputType > valbuf;
					internal::getCoordinatesTiles( coorArr, coorBuf, valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( coorArr.data() ), false, static_cast< void * >( coorBuf.data() ), n );
					
					// coordinates for mask
					std::vector< char > coorArr_mask;
					std::vector< char > coorBuf_mask;
					std::vector< maskType > valbuf_mask;
					internal::getCoordinatesTiles( coorArr_mask, coorBuf_mask, valbuf_mask, coordinates_id, C_mask );
					internal::Coordinates< reference > coors_mask;
					coors_mask.set( static_cast< void * >( coorArr_mask.data() ), false, static_cast< void * >( coorBuf_mask.data() ), n );
					*/	

					std::vector< char >* ptr_coorArr;
					std::vector< char >* ptr_coorBuf;
					std::vector< OutputType >* ptr_valbuf;

					internal::getThreadsBuffers( ptr_coorArr, ptr_coorBuf, ptr_valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( ptr_coorArr->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );
					OutputType * valbuf = ptr_valbuf->data();

					// coordinates for mask
					std::vector< char >* ptr_coorArr_mask;
					std::vector< char >* ptr_coorBuf_mask;
					std::vector< OutputType >* ptr_valbuf_mask;

					internal::getThreadsBuffers( ptr_coorArr_mask, ptr_coorBuf_mask, ptr_valbuf_mask, coordinates_id, C_mask );
					internal::Coordinates< reference > coors_mask;
					coors_mask.set( static_cast< void * >( ptr_coorArr_mask->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );								

					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						
						// we traverse C_mask to find column indices of nonzero elements
						coors_mask.clear();
						for( auto k = C_mask_raw.col_start[ i ]; k < C_mask_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = C_mask_raw.row_index[ k ];
							coors_mask.assign( k_col );							
						}
						// read column indices of nonzeros in coors_mask and copy them into nonzero_indices_mask
						unsigned int nonzero_indices_mask[ coors_mask.nonzeroes() ];
						const size_t offset = 0;
						coors_mask.packValues( nonzero_indices_mask, offset, nullptr, nullptr );

						// sort of nonzero_indices_mask
						std::sort( nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes() );
											
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ];
								l < B_raw.col_start[ k_col + 1 ];
								++l
							) {
								const size_t l_col = B_raw.row_index[ l ];
		#ifdef _DEBUG
								std::cout << "\t A( " << i << ", " << k_col << " ) = "
									<< A_raw.getValue( k,
										mulMonoid.template getIdentity< typename Operator::D1 >() )
									<< " will be multiplied with B( " << k_col << ", " << l_col << " ) = "
									<< B_raw.getValue( l,
										mulMonoid.template getIdentity< typename Operator::D2 >() )
									<< " to accumulate into C( " << i << ", " << l_col << " )\n";
		#endif
								// search column indices that are common to the mask and to C
								//  use binary search on sorted nonzero_indices_mask. STL implementation								
								if( std::binary_search( nonzero_indices_mask, nonzero_indices_mask + coors_mask.nonzeroes(), l_col, []( const int & a, const int & b ) { return a < b;}) ){
									if( !coors.assign( l_col ) ) {
										valbuf[ l_col ] = monoid.template getIdentity< OutputType >();
										(void) grb::apply( valbuf[ l_col ],
											A_raw.getValue( k,
												mulMonoid.template getIdentity< typename Operator::D1 >() ),
											B_raw.getValue( l,
												mulMonoid.template getIdentity< typename Operator::D2 >() ),
											oper );
									} else {
										OutputType temp = monoid.template getIdentity< OutputType >();
										(void) grb::apply( temp,
											A_raw.getValue( k,
												mulMonoid.template getIdentity< typename Operator::D1 >() ),
											B_raw.getValue( l,
												mulMonoid.template getIdentity< typename Operator::D2 >() ),
											oper );
										(void) grb::foldl( valbuf[ l_col ], temp, monoid.getOperator() );
									}
								}											
							}
						}

						for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
							assert( nnz_local < nnz_local_old );
							const size_t j = coors.index( k );
							// update CRS							
							C_raw.setValue( nnz_local, valbuf[ j ] );												
							// update count
							(void) ++nnz_local;
						}					
					}

					return SUCCESS;
				};

				ret = ret ? ret :
							internal::le.addStageLevel3( std::move( func ),
								// name of operation
								internal::Opcode::BLAS3_MXM_GENERIC,
								// size of output matrix
								grb::nrows( C ),
								// size of data type in matrix C
								sizeof( OutputType ),
								// dense_descr
								true,
								// dense_mask
								true,
								// matrices for mxm
								&A, &B, &C, &C_mask, std::move( func_count_nonzeros ), std::move( func_prefix_sum ) );
			}
			return ret;
		}

		template< bool allow_void,
			Descriptor descr,
			class MulMonoid,
			typename OutputType,
			typename InputType1,
			typename InputType2,
			typename RIT,
			typename CIT,
			typename NIT,
			class Operator,
			class Monoid >
		RC mxm_generic( Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
			const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
			const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
			const Operator & oper,
			const Monoid & monoid,
			const MulMonoid & mulMonoid,
			const Phase & phase,
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					grb::is_operator< Operator >::value && grb::is_monoid< Monoid >::value,
				void >::type * const = nullptr ) {

			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)" );

#ifdef _DEBUG
			std::cout << "In grb::internal::mxm_generic (nonblocking, unmasked)\n";
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// get whether we are required to stick to CRS
			constexpr bool crs_only = descr & descriptors::force_row_major;

			// static checks
			static_assert( ! ( crs_only && trans_left ),
				"Cannot (presently) transpose A "
				"and force the use of CRS" );
			static_assert( ! ( crs_only && trans_right ),
				"Cannot (presently) transpose B "
				"and force the use of CRS" );

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = ! trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t k = ! trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t k_B = ! trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = ! trans_right ? grb::ncols( B ) : grb::nrows( B );
			assert( phase != TRY );

			if( m != m_A || k != k_B || n != n_B ) {
				return MISMATCH;
			}

			const auto & A_raw = ! trans_left ? internal::getCRS( A ) : internal::getCCS( A );
			const auto & B_raw = ! trans_right ? internal::getCRS( B ) : internal::getCCS( B );
			auto & C_raw = internal::getCRS( C );
			auto & CCS_raw = internal::getCCS( C );

			char * arr = nullptr;
			char * buf = nullptr;
			OutputType * valbuf = nullptr;
			internal::getMatrixBuffers( arr, buf, valbuf, 1, C );
			//config::NonzeroIndexType * C_col_index = internal::template
			//	getReferenceBuffer< typename config::NonzeroIndexType >( n + 1 );

			// initialisations
			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n );

			if( ! crs_only ) {
#ifdef _H_GRB_NONBLOCKING_OMP_BLAS3
#pragma omp parallel
				{
					size_t start, end;
					config::OMP::localRange( start, end, 0, n + 1 );
#else
				const size_t start = 0;
				const size_t end = n + 1;
#endif
					for( size_t j = start; j < end; ++j ) {
						CCS_raw.col_start[ j ] = 0;
					}
#ifdef _H_GRB_NONBLOCKING_OMP_BLAS3
				}
#endif
			}
			// end initialisations

			// symbolic phase (counting sort, step 1)
			size_t nzc = 0; // output nonzero count
			if( crs_only && phase == RESIZE ) {
				// we are using an auxialiary CRS that we cannot resize ourselves
				// instead, we update the offset array only
				C_raw.col_start[ 0 ] = 0;
			}
			// if crs_only, then the below implements its resize phase
			// if not crs_only, then the below is both crucial for the resize phase,
			// as well as for enabling the insertions of output values in the output CCS
			// this step is meant to computing the total number of zeros in the output C
			if( ( crs_only && phase == RESIZE ) || ! crs_only ) {
				for( size_t i = 0; i < m; ++i ) {					
					coors.clear();
					for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
							const size_t l_col = B_raw.row_index[ l ];
							if( ! coors.assign( l_col ) ) {
								(void)++nzc;								
							}
						}
					}			

					// update CRS -> col_start
					C_raw.col_start[ i + 1 ] = nzc;
				}
			}
						
			if( phase == RESIZE ) {
				if( ! crs_only ) {
					// do final resize
					// this will update cap of C to nzc
					const RC ret = grb::resize( C, nzc );
					std::cout << "matrix ID = " << grb::getID( C ) << ", internal::getNonzeroCapacity (after resize mxm)= " << grb::capacity(C) << std::endl;					
#ifndef NDEBUG
					const size_t old_nzc = nzc;
#endif				
					// set nzc to zero
					nzc= 0;
		
					//once C holds enough capacity to store nzc, we modify the elements of the arrays CRS -> row_indices and col_start
					// this basically consists of repeating the resize step
					//std::cout << "(mxm) matrix ID = " << grb::getID( C ) << ", internal::getNonzeroCapacity (after resize mxm)= " << internal::getNonzeroCapacity( C ) << std::endl;					

					for( size_t i = 0; i < m; ++i ) {
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
								const size_t l_col = B_raw.row_index[ l ];
								coors.assign( l_col );
							}							
						}

						for( size_t k = 0; k < coors.nonzeroes(); k++ ) {
							assert( nzc < old_nzc );
							const size_t j = coors.index( k );
							// update CRS -> row_index
							C_raw.row_index[ nzc ] = j;																				
							(void)++nzc;
						}
						// update CRS -> col_start
						//C_raw.col_start[ i + 1 ] = nzc;
					}

#ifndef NDEBUG
					assert( nzc == old_nzc );
#endif

					/*
					std::cout << "row pointers of C " << std::endl;
					for( size_t i = 0; i < grb::nrows( C ) + 1; i++ ) {
					    std::cout << C_raw.col_start[ i ] << ",";
					}
					std::cout << std::endl;

					std::cout << "col indices C " << std::endl;
					for( size_t i = 0; i < internal::getNonzeroCapacity( C ); i++ ) {
					    std::cout << C_raw.row_index[ i ] << ",";
					}
					std::cout << std::endl;
					*/
					return ret;

				} else {
					// we are using an auxiliary CRS that we cannot resize
					// instead, we updated the offset array in the above and can now exit
					return SUCCESS;
				}
			}

			// computational phase
			assert( phase == EXECUTE );
			if( grb::capacity( C ) < nzc ) {
#ifdef _DEBUG
				std::cerr << "\t not enough capacity to execute requested operation\n";
#endif
				const RC clear_rc = grb::clear( C );
				if( clear_rc != SUCCESS ) {
					return PANIC;
				} else {
					return FAILED;
				}
			}

			RC ret = SUCCESS;

			if( phase == EXECUTE ) {
				
				// lambda  function to count the nnz in each tile
				internal::Pipeline::count_nnz_local_type func_count_nonzeros = [ &A, &B, &C]( const size_t lower_bound, const size_t upper_bound) {
					
					// output matrix C sizes
					// ncols(C) = ncols(B)
					const size_t n = grb::ncols( B );					
					const auto & A_raw = internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B ) ;					
					auto & nnz_tiles_C = internal::getNonzerosTiles( C );	

					// we retrieve information about the tiles
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;				
					
					
					/*
					// THIS IMPLEMENTATION OF COORS WORKS BUT ALLOCATES MEMORY AT RUN TIME
					const size_t coorArr_elements = internal::Coordinates< reference >::arraySize( n )* internal::SizeOf< OutputType >::value;
					const size_t coorBuf_elements = internal::Coordinates< reference >::bufferSize( n )*internal::SizeOf< OutputType >::value;
					char arr[ coorArr_elements ];					
					char buf[ coorBuf_elements ];

					internal::Coordinates< reference > coors;
					coors.set( arr, false, buf, n );	
					*/

					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();										

					/*
					// this allocates memory dynamically
					std::vector< char > coorArr;
					std::vector< char > coorBuf;
					std::vector< OutputType > valbuf;

					internal::getCoordinatesTiles( coorArr, coorBuf, valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( coorArr.data() ), false, static_cast< void * >( coorBuf.data() ), n );
					*/
					
					std::vector< char >* ptr_coorArr;
					std::vector< char >* ptr_coorBuf;
					std::vector< OutputType >* ptr_valbuf;

					internal::getThreadsBuffers( ptr_coorArr, ptr_coorBuf, ptr_valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( ptr_coorArr->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );					

					/*
					// using the SPA from memory allocated using NUMA
					char * arr = nullptr;
					char * buf = nullptr;
					OutputType * valbuf = nullptr;
					internal::getThreadsBuffers( arr, buf, valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( arr, false, valbuf, n );
					*/
				
					size_t nnz_current_tile = 0;				

					for( size_t i = lower_bound; i < upper_bound; ++i ) {											
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
								const size_t l_col = B_raw.row_index[ l ];
								coors.assign( l_col );
							}
						}
						nnz_current_tile += coors.nonzeroes();													
					}

					// assign corresponding element tile_id of nnz_tiles_C					
					nnz_tiles_C[ tile_id ] = nnz_current_tile;																				

					return SUCCESS;
				};

				// lambda function to compute the prefix sum of local nnz
				internal::Pipeline::prefix_sum_nnz_mxm_type func_prefix_sum = [ &C ]() {			
					const auto & nnz_tiles_C = internal::getNonzerosTiles( C );
					auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );

					/*
					std::cout << "(mxm prefix) matrix ID = " << grb::getID( C ) << std::endl;
					std::cout << "vector nnz_tiles= ";
					for(auto val : nnz_tiles_C)
					{
						std::cout << val << ", ";
					}
					std::cout << std::endl;
					*/					

					prefix_sum_tiles_C[ 0 ] = nnz_tiles_C[ 0 ];
					// TODO: parallel prefix sum
					for( size_t i = 1; i < prefix_sum_tiles_C.size(); i++ ) {
						prefix_sum_tiles_C[ i ] = prefix_sum_tiles_C[ i - 1 ] + nnz_tiles_C[ i ];
					}
					
					// update nnz of C
					size_t total_nnz = 0;
					for( size_t i = 0; i < nnz_tiles_C.size(); i++ ) {
						total_nnz += nnz_tiles_C[ i ];
					}				
										
					internal::setCurrentNonzeroes( C, total_nnz );
					internal::setStatusNnzTiles( C, true );
					internal::setStatusPrefixTiles( C, true );

					//std::cout << "internal::setCurrentNonzeroes(...), total_nnz = " << total_nnz << std::endl;
				
					/*
					if( tile_id == 0)
					{
					    std::cout << "values of matrix after prefix sum MXM" << std::endl;
					    std::cout << "****** matrix C" << std::endl;
					    std::cout << "nrows(C) = " << grb::nrows( C ) << ", ncols(C) = " << grb::ncols( C );
					    std::cout << "internal::getCurrentNonzeroes(C) = " << internal::getCurrentNonzeroes( C );
					    std::cout << ", number of tiles = " << num_tiles << ", total_nnz = " << total_nnz;
					    std::cout << ", lower bound = " << lower_bound << ", upper bound = " << upper_bound;
					    std::cout << std::endl;


					    auto & C_raw = internal::getCRS( C );
					    std::cout << "row pointers of C: \n" ;
					    for( size_t i = 0; i < grb::nrows( C ) + 1; i++ ) {
					        std::cout << C_raw.col_start[ i ] << ",";
					    }
					    std::cout << std::endl;

					    std::cout << "values of C: \n";
					    for( size_t i = 0; i < internal::getCurrentNonzeroes( C ); i++ ) {
					        std::cout << C_raw.values[ i ] << ",";
					    }
					    std::cout << std::endl;

					    std::cout << "col indices of C: \n";
					    for( size_t i = 0; i < internal::getCurrentNonzeroes( C ); i++ ) {
					        std::cout << C_raw.row_index[ i ] << ",";
					    }
					    std::cout << std::endl;
					}
					*/					

					return SUCCESS;
				};

				// lambda function that corresponds to the actual computational phase
				internal::Pipeline::stage_type func = [ &A, &B, &C, &oper, &monoid, &mulMonoid ](
														  internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
					(void)pipeline;
					// output matrix C sizes
					const size_t n = grb::ncols( C );
					
					const auto & A_raw = internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B ) ;
					auto & C_raw = internal::getCRS( C );							
				
					auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );

					// TODO: analytic model for tile size
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;

					size_t previous_nnz;
					size_t current_nnz;

					// special case for first tile
					if( 0 == tile_id ) {
						previous_nnz = 0;
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					} else {
						previous_nnz = prefix_sum_tiles_C[ tile_id - 1 ];
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					}
					(void)current_nnz;

#ifndef NDEBUG
					const size_t nnz_local_old = current_nnz - previous_nnz;
#endif
					size_t nnz_local = previous_nnz;					
					
					//const size_t coordinates_id = grb::config::OMP::current_thread_ID();
					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();					

					std::vector< char >* ptr_coorArr;
					std::vector< char >* ptr_coorBuf;
					std::vector< OutputType >* ptr_valbuf;

					internal::getThreadsBuffers( ptr_coorArr, ptr_coorBuf, ptr_valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( ptr_coorArr->data() ), false, static_cast< void * > ( ptr_coorBuf->data() ), n );
					OutputType * valbuf = ptr_valbuf->data();

					/*
					// THIS IMPLEMENTATION OF COORS WORKS BUT ALLOCATES MEMORY AT RUN TIME
					const size_t coorArr_elements = internal::Coordinates< reference >::arraySize( n )*internal::SizeOf< OutputType >::value;
					char arr[ coorArr_elements ];
					const size_t coorBuf_elements = internal::Coordinates< reference >::bufferSize( n )*internal::SizeOf< OutputType >::value;
					char buf[ coorBuf_elements ];
					
					const size_t valbuf_elements = n * internal::SizeOf< OutputType >::value;									
					OutputType valbuf[valbuf_elements ];

					internal::Coordinates< reference > coors;
					coors.set(arr, false, buf, n );	
					*/

					// Computational phase here
					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
								const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG
								std::cout << "\t A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() )
										  << " will be multiplied with B( " << k_col << ", " << l_col << " ) = " << B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() )
										  << " to accumulate into C( " << i << ", " << l_col << " )\n";
#endif
								if( ! coors.assign( l_col ) ) {
									valbuf[ l_col ] = monoid.template getIdentity< OutputType >();
									(void)grb::apply( valbuf[ l_col ], A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ),
										B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
								} else {
									OutputType temp = monoid.template getIdentity< OutputType >();
									(void)grb::apply( temp, A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ),
										B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
									(void)grb::foldl( valbuf[ l_col ], temp, monoid.getOperator() );
								}
							}
						}

						for( size_t k = 0; k < coors.nonzeroes(); k++ ) {
							assert( nnz_local < nnz_local_old );
							const size_t j = coors.index( k );														
							// update CRS -> values array
							C_raw.setValue( nnz_local, valbuf[ j ] );							
							// update count
							(void)++nnz_local;
						}
						// C_raw.col_start[ i + 1 ] = nnz_local;
					}

					/*
					#pragma omp critical
					{
					    if(tile_id == 0)
					    {
					        std::cout << "***** MXM_EXECUTE_FINISHED *****" << std::endl;
					    }
					    //int thread_id = omp_get_thread_num();
					//	//std::cout << "BLAS3_MXM_GENERIC_EXECUTE has finished, ";
					    std::cout << "TILE ID= " << tile_id << ", [" << lower_bound << ", " << upper_bound << "]" << ", nnz local = " << nnz_tiles_C[tile_id];
					    std::cout << ", previous_nnz = "<< previous_nnz << ", current_nnz = "<< current_nnz << " (" << prefix_sum_tiles_C[tile_id] << ")"<< std::endl;
					}
					*/

					return SUCCESS;
				}; // end lambda function

				ret = ret ? ret :
							internal::le.addStageLevel3( std::move( func ),
								// name of operation
								internal::Opcode::BLAS3_MXM_GENERIC,
								// size of output matrix
								grb::nrows( C ),
								// size of data type in matrix C
								sizeof( OutputType ),
								// dense_descr
								true,
								// dense_mask
								true,
								// matrices for mxm
								&A, &B, &C, nullptr, std::move( func_count_nonzeros ), std::move( func_prefix_sum ) );
			}

			return ret;
		}

	} // namespace internal


	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType1, typename InputType2,
		typename RIT, typename CIT, typename NIT, typename maskType,
		class Semiring
	>
	RC mxm_masked(
		Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
		const Matrix< maskType, nonblocking, RIT, CIT, NIT > & C_mask,
		const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
		const Semiring &ring = Semiring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< maskType >::value &&
			grb::is_semiring< Semiring >::value,
		void >::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D1, InputType1 >::value
			), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Semiring::D4, OutputType >::value
			), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "In grb::mxm (reference, unmasked, semiring)\n";
#endif

		return internal::mxm_masked_generic< true, descr, true >(
			C, C_mask, A, B,
			ring.getMultiplicativeOperator(),
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeMonoid(),
			phase
		);
	}

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, typename RIT, typename CIT, typename NIT, class Semiring >
	RC mxm( Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
		const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
		const Semiring & ring = Semiring(),
		const Phase & phase = EXECUTE,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_semiring< Semiring >::value,
			void >::type * const = nullptr ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D1, InputType1 >::value ), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the given operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Semiring::D4, OutputType >::value ), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given operator" );

#ifdef _DEBUG
		std::cout << "In grb::mxm (nonblocking, unmasked, semiring)\n";
#endif
		/*
		if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
		    std::cerr << "Warning: mxm (nonblocking, unmasked, semiring) currently "
		              << "delegates to a blocking implementation\n"
		              << "         Further similar such warnings will be suppressed.\n";
		    internal::NONBLOCKING::warn_if_not_native = false;
		}
		*/
		// grb::internal::le.execution();
		return internal::mxm_generic< true, descr >( C, A, B, ring.getMultiplicativeOperator(), ring.getAdditiveMonoid(), ring.getMultiplicativeMonoid(), phase );
	}

	template< Descriptor descr = grb::descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, typename RIT, typename CIT, typename NIT, class Operator, class Monoid >
	RC mxm( Matrix< OutputType, nonblocking, RIT, CIT, NIT > & C,
		const Matrix< InputType1, nonblocking, RIT, CIT, NIT > & A,
		const Matrix< InputType2, nonblocking, RIT, CIT, NIT > & B,
		const Monoid & addM,
		const Operator & mulOp,
		const Phase & phase = EXECUTE,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< Operator >::value &&
				grb::is_monoid< Monoid >::value,
			void >::type * const = nullptr ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::mxm",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::mxm",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::mxm",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D1, typename Operator::D3 >::value ), "grb::mxm",
			"the output domain of the multiplication operator does not match the "
			"first domain of the given addition monoid" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D2, OutputType >::value ), "grb::mxm",
			"the second domain of the given addition monoid does not match the "
			"type of the output matrix C" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Monoid::D3, OutputType >::value ), "grb::mxm",
			"the output type of the given addition monoid does not match the type "
			"of the output matrix C" );
		static_assert( ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
			"grb::mxm: the operator-monoid version of mxm cannot be used if either "
			"of the input matrices is a pattern matrix (of type void)" );
		/*
		if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
		    std::cerr << "Warning: mxm (nonblocking, unmasked, monoid-op) currently "
		              << "delegates to a blocking implementation\n"
		              << "         Further similar such warnings will be suppressed.\n";
		    internal::NONBLOCKING::warn_if_not_native = false;
		}
		*/

		// grb::internal::le.execution();
		return internal::mxm_generic< false, descr >( C, A, B, mulOp, addM, Monoid(), phase );
	}

	namespace internal {

		template< Descriptor descr = descriptors::no_operation, bool matrix_is_void, typename OutputType, typename InputType1, typename InputType2, typename InputType3, typename Coords >
		RC matrix_zip_generic( Matrix< OutputType, nonblocking > & A,
			const Vector< InputType1, nonblocking, Coords > & x,
			const Vector< InputType2, nonblocking, Coords > & y,
			const Vector< InputType3, nonblocking, Coords > & z,
			const Phase & phase ) {
			if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
				std::cerr << "Warning: zip (matrix<-vector<-vector<-vector, nonblocking) "
						  << "currently delegates to a blocking implementation.\n"
						  << "         Further similar such warnings will be suppressed.\n";
				internal::NONBLOCKING::warn_if_not_native = false;
			}
			/*
			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return matrix_zip_generic< descr, matrix_is_void, OutputType, InputType1, InputType2, InputType3, Coords >(
			    getRefMatrix( A ), getRefVector( x ), getRefVector( y ), getRefVector( z ), phase );
			*/
			(void)A;
			(void)x;
			(void)y;
			(void)z;
			(void)phase;
			return UNSUPPORTED;
		}

	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, typename InputType3, typename Coords >
	RC zip( Matrix< OutputType, nonblocking > & A,
		const Vector< InputType1, nonblocking, Coords > & x,
		const Vector< InputType2, nonblocking, Coords > & y,
		const Vector< InputType3, nonblocking, Coords > & z,
		const Phase & phase = EXECUTE ) {
		/*
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType1 >::value,
		    "grb::zip (two vectors to matrix) called "
		    "using non-integral left-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType2 >::value,
		    "grb::zip (two vectors to matrix) called "
		    "using non-integral right-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_same< OutputType, InputType3 >::value,
		    "grb::zip (two vectors to matrix) called "
		    "with differing vector nonzero and output matrix domains" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
		    return ret;
		}
		if( n != grb::size( y ) ) {
		    return MISMATCH;
		}
		if( n != grb::size( z ) ) {
		    return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
		    return ILLEGAL;
		}
		if( nz != grb::nnz( z ) ) {
		    return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, false >( A, x, y, z, phase );
		*/
		(void)A;
		(void)x;
		(void)y;
		(void)z;
		(void)phase;
		return UNSUPPORTED;
	}

	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename Coords >
	RC zip( Matrix< void, nonblocking > & A, const Vector< InputType1, nonblocking, Coords > & x, const Vector< InputType2, nonblocking, Coords > & y, const Phase & phase = EXECUTE ) {
		/*
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType1 >::value,
		    "grb::zip (two vectors to void matrix) called using non-integral "
		    "left-hand vector elements" );
		static_assert( ! ( descr & descriptors::no_casting ) || std::is_integral< InputType2 >::value,
		    "grb::zip (two vectors to void matrix) called using non-integral "
		    "right-hand vector elements" );

		const size_t n = grb::size( x );
		const size_t nz = grb::nnz( x );
		const RC ret = grb::clear( A );
		if( ret != SUCCESS ) {
		    return ret;
		}
		if( n != grb::size( y ) ) {
		    return MISMATCH;
		}
		if( nz != grb::nnz( y ) ) {
		    return ILLEGAL;
		}

		return internal::matrix_zip_generic< descr, true >( A, x, y, x, phase );
		*/

		(void)A;
		(void)x;
		(void)y;
		(void)phase;
		return UNSUPPORTED;
	}

	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename OutputType, typename Coords, class Operator >
	RC outer( Matrix< OutputType, nonblocking > & A,
		const Vector< InputType1, nonblocking, Coords > & u,
		const Vector< InputType2, nonblocking, Coords > & v,
		const Operator & mul = Operator(),
		const Phase & phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< Operator >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< OutputType >::value,
			void >::type * const = nullptr ) {
		/*
		if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
		    std::cerr << "Warning: outer (nonblocking) currently delegates to a "
		              << "blocking implementation.\n"
		              << "         Further similar such warnings will be suppressed.\n";
		    internal::NONBLOCKING::warn_if_not_native = false;
		}

		// nonblocking execution is not supported
		// first, execute any computation that is not completed
		internal::le.execution();

		// second, delegate to the reference backend
		return outer< descr, InputType1, InputType2, OutputType, Coords, Operator >( internal::getRefMatrix( A ), internal::getRefVector( u ), internal::getRefVector( v ), mul, phase );
		*/
		(void)A;
		(void)u;
		(void)v;
		(void)mul;
		(void)phase;
		return UNSUPPORTED;
	}

	namespace internal {

		template< bool allow_void, Descriptor descr, class MulMonoid, typename OutputType, typename InputType1, typename InputType2, class Operator >
		RC eWiseApply_matrix_generic( Matrix< OutputType, nonblocking > & C,
			const Matrix< InputType1, nonblocking > & A,
			const Matrix< InputType2, nonblocking > & B,
			const Operator & oper,
			const MulMonoid & mulMonoid,
			const Phase & phase,
			const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value &&
					grb::is_operator< Operator >::value,
				void >::type * const = nullptr ) {

			// le.execution();
			assert( ! ( descr & descriptors::force_row_major ) );
			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::internal::eWiseApply_matrix_generic: the non-monoid version of "
				"elementwise mxm can only be used if neither of the input matrices "
				"is a pattern matrix (of type void)" );
			assert( phase != TRY );

#ifdef _DEBUG
			std::cout << "In grb::internal::eWiseApply_matrix_generic, nonblocking\n";
#endif

			// get whether the matrices should be transposed prior to execution
			constexpr bool trans_left = descr & descriptors::transpose_left;
			constexpr bool trans_right = descr & descriptors::transpose_right;

			// run-time checks
			const size_t m = grb::nrows( C );
			const size_t n = grb::ncols( C );
			const size_t m_A = ! trans_left ? grb::nrows( A ) : grb::ncols( A );
			const size_t n_A = ! trans_left ? grb::ncols( A ) : grb::nrows( A );
			const size_t m_B = ! trans_right ? grb::nrows( B ) : grb::ncols( B );
			const size_t n_B = ! trans_right ? grb::ncols( B ) : grb::nrows( B );

			if( m != m_A || m != m_B || n != n_A || n != n_B ) {
				return MISMATCH;
			}

			const auto & A_raw = ! trans_left ? internal::getCRS( A ) : internal::getCCS( A );
			const auto & B_raw = ! trans_right ? internal::getCRS( B ) : internal::getCCS( B );
			auto & C_raw = internal::getCRS( C );

#ifdef _DEBUG
			std::cout << "\t\t A offset array = { ";
			for( size_t i = 0; i <= m_A; ++i ) {
				std::cout << A_raw.col_start[ i ] << " ";
			}
			std::cout << "}\n";
			for( size_t i = 0; i < m_A; ++i ) {
				for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
					std::cout << "\t\t ( " << i << ", " << A_raw.row_index[ k ] << " ) = " << A_raw.getPrintValue( k ) << "\n";
				}
			}
			std::cout << "\t\t B offset array = { ";
			for( size_t j = 0; j <= m_B; ++j ) {
				std::cout << B_raw.col_start[ j ] << " ";
			}
			std::cout << "}\n";
			for( size_t j = 0; j < m_B; ++j ) {
				for( size_t k = B_raw.col_start[ j ]; k < B_raw.col_start[ j + 1 ]; ++k ) {
					std::cout << "\t\t ( " << B_raw.row_index[ k ] << ", " << j << " ) = " << B_raw.getPrintValue( k ) << "\n";
				}
			}
#endif

			// retrieve buffers
			char *arr1, *arr2, *arr3, *buf1, *buf2, *buf3;
			arr1 = arr2 = buf1 = buf2 = nullptr;
			InputType1 * vbuf1 = nullptr;
			InputType2 * vbuf2 = nullptr;
			OutputType * valbuf = nullptr;
			internal::getMatrixBuffers( arr1, buf1, vbuf1, 1, A );
			internal::getMatrixBuffers( arr2, buf2, vbuf2, 1, B );
			internal::getMatrixBuffers( arr3, buf3, valbuf, 1, C );
			// end buffer retrieval

			// initialisations
			internal::Coordinates< reference > coors1, coors2;
			coors1.set( arr1, false, buf1, n );
			coors2.set( arr2, false, buf2, n );
			// end initialisations

			// nonzero count
			size_t nzc = 0;

			// symbolic phase
			if( phase == RESIZE ) {
				// std::cout << "***** EWISE_RESIZE *****" << std::endl;
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
					}
					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							(void)++nzc;
						}
					}
				}

				const RC ret = grb::resize( C, nzc );

				nzc = 0;

				//std::cout << "(eWise) matrix ID = " << grb::getID( C ) << ", internal::getNonzeroCapacity (after resize eWise)= " << internal::getNonzeroCapacity(C) << std::endl;

				// this is to update the arrays row_index and col_start CRS of C
				for( size_t i = 0; i < m; ++i ) {
					coors1.clear();
					coors2.clear();
					for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						coors1.assign( k_col );
					}

					for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
						const size_t l_col = B_raw.row_index[ l ];
						if( coors1.assigned( l_col ) ) {
							coors2.assign( l_col );
						}
					}

					for( size_t k = 0; k < coors2.nonzeroes(); ++k ) {
						const size_t j = coors2.index( k );
						// update CRS
						C_raw.row_index[ nzc ] = j;
						// update count
						(void)++nzc;
					}
					C_raw.col_start[ i + 1 ] = nzc;
				}

				if( ret != SUCCESS ) {
					return ret;
				}
			}

			if( EXECUTE == phase ) {
				// std::cout << "capacity of C before counting phase: " << internal::getNonzeroCapacity( C ) << std::endl;
				RC ret = SUCCESS;

				// lambda  function to count the nnz in each tile
				internal::Pipeline::count_nnz_local_type func_count_nonzeros = [ &A, &B, &C, &mulMonoid, &oper ]( const size_t lower_bound, const size_t upper_bound ) {									
					
					const auto & A_raw =internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B );					

					auto & nnz_tiles_C = internal::getNonzerosTiles( C );	
					//const size_t m = grb::nrows( C );
					const size_t n = grb::ncols( C );			

					// we retrieve information about the tiles
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;
					
					/*
					// THIS IMPLEMENTATION OF COORS WORKS BUT ALLOCATES MEMORY AT RUN TIME
					const size_t coorArr_elements = internal::Coordinates< reference >::arraySize( n ) * internal::SizeOf< InputType1 >::value;		
					char arr1[ coorArr_elements ];										
					const size_t coorBuf_elements = internal::Coordinates< reference >::bufferSize( n )* internal::SizeOf< InputType1 >::value;
					char buf1[ coorBuf_elements ];
					internal::Coordinates< reference > coors1;
					coors1.set( arr1, false, buf1, n );
					*/

					//const size_t coordinates_id = grb::config::OMP::current_thread_ID();
					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();
					std::vector< char > arr1;
					std::vector< char > buf1;
					std::vector< OutputType > valbuf1;
					internal::getCoordinatesTiles( arr1, buf1, valbuf1, coordinates_id, C );
					internal::Coordinates< reference > coors1;
					coors1.set( static_cast< void * >( arr1.data() ), false, static_cast< void * >( buf1.data() ), n );
					
					/*
					// we retrive the coordinates that are already store in matrices. OLD
					const size_t num_threads = omp_get_num_threads();
					const size_t num_tiles = nnz_tiles_C.size();
					const size_t coordinates_id = ( num_threads < num_tiles ) ? omp_get_thread_num() : tile_id;

					std::vector< char > coorArr_A, coorBuf_A;
					std::vector< InputType1 > valbuf_A;

					internal::getCoordinatesTiles( coorArr_A, coorBuf_A, valbuf_A, coordinates_id, A );

					internal::Coordinates< reference > coors1;
					coors1.set( static_cast< void * >( coorArr_A.data() ), false, static_cast< void * >( coorBuf_A.data() ), n );
					*/

					size_t nnz_current_tile = 0;

					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						coors1.clear();
						for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							coors1.assign( k_col );
						}
						for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
							const size_t l_col = B_raw.row_index[ l ];
							if( coors1.assigned( l_col ) ) {
								(void)++nnz_current_tile;
							}
						}
					}

					// assign number of nonzeros for local tile
					nnz_tiles_C[ tile_id ] = nnz_current_tile;

					/*
					#pragma omp critical
					{
					    if(0 == tile_id)
					    {
					        std::cout << "-------COUNTING PHASE-------" << std::endl;
					    }
					    std::cout << "[" << lower_bound << ", " << upper_bound << "]" << ", tile_id = " << tile_id;
					    std::cout << ", nnz = " << nnz_current_tile << std::endl;
					}
					*/
					return SUCCESS;
				};

				// lambda function to compute the prefix sum of local nnz
				internal::Pipeline::prefix_sum_nnz_mxm_type func_prefix_sum = [ &C ]( ) {
					
					const auto & nnz_tiles_C = internal::getNonzerosTiles( C );
					auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );

					/*
					std::cout << "(eWise prefix) matrix ID = " << grb::getID( C ) << std::endl;
					std::cout << "vector nnz_tiles= ";
					for(auto val : nnz_tiles_C)
					{
						std::cout << val << ", ";
					}
					std::cout << std::endl;
					*/

					// when the prefix sum is called, nnz_tiles_C has been completely computed.
					prefix_sum_tiles_C[ 0 ] = nnz_tiles_C[ 0 ];
					// TODO: parallel prefix sum
					for( size_t i = 1; i < prefix_sum_tiles_C.size(); i++ ) {
						prefix_sum_tiles_C[ i ] = prefix_sum_tiles_C[ i - 1 ] + nnz_tiles_C[ i ];
					}

					// Then we update the current number of nonzeros at this point
					size_t total_nnz = 0;
					for( auto nnz_local : nnz_tiles_C ) {
						total_nnz += nnz_local;
					}

					//std::cout << "matrix ID = " << grb::getID( C );
					//std::cout << ", internal::setCurrentNonzeroes(...), total_nnz = " << total_nnz << std::endl;

					// set final number of nonzeroes in output matrix
					internal::setCurrentNonzeroes( C, total_nnz );
					//std::cout << "(eWise prefix) internal::setCurrentNonzeroes(...), total_nnz = " << total_nnz << std::endl;
					
					/*
					std::cout << "-------PREFIX SUM -------" << std::endl;
					std::cout << "total nnz = " << total_nnz << std::endl;
					for (size_t i = 0; i < nnz_tiles_C.size(); i++)
					{
					    std::cout << "tile_id = " << i << ", nnz_tile = " << nnz_tiles_C[ i ] << std::endl;
					}
					*/

					return SUCCESS;
				};

				internal::Pipeline::stage_type func = [ &A, &B, &C, &oper, &mulMonoid ](
														  internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
					(void)pipeline;					

					const auto & A_raw =internal::getCRS( A );
					const auto & B_raw = internal::getCRS( B );
					auto & C_raw = internal::getCRS( C );

					// retrieve number of columns of C					
					const size_t n = grb::ncols( C );

					const auto & prefix_sum_tiles_C = internal::getPrefixSumTiles( C );

					// TODO: analytic model for tile size
					const size_t tile_size = grb::internal::NONBLOCKING::manualFixedTileSize();
					const size_t tile_id = lower_bound / tile_size;

					size_t previous_nnz;
					size_t current_nnz;

					// special case for first tile
					if( 0 == tile_id ) {
						previous_nnz = 0;
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					} else {
						previous_nnz = prefix_sum_tiles_C[ tile_id - 1 ];
						current_nnz = prefix_sum_tiles_C[ tile_id ];
					}

					(void)current_nnz;
					size_t nnz_current_tile = previous_nnz;

					/*
					// we retrive the coordinates that are already store in matrices
					const size_t num_threads = omp_get_num_threads();
					const size_t num_tiles = prefix_sum_tiles_C.size();
					const size_t coordinates_id = ( num_threads < num_tiles ) ? omp_get_thread_num() : tile_id;

					std::vector< char > coorArr_A, coorArr_B, coorArr;
					std::vector< char > coorBuf_A, coorBuf_B, coorBuf;
					std::vector< InputType1 > valbuf_A;
					std::vector< InputType2 > valbuf_B;
					std::vector< OutputType > valbuf;

					internal::getCoordinatesTiles( coorArr_A, coorBuf_A, valbuf_A, coordinates_id, A );
					internal::getCoordinatesTiles( coorArr_B, coorBuf_B, valbuf_B, coordinates_id, B );
					internal::getCoordinatesTiles( coorArr, coorBuf, valbuf, coordinates_id, C );

					internal::Coordinates< reference > coors1, coors2;
					coors1.set( static_cast< void * >( coorArr_A.data() ), false, static_cast< void * >( coorBuf_A.data() ), n );
					coors2.set( static_cast< void * >( coorArr_B.data() ), false, static_cast< void * >( coorBuf_B.data() ), n );
					*/
					
					/*
					// THIS IMPLEMENTATION OF COORS WORKS BUT ALLOCATES MEMORY AT RUN TIME
					const size_t coorArr_elements = internal::Coordinates< reference >::arraySize( n ) * internal::SizeOf< InputType1 >::value;					
					char arr1[ coorArr_elements ];					
					
					const size_t coorBuf_elements = internal::Coordinates< reference >::bufferSize( n ) *  internal::SizeOf< InputType1 >::value;
					char buf1[ coorBuf_elements ];

					internal::Coordinates< reference > coors1;
					coors1.set( arr1, false, buf1, n );

					const size_t coorArr_elements2 = internal::Coordinates< reference >::arraySize( n ) * internal::SizeOf< InputType2 >::value;
					char arr2[ coorArr_elements2 ];

					const size_t coorBuf_elements2 = internal::Coordinates< reference >::bufferSize( n ) * internal::SizeOf< InputType2 >::value;
					char buf2[ coorBuf_elements2 ];

					internal::Coordinates< reference > coors2;
					coors2.set( arr2, false, buf2, n );

					const size_t valbuf_elements = n * internal::SizeOf< OutputType >::value;
					std::vector< OutputType > valbuf( valbuf_elements );
					//OutputType valbuf[ valbuf_elements ];		
					*/
					
					//const size_t coordinates_id = grb::config::OMP::current_thread_ID();					
					const unsigned int coordinates_id =
						omp_get_thread_num() * config::CACHE_LINE_SIZE::value();
					std::vector< char > arr1;
					std::vector< char > buf1;
					std::vector< InputType1 > valbuf1;
					internal::getCoordinatesTiles( arr1, buf1, valbuf1, coordinates_id, A );
					internal::Coordinates< reference > coors1;
					coors1.set( static_cast< void * >( arr1.data() ), false, static_cast< void * >( buf1.data() ), n );			

					std::vector< char > arr2;
					std::vector< char > buf2;
					std::vector< InputType2 > valbuf2;
					internal::getCoordinatesTiles( arr2, buf2, valbuf2, coordinates_id, B );
					internal::Coordinates< reference > coors2;
					coors2.set( static_cast< void * >( arr2.data() ), false, static_cast< void * >( buf2.data() ), n );	

					std::vector< char > arr;
					std::vector< char > buf;
					std::vector< OutputType > valbuf;
					internal::getCoordinatesTiles( arr, buf, valbuf, coordinates_id, C );
					internal::Coordinates< reference > coors;
					coors.set( static_cast< void * >( arr.data() ), false, static_cast< void * >( buf.data() ), n );	

					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						coors1.clear();
						coors2.clear();
#ifdef _DEBUG
						std::cout << "\t The elements ";
#endif
						for( size_t k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							coors1.assign( k_col );
							valbuf[ k_col ] = A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() );
#ifdef _DEBUG
							std::cout << "A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ) << ", ";
#endif
						}
#ifdef _DEBUG
						std::cout << "are multiplied pairwise with ";
#endif
						for( size_t l = B_raw.col_start[ i ]; l < B_raw.col_start[ i + 1 ]; ++l ) {
							const size_t l_col = B_raw.row_index[ l ];
							if( coors1.assigned( l_col ) ) {
								coors2.assign( l_col );
								(void)grb::apply( valbuf[ l_col ], valbuf[ l_col ], B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ), oper );
#ifdef _DEBUG
								std::cout << "B( " << i << ", " << l_col << " ) = " << B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ) << " to yield C( " << i << ", "
										  << l_col << " ), ";
#endif
							}
						}
#ifdef _DEBUG
						std::cout << "\n";
#endif
						for( size_t k = 0; k < coors2.nonzeroes(); ++k ) {
							const size_t j = coors2.index( k );
							// update CRS
							// C_raw.row_index[ nnz_current_tile ] = j;
							C_raw.setValue( nnz_current_tile, valbuf[ j ] );							
							// update count
							(void)++nnz_current_tile;
						}
						// C_raw.col_start[ i + 1 ] = nnz_current_tile;						

#ifdef _DEBUG
						std::cout << "\n";
#endif
					}

					/*
					#pragma omp critical
					{
					    const size_t num_tiles = nnz_tiles_C.size();
					    if( 0 == tile_id ) {
					        std::cout << "-------COMPUTATIONAL PHASE-------" << std::endl;
					        std::cout << "number of tiles= " << num_tiles << std::endl;
					    }

					    const size_t nnz_tile_old = current_nnz - previous_nnz;
					    const size_t nnz_tile_new = nnz_current_tile - previous_nnz;
					    std::cout << "[" << lower_bound << ", " << upper_bound << "]";
					    std::cout << ", local_nnz = " << nnz_tiles_C[ tile_id ];
					    std::cout << ", nnz_tile_old = " << nnz_tile_old << " , nnz_tile_new = " << nnz_tile_new;
					    std::cout << std::endl;


					    for( size_t i = 0; i < ptr_rows.size(); i++ ) {
					        std::cout << ptr_rows[ i ] << ",";
					        std::cout << std::endl;
					    }

					    for( size_t i = 0; i < val_vector.size(); i++ ) {
					        std::cout << val_vector[ i ] << ",";
					    }
					    std::cout << std::endl;

					    if(0 == tile_id)
					    {
					        std::cout << "***** EWISE_EXECUTE FINISHED*****" << std::endl;
					    }
					}
					*/

					return SUCCESS;
				};

				ret = ret ? ret :
							internal::le.addStageLevel3( std::move( func ),
								// name of operation
								internal::Opcode::BLAS3_EWISEAPPLY_GENERIC,
								// size of output matrix
								grb::nrows( C ),
								// size of data type in matrix C
								sizeof( OutputType ),
								// dense_descr
								true,
								// dense_mask
								true,
								// matrices for mxm
								&A, &B, &C, nullptr,std::move( func_count_nonzeros ), std::move( func_prefix_sum ) );
			}
			
			// done
			return SUCCESS;
		}

	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class MulMonoid >
	RC eWiseApply( Matrix< OutputType, nonblocking > & C,
		const Matrix< InputType1, nonblocking > & A,
		const Matrix< InputType2, nonblocking > & B,
		const MulMonoid & mulmono,
		const Phase phase = EXECUTE,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_monoid< MulMonoid >::value,
			void >::type * const = nullptr ) {

		// grb::internal::le.execution();
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D1, InputType1 >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D2, InputType2 >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D3, OutputType >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator" );

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (reference, monoid)\n";
#endif

		return internal::eWiseApply_matrix_generic< true, descr >( C, A, B, mulmono.getOperator(), mulmono, phase );
	}

	template< Descriptor descr = grb::descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, class Operator >
	RC eWiseApply( Matrix< OutputType, nonblocking > & C,
		const Matrix< InputType1, nonblocking > & A,
		const Matrix< InputType2, nonblocking > & B,
		const Operator & mulOp,
		const Phase phase = EXECUTE,
		const typename std::enable_if< ! grb::is_object< OutputType >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && grb::is_operator< Operator >::value,
			void >::type * const = nullptr ) {
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::eWiseApply (reference, matrix <- matrix x matrix, operator)",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		static_assert( ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
			"grb::eWiseApply (reference, matrix <- matrix x matrix, operator): "
			"the operator version of eWiseApply cannot be used if either of the "
			"input matrices is a pattern matrix (of type void)" );

		// grb::internal::le.execution();
		typename grb::Monoid< grb::operators::mul< double >, grb::identities::one > dummyMonoid;

		return internal::eWiseApply_matrix_generic< false, descr >( C, A, B, mulOp, dummyMonoid, phase );
	}

	template< 
			Descriptor descr = descriptors::no_operation, 
			typename InputType, 
			typename RIT, typename CIT,
			typename NIT, 
			typename IOType,
			typename MaskType,
			class Monoid		
		>		
		RC foldl( IOType & x,
			Matrix< InputType, nonblocking, RIT, CIT, NIT > & A,
			const Matrix< MaskType, nonblocking, RIT, CIT, NIT > & mask,
			const Monoid & monoid,
			const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value && ! grb::is_object< MaskType >::value && grb::is_monoid< Monoid >::value,
				void >::type * const = nullptr ) {
			// static checks
			static_assert( ! std::is_same< InputType, void >::value,
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"the operator version of foldl cannot be used if the "
				"input matrix is a pattern matrix (of type void)" );
			static_assert( ! std::is_same< IOType, void >::value,
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"the operator version of foldl cannot be used if the "
				"result is of type void" );
			static_assert( ( std::is_same< typename Monoid::D1, IOType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with a prefactor input type that does not match the first domain of the given operator" );
			static_assert( ( std::is_same< typename Monoid::D2, InputType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with a postfactor input type that does not match the first domain of the given operator" );
			static_assert( ( std::is_same< typename Monoid::D3, IOType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with an output type that does not match the output domain of the given operator" );

#ifdef _DEBUG
			std::cout << "In grb::foldl (nonblocking, mask, matrix, monoid)\n";
#endif

			// TODO: implement foldl with mask

			return UNSUPPORTED;
		}

		template< 
			Descriptor descr = descriptors::no_operation, 
			typename InputType, 
			typename RIT, typename CIT,
			typename NIT, 
			typename IOType,			
			class Monoid		
		>		
		RC foldl( IOType & x,
			Matrix< InputType, nonblocking, RIT, CIT, NIT > & A,
			const Monoid & monoid,
			const typename std::enable_if< ! grb::is_object< IOType >::value && ! grb::is_object< InputType >::value && grb::is_monoid< Monoid >::value, void >::type * const = nullptr ) {
			// static checks
			static_assert( ! std::is_same< InputType, void >::value,
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"the operator version of foldl cannot be used if the "
				"input matrix is a pattern matrix (of type void)" );
			static_assert( ! std::is_same< IOType, void >::value,
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"the operator version of foldl cannot be used if the "
				"result is of type void" );
			static_assert( ( std::is_same< typename Monoid::D1, IOType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with a prefactor input type that does not match the first domain of the given operator" );
			static_assert( ( std::is_same< typename Monoid::D2, InputType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with a postfactor input type that does not match the first domain of the given operator" );
			static_assert( ( std::is_same< typename Monoid::D3, IOType >::value ),
				"grb::foldl ( reference, IOType <- op( InputType, IOType ): "
				"called with an output type that does not match the output domain of the given operator" );

#ifdef _DEBUG
			std::cout << "In grb::foldl (nonblocking, matrix, monoid)\n";
#endif

			return internal::foldl_unmasked_generic<descr>( x, A, monoid );
		}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // ``_H_GRB_NONBLOCKING_BLAS3''
