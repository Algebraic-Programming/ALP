
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

#include <type_traits> //for std::enable_if

#include <omp.h>

#include <graphblas/base/blas3.hpp>
#include <graphblas/utils/iterators/MatrixVectorIterator.hpp>

#include "io.hpp"
#include "matrix.hpp"

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
			/*
	// nonblocking execution is not supported
	// first, execute any computation that is not completed
	le.execution();

	// second, delegate to the reference backend
	return mxm_generic< allow_void, descr, MulMonoid, OutputType, InputType1, InputType2, RIT, CIT, NIT, Operator, Monoid >(
		getRefMatrix( C ), getRefMatrix( A ), getRefMatrix( B ), oper, monoid, mulMonoid, phase );
		*/

			/*

			for BLAS3
			This implementation is based upon the implementation of mxm_generic in the reference backend
			IDEA: use same implementation of the one presented in the reference backend. We know that such an implementation
			uses the row-wise matrix-matrix product. For this reason, to compute one row of C = AB, for C of size mxn, A of size mxk
			and B of size kxn, one row of A is needed and the whole matrix B must be available. We make use of this as follows:
			we can compute the output matrix C by split it into several row-tiles, that is each tile of C of size (T, k) can be computed
			by using a tile of the same size and the whole matrix B. As a consequence, matrix can be safely computed in parallel
			Assumption: we use the CSR format only
			*/

			static_assert( allow_void || ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
				"grb::mxm_generic: the operator-monoid version of mxm cannot be "
				"used if either of the input matrices is a pattern matrix (of type "
				"void)" );

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
			config::NonzeroIndexType * C_col_index = internal::template getReferenceBuffer< typename config::NonzeroIndexType >( n + 1 );

			// initialisations
			internal::Coordinates< reference > coors;
			coors.set( arr, false, buf, n );

			// symbolic phase (counting sort, step 1)
			size_t nzc = 0; // output nonzero count
			if( crs_only && phase == RESIZE ) {
				// we are using an auxialiary CRS that we cannot resize ourselves
				// instead, we update the offset array only
				C_raw.col_start[ 0 ] = 0;
			}

			constexpr const bool dense_descr = descr & descriptors::dense;
			//  lambda function to be added into the pipeline
			internal::Pipeline::stage_type func = [ &A, &B, &C, &oper, &monoid, &mulMonoid, &phase, &m, &n, &m_A, &k, &k_B, &n_B, &A_raw, &B_raw, &C_raw, &CCS_raw, &arr, &buf, &valbuf, &C_col_index,
													  &coors, &nzc ]( internal::Pipeline & pipeline, const size_t lower_bound, const size_t upper_bound ) {
				// COMMENT: this is not actually computed in mxm since we pass phase = Execute.
				// What is this function for?
				// if crs_only, then the below implements its resize phase
				// if not crs_only, then the below is both crucial for the resize phase,
				// as well as for enabling the insertions of output values in the output CCS
				if( ( crs_only && phase == RESIZE ) || ! crs_only ) {
					for( size_t i = lower_bound; i < upper_bound; ++i ) {
						// coors is passed to every thread, this would generate data races. USE A LOCK HERE
						coors.clear();
						for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
							const size_t k_col = A_raw.row_index[ k ];
							for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
								const size_t l_col = B_raw.row_index[ l ];
								if( ! coors.assign( l_col ) ) {
									(void)++nzc;
									if( ! crs_only ) {
										(void)++CCS_raw.col_start[ l_col + 1 ];
									}
								}
							}
						}
						if( crs_only && phase == RESIZE ) {
							// we are using an auxialiary CRS that we cannot resize ourselves
							// instead, we update the offset array only
							C_raw.col_start[ i + 1 ] = nzc;
						}
					}
				}

				if( phase == RESIZE ) {
					if( ! crs_only ) {
						// do final resize
						const RC ret = grb::resize( C, nzc );
						return ret;
					} else {
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

				// prefix sum for C_col_index,
				// set CCS_raw.col_start to all zero
#ifndef NDEBUG
				if( ! crs_only ) {
					assert( CCS_raw.col_start[ 0 ] == 0 );
				}
#endif
				C_col_index[ 0 ] = 0;
				for( size_t j = 1; j < n; ++j ) {
					if( ! crs_only ) {
						CCS_raw.col_start[ j + 1 ] += CCS_raw.col_start[ j ];
					}
					C_col_index[ j ] = 0;
				}
#ifndef NDEBUG
				if( ! crs_only ) {
					assert( CCS_raw.col_start[ n ] == nzc );
				}
#endif

#ifndef NDEBUG
				const size_t old_nzc = nzc;
#endif
				// use previously computed CCS offset array to update CCS during the
				// computational phase
				nzc = 0;
				C_raw.col_start[ 0 ] = 0;
				for( size_t i = lower_bound; i < upper_bound; ++i ) {
					coors.clear();
					for( auto k = A_raw.col_start[ i ]; k < A_raw.col_start[ i + 1 ]; ++k ) {
						const size_t k_col = A_raw.row_index[ k ];
						for( auto l = B_raw.col_start[ k_col ]; l < B_raw.col_start[ k_col + 1 ]; ++l ) {
							const size_t l_col = B_raw.row_index[ l ];
#ifdef _DEBUG
							std::cout << "\t A( " << i << ", " << k_col << " ) = " << A_raw.getValue( k, mulMonoid.template getIdentity< typename Operator::D1 >() ) << " will be multiplied with B( "
									  << k_col << ", " << l_col << " ) = " << B_raw.getValue( l, mulMonoid.template getIdentity< typename Operator::D2 >() ) << " to accumulate into C( " << i << ", "
									  << l_col << " )\n";
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
					for( size_t k = 0; k < coors.nonzeroes(); ++k ) {
						assert( nzc < old_nzc );
						const size_t j = coors.index( k );
						// update CRS
						C_raw.row_index[ nzc ] = j;
						C_raw.setValue( nzc, valbuf[ j ] );
						// update CCS
						if( ! crs_only ) {
							const size_t CCS_index = C_col_index[ j ]++ + CCS_raw.col_start[ j ];
							CCS_raw.row_index[ CCS_index ] = i;
							CCS_raw.setValue( CCS_index, valbuf[ j ] );
						}
						// update count
						(void)++nzc;
					}
					C_raw.col_start[ i + 1 ] = nzc;
				}

#ifndef NDEBUG
				if( ! crs_only ) {
					for( size_t j = 0; j < n; ++j ) {
						assert( CCS_raw.col_start[ j + 1 ] - CCS_raw.col_start[ j ] == C_col_index[ j ] );
					}
				}
				assert( nzc == old_nzc );
#endif

				// set final number of nonzeroes in output matrix
				internal::setCurrentNonzeroes( C, nzc );

				// done
				return SUCCESS;
			}; // end lambda function

			// Add function into a pipeline
			RC ret;
			ret = ret ? ret :
						internal::le.addStage( std::move( func ),
							// name of operation
							internal::Opcode::BLAS3_MXM_GENERIC,
							// size of output matrix
							getNonzeroCapacity( &C ),
							// size of data type in matrix C
							sizeof( OutputType ),
							// dense_descr
							dense_descr,
							// dense_mask
							true,
							// output vector
							nullptr, nullptr,
							// coordinates output vectors
							nullptr, nullptr,
							// input pointers
							nullptr, nullptr, nullptr, nullptr,
							// coordinates of input vectors
							nullptr, nullptr,
							// input_matrix
							nullptr, nullptr, nullptr,
							// matrices for mxm
							&A, &B, &C );
			return ret;
		}

	} // namespace internal

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

		if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
			std::cerr << "Warning: mxm (nonblocking, unmasked, semiring) currently "
					  << "delegates to a blocking implementation\n"
					  << "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

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

		if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
			std::cerr << "Warning: mxm (nonblocking, unmasked, monoid-op) currently "
					  << "delegates to a blocking implementation\n"
					  << "         Further similar such warnings will be suppressed.\n";
			internal::NONBLOCKING::warn_if_not_native = false;
		}

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

			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return matrix_zip_generic< descr, matrix_is_void, OutputType, InputType1, InputType2, InputType3, Coords >(
				getRefMatrix( A ), getRefVector( x ), getRefVector( y ), getRefVector( z ), phase );
		}

	} // namespace internal

	template< Descriptor descr = descriptors::no_operation, typename OutputType, typename InputType1, typename InputType2, typename InputType3, typename Coords >
	RC zip( Matrix< OutputType, nonblocking > & A,
		const Vector< InputType1, nonblocking, Coords > & x,
		const Vector< InputType2, nonblocking, Coords > & y,
		const Vector< InputType3, nonblocking, Coords > & z,
		const Phase & phase = EXECUTE ) {
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
	}

	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename Coords >
	RC zip( Matrix< void, nonblocking > & A, const Vector< InputType1, nonblocking, Coords > & x, const Vector< InputType2, nonblocking, Coords > & y, const Phase & phase = EXECUTE ) {
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
	}

	template< Descriptor descr = descriptors::no_operation, typename InputType1, typename InputType2, typename OutputType, typename Coords, class Operator >
	RC outer( Matrix< OutputType, nonblocking > & A,
		const Vector< InputType1, nonblocking, Coords > & u,
		const Vector< InputType2, nonblocking, Coords > & v,
		const Operator & mul = Operator(),
		const Phase & phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< Operator >::value && ! grb::is_object< InputType1 >::value && ! grb::is_object< InputType2 >::value && ! grb::is_object< OutputType >::value,
			void >::type * const = nullptr ) {
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
			if( internal::NONBLOCKING::warn_if_not_native && config::PIPELINE::warn_if_not_native ) {
				std::cerr << "Warning: eWiseApply (nonblocking) currently delegates to a "
						  << "blocking implementation.\n"
						  << "         Further similar such warnings will be suppressed.\n";
				internal::NONBLOCKING::warn_if_not_native = false;
			}

			// nonblocking execution is not supported
			// first, execute any computation that is not completed
			le.execution();

			// second, delegate to the reference backend
			return eWiseApply_matrix_generic< allow_void, descr, MulMonoid, OutputType, InputType1, InputType2, Operator >(
				getRefMatrix( C ), getRefMatrix( A ), getRefMatrix( B ), oper, mulMonoid, phase );
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
		// static checks
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D1, InputType1 >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D2, InputType2 >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with a postfactor input matrix B that does not match the "
			"second domain of the monoid operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename MulMonoid::D3, OutputType >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, monoid)",
			"called with an output matrix C that does not match the output domain "
			"of the monoid operator" );

#ifdef _DEBUG
		std::cout << "In grb::eWiseApply_matrix_generic (nonblocking, monoid)\n";
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
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D1, InputType1 >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with a prefactor input matrix A that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D2, InputType2 >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with a postfactor input matrix B that does not match the first "
			"domain of the given multiplication operator" );
		NO_CAST_ASSERT( ( ! ( descr & descriptors::no_casting ) || std::is_same< typename Operator::D3, OutputType >::value ), "grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator)",
			"called with an output matrix C that does not match the output domain "
			"of the given multiplication operator" );
		static_assert( ( ! ( std::is_same< InputType1, void >::value || std::is_same< InputType2, void >::value ) ),
			"grb::eWiseApply (nonblocking, matrix <- matrix x matrix, operator): "
			"the operator version of eWiseApply cannot be used if either of the "
			"input matrices is a pattern matrix (of type void)" );

		typename grb::Monoid< grb::operators::mul< double >, grb::identities::one > dummyMonoid;
		return internal::eWiseApply_matrix_generic< false, descr >( C, A, B, mulOp, dummyMonoid, phase );
	}

} // namespace grb

#undef NO_CAST_ASSERT

#endif // ``_H_GRB_NONBLOCKING_BLAS3''
