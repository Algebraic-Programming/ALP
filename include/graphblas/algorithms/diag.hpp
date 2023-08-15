
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
 * Implements a set of diag algorithms.
 *
 * @author A. N. Yzelman
 * @date 15th of August, 2023
 */

#ifndef _H_GRB_ALGORITHMS_DIAG
#define _H_GRB_ALGORITHMS_DIAG

namespace grb {

	namespace algorithms {

		/**
		 * Selects the diagonal of an input matrix and copies it to a given output
		 * matrix.
		 *
		 * @tparam descr The descriptor under which to perform the copy with a diagonal
		 *               mask.
		 *
		 * \note If this function is to \em filter (instead of select) the diagonal of
		 *       \a in, then pass #grb::descriptors::invert_mask to this function.
		 *
		 * @tparam NonzeroT The nonzero value of T.
		 *
		 * @param[out] out The output matrix, the diagonal of \a in.
		 * @param[in]  in  The input matrix to take the diagonal of.
		 *
		 * Both matrices should have the same dimensions.
		 *
		 * This algorithm requires a workspace that is a void matrix of the same size
		 * as \a out and \a in, with capacity \f$ \min \{ m, n \} \f$, where
		 * \f$ m \times n \f$ is the size of \a in.
		 *
		 * @param[in,out] workSpace The initial values are ignored and the contents
		 *                          on output are undefined.
		 *
		 * The capacity of \a out is assumed to be sufficient to store the requested
		 * output -- if the required capacity is unknown, first call this algorithm
		 * with a #grb::RESIZE phase.
		 *
		 * @param[in] phase The phase of execution. Optional; default is
		 *                  #grb::EXECUTE.
		 *
		 * @returns #grb::MISMATCH If the \a in or \a out do not match with one another.
		 * @returns #grb::ILLEGAL  If the capacity of \a out is less than the number of
		 *                         nonzeroes in \a in.
		 * @returns #grb::SUCCESS  When the requested computation has completed.
		 */
		template<
			grb::Descriptor descr = grb::descriptors::no_operation,
			typename NonzeroT
		>
		RC diag(
			Matrix< NonzeroT > &out, const Matrix< NonzeroT > &in,
			Matrix< void > workSpace, const Phase phase = grb::EXECUTE
		) {
			const size_t mA = grb::nrows( out );
			const size_t nA = grb::ncols( out );
			const size_t n = std::min( grb::nrows( out ), grb::ncols( out ) );

			// basic run-time checks
			{
				if( nA != grb::ncols( in ) || mA != grb::nrows( in ) ) {
					return MISMATCH;
				}
				if( nA != grb::ncols( workSpace ) || mA != grb::nrows( workSpace ) ) {
					return MISMATCH;
				}
			}
			if( grb::capacity( workSpace ) < n ) {
				return ILLEGAL;
			}

			// throw away any contents of workspace
			grb::RC rc = grb::clear( workSpace );
			if( rc != grb::SUCCESS ) { return rc; };

			// build diagonal mask
			// TODO FIXME
			// the below should be replaced with the machanisms of GitHub #228
			std::vector< size_t > diagI;
			for( size_t i = 0; i < n; ++i ) {
				diagI.push_back( i );
			}
			rc = grb::buildMatrixUnique(
					out,
					diagI.cbegin(), diagI.cend(), 
					diagI.cbegin(), diagI.cend(), 
					grb::SEQUENTIAL
				);

			if( phase == grb::EXECUTE || phase == grb::TRY ) {
				rc = rc ? rc : grb::clear( out );
			}

			rc = rc ? rc : grb::set< descr >( out, workSpace, in, phase );

			// done
			return rc;
		}

		/**
		 * \todo TODO documentation
		 * \todo TODO void NonzeroT variant must be implemented via overload
		 * \todo TODO this function makes use of the vector-to-matrix converter.
		 *            need to double-check that this allows for parallel ingestion.
		 * \note No descriptors should probably be allowed here
		 */
		template<
			grb::Descriptor descr = grb::descriptors::no_operation,
			typename NonzeroT
		>
		RC diag(
			Matrix< NonzeroT > &out, const Vector< NonzeroT > &in,
			const Phase phase = EXECUTE
		) {
			const size_t n = grb::nrows( out );
			if( n != grb::ncols( out ) || grb::size( in ) != n ) {
				return grb::MISMATCH;
			}
			if( grb::capacity( out ) < n ) {
				return grb::ILLEGAL;
			}
			auto converter = grb::utils::makeVectorToMatrixConverter< NonzeroT, NonzeroT >(
					in, [](const size_t &ind, const NonzeroT &val) {
						std::pair< std::pair< size_t, size_t >, NonzeroT > triple;
						triple.first.first = triple.first.second = ind;
						triple.second = val;
						return triple;
					}
				);
			grb::RC ret = grb::clear( out );
			ret = ret ? ret : grb::buildMatrixUnique(
					out,
					converter.begin(), converter.end(), grb::PARALLEL
				);
			return ret;
		}

		/**
		 * \todo TODO documentation
		 */
		template<
			grb::Descriptor descr = grb::descriptors::no_operation,
			typename NonzeroT
		>
		RC diag(
			Vector< NonzeroT > &out, const Matrix< NonzeroT > &in,
			Matrix< void > &workSpace, Vector< bool > &workSpaceV,
			const Phase phase = EXECUTE
		) {
			const size_t n = grb::size( out );

			// basic run-time checks
			{
				if( n != grb::size( workSpaceV ) ) {
					return grb::MISMATCH;
				}
				if( n != std::min( grb::ncols( in ), grb::nrows( in ) ) ) {
					return grb::MISMATCH;
				}
 				if( grb::nrows( in ) != grb::nrows( workSpace ) ) {
					return grb::MISMATCH;
				}
				if( grb::ncols( in ) != grb::ncols( workSpace ) ) {
					return grb::MISMATCH;
				}
			}
			if( grb::capacity( workSpace ) < n ) {
				return ILLEGAL;
			}
			if( grb::capacity( workSpaceV ) < n ) {
				return ILLEGAL;
			}

			// throw away any contents of workspace
			grb::RC rc = grb::clear( workSpace );
			if( rc != grb::SUCCESS ) { return rc; };

			// build diagonal matrix and vector of ones
			// TODO FIXME
			// the below should be replaced with the machanisms of GitHub #228
			std::vector< size_t > diagI;
			std::vector< bool > diagV;
			for( size_t i = 0; i < n; ++i ) {
				diagI.push_back( i );
				diagV.push_back( false );
			}
			grb::operators::right_assign< NonzeroT > rightAssignOp;
			rc = grb::buildMatrixUnique(
					out,
					diagI.cbegin(), diagI.cend(), 
					diagI.cbegin(), diagI.cend(), 
					diagV.cbegin(), diagV.cend(),
					grb::SEQUENTIAL
				);
			rc = rc ? rc : grb::foldl( workSpace, in, rightAssignOp );
			rc = rc ? rc : grb::set( workSpaceV, true );

			// clear output if needed
			if( phase == grb::EXECUTE || phase == grb::TRY ) {
				rc = rc ? rc : grb::clear( out );
			}

			rc = rc ? rc : grb::mxv< descr >( out, workSpace, workSpaceV, phase );

			// done
			return rc;
		}

	} // end namespace grb::algorithms

} // end namespace grb

#endif // end macro _H_GRB_ALGORITHMS_DIAG

