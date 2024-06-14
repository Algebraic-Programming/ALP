
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
 * The analytic model to be used by the AscendC code, at operator run-time.
 *
 * @author A. N. Yzelman
 * @date 25th of October, 2023
 */

#ifndef _H_ALP_ASCEND_ANALYTIC_MODEL
#define _H_ALP_ASCEND_ANALYTIC_MODEL

#include <cmath>
#include <string>
#include <array>
#include <vector> // TODO FIXME factor this out -- too high runtime overhead
#include <algorithm>

#ifdef _DEBUG
 #include <iostream>
#endif

#include <assert.h>

#ifndef ASC_FORCE_BINARY_SEARCH
 /** Set this macro to true to force a binary search */
 #define ASC_FORCE_BINARY_SEARCH false
#endif


/** The ALP@Ascend namespace for run-time components. */
namespace asc {

	/**
	 * The analytic model is parametrised in the dimensionality of the process
	 * mesh and the problem mesh.
	 *
	 * For the tensors that are in the pipeline, it furthermore requires static
	 * knowledge on whether the dynamic axes (the axes over which the user program
	 * iterates) of the tensors involved with the pipeline, differ.
	 */
	template< size_t process_order, size_t problem_order, bool has_differing_dyn_axes >
	class AnalyticModel {

		private:

			/** Whether to force a binary search */
			static constexpr const bool force_binary = ASC_FORCE_BINARY_SEARCH;

			/** The total scratchpad space, in bytes. */
			const size_t totalSpace;

			std::array< size_t, process_order > processSizes;

			std::array< size_t, problem_order > problemSizes;

			std::array< bool, problem_order > iterationAxes;

			std::vector< std::pair< std::vector< unsigned int >, size_t > > tensors;

			std::array< size_t, problem_order > blockLengths;

			std::vector< unsigned int > largestDynamicAxes;

			size_t largestSize;

			size_t largestStaticSize;

			size_t totalStaticSize;

			/** The size of buffers used by the AscendC program. */
			size_t bufferSize;

			unsigned int numStages;

			unsigned int nDynamicAxes;

			/** Whether the block lengths have been computed. */
			bool lock;

			/** Checks whether current block lengths overrun the buffer */
			bool feasible() const {
				const size_t effectiveBufferSize = totalSpace - bufferSize;
				size_t required = 0;
				for( const auto &pair : tensors ) {
					size_t size = pair.second;
					for( const auto &dyn_axis : pair.first ) {
						const size_t tileSize = std::max( 1ul, blockLengths[ dyn_axis ] );
						size *= tileSize;
					}
					required += size;
				}
#ifdef _DEBUG
				std::cout << "\t\tfeasibility of current solution: " << required << " <= "
					<< effectiveBufferSize << "\n";
#endif
				return required <= effectiveBufferSize;
			}

			void analyticSolve() {
				const size_t n = tensors.size();
				const size_t effectiveBufferSize = totalSpace - bufferSize;
				const size_t maxMul = effectiveBufferSize / totalStaticSize;
				const unsigned int d = largestDynamicAxes.size();
#ifdef _DEBUG
				std::cout << "\tanalyticSolve called with n = " << n << ", "
					<< "effectiveBufferSize = " << effectiveBufferSize << ", "
					<< "largestStaticSize = " << largestStaticSize << ", "
					<< "totalStaticSize = " << totalStaticSize << ", "
					<< "maxMul = " << maxMul << ", and "
					<< "d = " << d << "\n";
#endif
				if( d == 1 ) {
#ifdef _DEBUG
					std::cout << "\t\tsuggested blocksize is " << maxMul << "\n";
#endif
					blockLengths[ largestDynamicAxes[ 0 ] ] = maxMul;
				} else {
					// taking max with 1 is safe since we already know 1, 1, ..., 1 is a sol
					const double root = std::max( std::pow(
							static_cast< double >(maxMul),
							static_cast< double >(1) / static_cast< double >(d) ),
						static_cast< double >(1) );
#ifdef _DEBUG
					std::cout << "\t\tinitial suggested blocksize is " << root << "\n";
#endif
					// select solution
					size_t sizeTaken = totalStaticSize;
					for( const auto &axis : largestDynamicAxes ) {
						blockLengths[ axis ] = root;
						sizeTaken *= root;
					}
					// add one until we fill up the buffer: O(d) work
					unsigned int incDim = 0;
					assert( totalStaticSize > 0 );
					while( sizeTaken + totalStaticSize <= effectiveBufferSize ) {
						(void) ++(blockLengths[ largestDynamicAxes[ incDim ] ]);
#ifdef _DEBUG
						std::cout << "\t\tblock_length" << largestDynamicAxes[ incDim ]
							<< "incremented with one\n";
#endif
						sizeTaken += totalStaticSize;
						(void) ++incDim;
						if( incDim % largestDynamicAxes.size() == 0 ) {
							assert( sizeTaken + totalStaticSize > effectiveBufferSize );
						}
					}
				}
#ifdef _DEBUG
				std::cout << "\t\tWill return the following solution:\n";
				for( unsigned int i = 0; i < problem_order; ++i ) {
					std::cout << "\t\t\tblock_length" << i << " = "
						<< blockLengths[ i ] << "\n";
				}
#endif
			}

			void binarySearch() {
				if( !feasible() ) {
					// only in this case we need to compute a non-trivial block length
					// we follow a greedy approach where we increase the dimension of the
					// blocking only if blocking in one direction was not feasible
					unsigned int dim = 1;
					std::array< size_t, problem_order > loSizes;
					std::array< size_t, problem_order > curSizes;
					std::array< size_t, problem_order > hiSizes;
					bool foundFeasible = false;
					std::array< size_t, problem_order > lastFeasible;
					// NOTE this finds the asymptotic optimum if there's one iteration axis
					// TODO work out the model in multiple dimensions
					while( !foundFeasible ) {
						// set up binary search
						assert( dim <= largestDynamicAxes.size() );
						for( unsigned int i = 0; i < dim; ++i ) {
							const size_t size = problemSizes[ largestDynamicAxes[ i ] ];
							loSizes[ i ] = 1;
#ifdef _DEBUG
							std::cout << "\tproblemSizes[ " << i << " ] = " << problemSizes[ i ]
								<< "\n";
#endif
							curSizes[ i ] = std::max( 1ul, size / 2 );
							hiSizes[ i ] = size;
							blockLengths[ i ] = 1;
						}
						// start binary search
						bool converged = false;
						while( !converged ) {
#ifdef _DEBUG
							for( unsigned int i = 0; i < dim; ++i ) {
								std::cout << "\tcurrent search: " << loSizes[ i ] << ", "
									<< curSizes[ i ] << ", " << hiSizes[ i ] << "\n";
							}
#endif
							// active & evaluate current guess
							bool notFeasible = true;
							{
								unsigned int curDim = 0;
								for( const auto &dyn_axis : largestDynamicAxes ) {
									blockLengths[ dyn_axis ] = curSizes[ curDim ];
									(void) ++curDim;
									if( curDim >= dim ) { break; }
								}
								notFeasible = !feasible();
							}
							// update search direction
							const std::array< size_t, problem_order > lastCur = curSizes;
							if( notFeasible ) {
								// mid point is not feasible, update hi and cur
								for( unsigned int i = 0; i < dim; ++i ) {
									hiSizes[ i ] = curSizes[ i ];
									curSizes[ i ] = std::max( 1ul,
										(hiSizes[ i ] - loSizes[ i ]) / 2 + loSizes[ i ] );
								}
							} else {
								foundFeasible = true;
								lastFeasible = curSizes;
								// mid point is feasible, update lo and cur
								for( unsigned int i = 0; i < dim; ++i ) {
									loSizes[ i ] = curSizes[ i ];
									curSizes[ i ] = std::max( 1ul,
										(hiSizes[ i ] - loSizes[ i ]) / 2 + loSizes[ i ] );
								}
							}
							// check convergence
							converged = true;
							for( unsigned int i = 0; i < dim; ++i ) {
								if( lastCur[ i ] != curSizes[ i ] ) {
									converged = false;
								}
							}
						} // end binary search
						if( !foundFeasible ) {
#ifdef _DEBUG
							std::cout << "\tend of binary search without finding any feasible "
								<< "solution at dim " << dim << "\n";
#endif
							(void) ++dim;
							if( dim >= largestDynamicAxes.size() ) {
								// This situation should never occur, because the trivial solution of
								// blockSize one everywhere should, before calling this function,
								// already have been determined to be feasible.
								throw std::runtime_error( "Search failed but this situation should "
									"never be encountered-- please submit a bug report" );
							}
						}
					}
					// re-activate last found feasible solution
					assert( foundFeasible );
					unsigned int curDim = 0;
					for( const auto &dyn_axis : largestDynamicAxes ) {
						blockLengths[ dyn_axis ] = lastFeasible[ curDim ];
						(void) ++curDim;
						if( curDim >= dim ) { break; }
					}
					assert( feasible() );
				}
			}

			void computeBlockLengths() {
#ifdef _DEBUG
				std::cout << "\tIn computeBlockLengths()\n"
					<< "\t\tlargestDynamicAxes.size() = " << largestDynamicAxes.size() << "\n";
#endif
				for( unsigned int i = 0; i < problem_order; ++i ) {
					blockLengths[ i ] = 1;
				}
				if( !feasible() ) {
					throw std::runtime_error( "Operator cannot be executed for the given "
						"problem sizes." );
				}
				std::vector< unsigned int > activeProcIDs; // TODO FIXME remove dependence on std::vector (for performance)
				unsigned int procGridDim = 0;
				for( unsigned int i = 0; i < process_order; ++i ) {
					assert( processSizes[ i ] > 0 );
					if( processSizes[ i ] > 1 ) {
						activeProcIDs.push_back( i );
						(void) ++procGridDim;
					}
				}
				if( procGridDim > largestDynamicAxes.size() ) {
					// we need to reduce the process mesh
					// we just alternate between expanding the first
					// largestDynamicAxes mesh sizes
					unsigned int curProcInd = 0;
					for( unsigned int i = largestDynamicAxes.size(); i < procGridDim; ++i ) {
						processSizes[ curProcInd ] *= processSizes[ i ];
						processSizes[ i ] = 1;
						(void) ++curProcInd;
						if( curProcInd % procGridDim == 0 ) {
							curProcInd = 0;
						}
					}
				}
				// compute effective dynamic sizes
				for( const auto &dyn_axis : largestDynamicAxes ) {
					const size_t n = problemSizes[ dyn_axis ];
					const size_t p = processSizes[ dyn_axis ];
					if( n % p == 0 ) {
						problemSizes[ dyn_axis ] = n / p;
					} else {
						problemSizes[ dyn_axis ] = n / p + 1;
					}
				}
				// check for trivial solution
				for( const auto &dyn_axis : largestDynamicAxes ) {
#ifdef _DEBUG
					std::cout << "\tSetting blockLengths[ " << dyn_axis << " ] to "
						<< problemSizes[ dyn_axis ] << "\n";
#endif
					blockLengths[ dyn_axis ] = problemSizes[ dyn_axis ];
				}
				if( !feasible() ) {
					// choose between solution strategy
					if( force_binary || (problem_order > 1 && has_differing_dyn_axes) ) {
						binarySearch();
					} else {
						analyticSolve();
					}
				}

				// done
				lock = true;
			}


		public:

			/**
			 * After successful creation, the analytic model is \em unlocked, meaning
			 * information of the pipeline may be ingested.
			 *
			 * TODO: the analytic model currently takes a single scratchpad size,
			 *       \a spsize. But probably it should take two: one for the vector
			 *       unit, and one for the tensor unit.
			 */
			AnalyticModel(
				const size_t spSize,
				std::array< size_t, process_order > procSizes,
				std::array< size_t, problem_order > probSizes,
				std::array< bool, problem_order > iterAxes
			) :
				totalSpace( spSize ),
				processSizes( std::move( procSizes ) ),
				problemSizes( std::move( probSizes ) ),
				iterationAxes( std::move( iterAxes ) ),
				largestSize( 0 ), largestStaticSize( 0 ), totalStaticSize( 0 ),
				bufferSize( 0 ), numStages( 0 ),
				lock( false )
			{
				nDynamicAxes = 0;
				for( unsigned int i = 0; i < problem_order; ++i ) {
					if( iterationAxes[ i ] ) {
						(void) ++nDynamicAxes;
					}
					blockLengths[ i ] = 0;
				}
			}

			/**
			 * Registers a buffer required by the pipeline.
			 *
			 * Buffers are not allowed to have dynamic dimensions.
			 *
			 * \warning This function does not check for violation of this requirement.
			 */
			void addBuffer(
				const size_t elemSize,
				const std::array< bool, problem_order > &tensor
			) noexcept {
				assert( !lock );
				size_t curSize = elemSize;
				for( unsigned int i = 0; i < problem_order; ++i ) {
					if( tensor[ i ] ) {
						curSize *= problemSizes[ i ];
					}
				}
				bufferSize += curSize;
			}

			/**
			 * Registers a general tensor required by the pipeline.
			 *
			 * The given tensor is guaranteed smaller than some other tensor that has
			 * been, or will be, passed to #addGlobalTensor.
			 */
			void addMinorTensor(
				const size_t elemSize,
				const std::array< bool, problem_order > &tensor
			) noexcept {
				assert( !lock );
				size_t staticSize = elemSize;
				std::vector< unsigned int > dynamicAxes;
				for( size_t i = 0; i < problem_order; ++i ) {
					if( tensor[ i ] ) {
						if( iterationAxes[ i ] ) {
							dynamicAxes.push_back( i );
						} else {
							staticSize *= problemSizes[ i ];
						}
					} 
				}
				totalStaticSize += staticSize;
				tensors.push_back( std::make_pair( dynamicAxes, staticSize ) );
#ifdef _DEBUG
				std::cout << "Added minor tensor with " << elemSize << "-byte elements, "
					<< dynamicAxes.size() << " dynamic axes, and a static size of "
					<< staticSize << " bytes.\n";
#endif
			}

			/**
			 * Registers a general tensor required by the pipeline.
			 */
			void addGlobalTensor(
				const size_t elemSize,
				const std::array< bool, problem_order > &tensor
			) {
				assert( !lock );
				size_t staticSize = elemSize;
				std::vector< unsigned int > dynamicAxes;
				for( size_t i = 0; i < problem_order; ++i ) {
					if( tensor[ i ] ) {
						if( iterationAxes[ i ] ) {
							dynamicAxes.push_back( i );
						} else {
							staticSize *= problemSizes[ i ];
						}
					}
				}
				totalStaticSize += staticSize;
				tensors.push_back( std::make_pair( dynamicAxes, staticSize ) );
				size_t globalSize = staticSize;
				for( const unsigned int &axis : dynamicAxes ) {
					globalSize *= problemSizes[ axis ];
				}
#ifdef _DEBUG
				std::cout << "\tadded global tensor with elements of " << elemSize
					<< " bytes, with a globalSize of " << globalSize
					<< " bytes, while the current largest size is " << largestSize
					<< ", and #dynamic axes is " << dynamicAxes.size()
					<< "\n";
#endif
				if( globalSize > largestSize ) {
					largestDynamicAxes = std::move( dynamicAxes );
					largestSize = globalSize;
					largestStaticSize = staticSize;
				}
			}

			/**
			 * This is actually a place-holder for a mechanism that gives the analytic
			 * model more precise information on the stages in the pipeline. Rationale
			 * on why this is needed: some stages (AscendC operators) require work space
			 * buffers.
			 *
			 * @param[in] n The number of stages in the pipeline.
			 */
			void setNumStages( const size_t n ) {
				numStages = n;
			}

			/**
			 * Computes the block sizes suggested by the analytic model.
			 *
			 * Locks the analytic model.
			 */
			size_t getBlockSize( const unsigned int axis ) {
				if( !lock ) {
					computeBlockLengths();
				}
				return blockLengths[ axis ];
			}

	};

}

#endif

