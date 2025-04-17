
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

/*
 * @author A. N. Yzelman
 * @date 4th of April, 2025
 */

#ifndef _H_GRB_REFERENCE_SPTRSVSCHEDULE
#define _H_GRB_REFERENCE_SPTRSVSCHEDULE

#include <limits>
#include <vector>

#include <graphblas/utils/autodeleter.hpp>


namespace grb {

	namespace internal {

		template< typename NIT >
		class SptrsvSchedule {

			private:

				/** One deleter for each chunk of thread-local data. */
				std::vector< utils::AutoDeleter< char > > _deleters;

				/**
				 * Fixed-size buffer for realising the default schedule.
				 *
				 * The first two positions are reserved for the range of the default
				 * schedule, while the last position is reserved for the end-position of
				 * the default schedule.
				 */
				NIT default_schedule[ 3 ];

				/** Computes and initialises a trivial schedule. */
				void initTrivial( const NIT n ) {
					// dynamic sanity check
					assert( data[ 0 ] == nullptr );
					// set trivial schedule
					default_schedule[ 0 ] = 0;
					default_schedule[ 1 ] = n;
					default_schedule[ 2 ] = 1;
					nRanges[ 0 ] = 1;
					// set pointers to trivial schedule
					data[ 0 ] = reinterpret_cast< char * >( &(default_schedule[0]) );
					endPositions[ 0 ] = reinterpret_cast< char * >( &(default_schedule[2]) );
					// verify trivial schedule
#ifndef NDEBUG
					{
						NIT * const interpreted = reinterpret_cast< NIT * >(data[0]);
						assert( interpreted[ 0 ] == 0 );
						assert( interpreted[ 1 ] == n );
					}
					{
						NIT * const interpreted = reinterpret_cast< NIT * >(endPositions[0]);
						assert( *interpreted == 1 );
					}
#endif
					// note that this is a trivial schedule
					is_simple = true;

					// the default schedule does not (cannot) sort input matrices
					is_sorted = false;
				}

				/**
				 * Implementation of move construction and assignment.
				 */
				void moveImpl( SptrsvSchedule &&toMove ) {
					_deleters = std::move( toMove._deleters );
					default_schedule = std::move( toMove.default_schedule );
					is_sorted = toMove.is_sorted;
					is_simple = toMove.is_simple;
					supersteps = toMove.supersteps;
					nThreads = toMove.nThreads;
					data = std::move( toMove.data );
					endPositions = std::move( toMove.endPositions );
					nRanges = std::move( toMove.nRanges );
					toMove.is_sorted = true;
					toMove.is_simple = false;
					toMove.supersteps = 0;
					toMove.nThreads = 0;
				}


			public:

				/** Whether the matrix is sorted as part of tuning. */
				bool is_sorted;

				/** Whether the schedule is simple.*/
				bool is_simple;

				/** Number of schedule steps. */
				size_t supersteps;

				/**
				 * The number of threads the schedule is designed for.
				 */
				size_t nThreads;

				/** One data pointer per thread. */
				std::vector< char * > data;

				/** One end-position array per thread. */
				std::vector< char * > endPositions;

				/** The total number of ranges per thread. */
				std::vector< NIT > nRanges;

				/** Move constructor. */
				SptrsvSchedule( SptrsvSchedule &&toMove ) {
					moveImpl( toMove );
				}

				/**
				 * Base constructor.
				 *
				 * @param[in] n The size of the linear system.
				 *
				 * Initialises the schedule to the trivial always-valid schedule, which is
				 * a single-thread single-superstep schedule that assigns all work to the
				 * first (and only) thread.
				 */
				SptrsvSchedule( const NIT n ) :
					_deleters( 1 ), is_sorted( true ), is_simple( false ),
					supersteps( 1 ), nThreads( 1 ),
					data( 1 ), endPositions( 1 ), nRanges( 1 )
				{
					data[ 0 ] = nullptr;
					endPositions[ 0 ] = nullptr;
					initTrivial( n );
				}

				/**
				 * Specialised constructor for multiple threads.
				 *
				 * @param[in] n The size of the linear system.
				 * @param[in] T The number of threads a schedule should be created for.
				 *
				 * Initialises the schedule to the trivial always-valid schedule, which is
				 * a single-superstep schedule that assigns all work to the first thread.
				 */
				SptrsvSchedule( const NIT n, const size_t T ) :
					_deleters( 2 * T ), is_sorted( true ), is_simple( false ),
					supersteps( 1 ), nThreads( T ),
					data( T, nullptr ), endPositions( T, nullptr ), nRanges( T, 0 )
				{
					if( T == 0 || T > std::numeric_limits< int >::max() ) {
						throw std::runtime_error( "Invalid number of threads" );
					}
					initTrivial( n );
				}

				/**
				 * Base destructor.
				 *
				 * Dynamic memory is freed through the #_deleters.
				 */
				~SptrsvSchedule() {
					// sanity checks only
					assert( nThreads != 0 );
					if( nThreads == 1 ) {
						assert( supersteps == 1 );
					}
				}

				/** Move assignment. */
				SptrsvSchedule& operator=( SptrsvSchedule &&toMove ) {
					moveImpl( toMove );
					return *this;
				}

				/**
				 * Allocates a thread-local chunk of data.
				 *
				 * Must be called from within the thread that will use it(!)
				 */
				void alloc( const size_t s, const size_t nRanges_in ) {
					assert( s < nThreads );
					assert( supersteps > 0 );
					assert( _deleters.size() >= s );
					assert( data.size() >= s );
					assert( data[ s ] == nullptr );
					assert( endPositions[ s ] == nullptr );
					if( nRanges[ s ] != nRanges_in ) {
						throw std::runtime_error( "Given nRanges differs from the one given at "
							"allocation time" );
					}
					grb::RC rc = grb::SUCCESS;
					if( nRanges_in > 0 ) {
						rc = utils::alloc(
							"grb::internal::SptrsvSchedule (default constructor)",
							"default thread-local data allocation, variant I",
							data[ s ], 2 * nRanges_in * sizeof(NIT), false, _deleters[ s ],
							endPositions[ s ], supersteps * sizeof( NIT ), false, _deleters[ s + nThreads ]
						);
					} else {
						data[ s ] = nullptr;
						rc = utils::alloc(
							"grb::internal::SptrsvSchedule (default constructor)",
							"default thread-local data allocation, variant II",
							endPositions[ s ], supersteps * sizeof( NIT ), false, _deleters[ s + nThreads ]
						);
					}
					if( rc != grb::SUCCESS ) {
						throw std::bad_alloc();
					}
				}

		};

	} // end namespace `grb::internal'

} // end namespace `grb'

#endif // _H_GRB_REFERENCE_SPTRSVSCHEDULE

