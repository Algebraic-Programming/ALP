
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

				/** Computes and initialises a trivial schedule. */
				void initTrivial( const NIT n ) {
					assert( data[ 0 ] == nullptr );
					const grb::RC rc = utils::alloc(
						"grb::internal::SptrsvSchedule (default constructor)",
						"default thread-local data allocation",
						data[ 0 ], 2 * sizeof(NIT), false, _deleters[ 0 ]
					);
					if( rc != grb::SUCCESS ) {
						throw std::bad_alloc();
					}
					NIT * const interpreted = reinterpret_cast< NIT * >(data[0]);
					interpreted[ 0 ] = 0;
					interpreted[ 1 ] = n;
				}


			public:

				/** Number of schedule steps. */
				size_t supersteps;

				/**
				 * The number of threads the schedule is designed for.
				 *
				 * \note Uses int as per OpenMP spec.
				 */
				int nThreads;

				/** One data pointer per thread. */
				std::vector< char * > data;

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
					_deleters( 1 ), supersteps( 1 ), nThreads( 1 ), data( 1 )
				{
					data[ 0 ] = nullptr;
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
					_deleters( T ), supersteps( 0 ), nThreads( T ), data( T )
				{
					data[ 0 ] = nullptr;
					if( T == 0 || T > std::numeric_limits< int >::max() ) {
						throw std::runtime_error( "Invalid number of threads" );
					}
					initTrivial( n );
				}

		};

	} // end namespace `grb::internal'

} // end namespace `grb'

#endif // _H_GRB_REFERENCE_SPTRSVSCHEDULE

