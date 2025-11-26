
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
 * @date 7th of February, 2022
 */

#ifndef _H_GRB_DMAPPER
#define _H_GRB_DMAPPER

#include <map>
#include <set>
#include <limits>

#include <assert.h>


namespace grb {

	namespace utils {

		/**
		 * Maps a non-deterministic sequence of indices of type \a IndexType to a
		 * deterministic sequence. Supports sequences of insertions and deletions.
		 *
		 * @tparam IndexType The type of the index sequences to take in and to return.
		 *
		 * If insertions and deletions happen deterministically, this mapper guarantees
		 * a deterministic sequence of indices is returned. This applies within a single
		 * program execution and across different program executions.
		 */
		template< typename IndexType >
		class DMapper {

			private:

				/**
				 * For newly encountered indices, used to assign a deterministic one.
				 */
				IndexType counter;

				/**
				 * A map of registered non-deterministic indices and their deterministic
				 * counterparts.
				 */
				std::map< IndexType, IndexType > mapper;

				/**
				 * Inverse map of #mapper
				 */
				std::map< IndexType, IndexType > invmap;

				/**
				 * A set of previously assigned deterministic indices that have been since
				 * removed (and are now free for reuse).
				 */
				std::set< IndexType > removals;


			public:

				/**
				 * Default constructor.
				 */
				DMapper() : counter( 0 ) {}

				/**
				 * Appends an insertion in the current index sequence.
				 *
				 * @param[in] in The index to be translated to a deterministic one.
				 *
				 * If \a in appeared as an earlier insertion, there must have been a matching
				 * call to delete using the same index \a in, or otherwise undefined behaviour
				 * will occur.
				 */
				IndexType insert( const IndexType in ) {
#ifdef _DEBUG
					std::cout << "DMapper::insert( " << in << " )" << std::endl;
#endif
#ifndef NDEBUG
					const auto &it = mapper.find( in );
					assert( it == mapper.end() );
#endif
					IndexType ret;
					if( removals.size() > 0 ) {
						const auto &removeIt = removals.begin();
						ret = *removeIt;
						removals.erase( removeIt );
					} else {
						assert( counter != std::numeric_limits< IndexType >::max() );
						ret = counter++;
					}
#ifdef _DEBUG
					std::cout << "\t inserting (" << in << ", " << ret << ") into mapper"
						<< std::endl;
#endif
					mapper.insert( {in, ret} );

#ifdef _DEBUG
					std::cout << "\t inserting (" << ret << ", " << in << ") into invmap"
						<< std::endl;
#endif
					invmap.insert( {ret, in} );
#ifdef _DEBUG
					std::cout << "\t returns " << ret << "\n";
#endif
					return ret;
				}

				/**
				 * Appends a deletion into the current index sequence.
				 *
				 * @param[in] in The index to be removed. The given index must have been
				 *               returned by a call to insert prior this function being
				 *               called.
				 *
				 * There must not have been a call to this function with the same \a in
				 * parameter after the call to #insert that returned \a in.
				 */
				void remove( const IndexType in ) {
#ifdef _DEBUG
					std::cout << "DMapper::remove( " << in << " )" << std::endl;
#endif
					const auto &it = invmap.find( in );
					assert( it != invmap.end() );
					const IndexType global_id = it->second;
#ifdef _DEBUG
					std::cout << "\t request corresponds to non-deterministic ID "
						<< global_id << "\n";
#endif
					{
						const auto &toremove = mapper.find( global_id );
						assert( toremove != mapper.end() );
						mapper.erase( toremove );
					}
					invmap.erase( it );
					removals.insert( in );
				}

				/**
				 * Clears this DMapper from all entries.
				 *
				 * After a call to this function, it shall be as though this instance was
				 * newly constructed.
				 */
				void clear() {
					counter = 0;
					mapper.clear();
					invmap.clear();
					removals.clear();
				}

		}; // end ``grb::utils::DMapper''

	} // end namespace ``grb::utils''

} // end namespace ``grb''

#endif // _H_GRB_DMAPPER

