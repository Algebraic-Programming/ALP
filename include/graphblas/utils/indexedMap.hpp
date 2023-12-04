
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
 * @date 5th of April, 2018
 */

#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <graphblas.hpp>

#ifndef _H_GRB_UTILS_INDEXEDMAP
#define _H_GRB_UTILS_INDEXEDMAP

namespace grb {

	namespace utils {

		/**
		 * Represents a set of integers with unique IDs.
		 *
		 * Assumes a source file exists that contains a series of keys of type
		 * \a KeyType. These are read in and assigned an integer of type
		 * \a ValueType. This mapping can aid the build-up of matrices and vectors
		 * during the execution of a GraphBLAS app. The numbering will be contiguous,
		 * meaning that 1) the relative order between keys in the input file is
		 * retained when translated to indices, and 2) the indices are in the range of
		 * 0 (inclusive) to m (exclusive), where m is the total number of keys that
		 * appear in the input file.
		 *
		 * The mapping can be built up in various ways detailed in #Mode.
		 *
		 * @tparam ValueType The type of integers (default: <tt>size_t</tt>).
		 * @tparam KeyType   The type of keys (default: <tt>std::string</tt>).
		 */
		template< typename KeyType = std::string, typename ValueType = size_t >
		class IndexedMap {

			// sanity check on type parameter
			static_assert( std::is_integral< ValueType >::value,
				"Template parameter ``ValueType'' to grb::utils::IndexedMap should "
				"be of integral type."
			);


		private:

			/** Distributed word map. */
			std::map< KeyType, ValueType > key2id;

			/** Stores the inverse mapping. */
			std::vector< KeyType > id2key;

			/** Whether the inverse map is to be stored. */
			bool inverse;

			/** The root node (or \a SIZE_MAX if there is no root). */
			size_t root;

			/** Disable default constructor. */
			IndexedMap() {};


		public:

			/**
			 * Various IO modes this Indexed Map can operate in.
			 *
			 * Only a parallel mode would require communication during the build-up or
			 * use of the map.
			 */
			enum Mode {

				/** The map is replicated at each user process. */
				REPLICATED = 0,

				/** The map is only available at a user-designated root node. */
				SEQUENTIAL
			};

			/**
			 * Default constructor of an indexed map.
			 *
			 * @param[in] filename  Which file to construct this IndexedMap from.
			 * @param[in] mode      Which mode the IndexedMap is operating in.
			 * @param[in] store_inv Whether the inverse map should be stored also.
			 * @param[in] root_pid  In case \a mode is \a SEQUENTIAL, which user
			 *                      process plays the role of the root process.
			 *
			 * A call to this constructor is collective across all user processes
			 * executing the same program.
			 */
			IndexedMap( const std::string filename,
				const Mode mode,
				const bool store_inv = false, const size_t root_pid = 0
			) : inverse( store_inv ) {
				// get SPMD info
				const size_t my_id = grb::spmd<>::pid();
				const size_t P = grb::spmd<>::nprocs();
				if( mode == SEQUENTIAL ) {
					if( root_pid > P ) {
						throw std::invalid_argument( "root PID must be in range of current "
							<< "number of user processes" );
					}
					root = root_pid;
					if( root != my_id ) {
						// done
						return;
					}
				}

				assert( mode == SEQUENTIAL || mode == REPLICATED );
				std::ifstream input;
				ValueType counter = 0;
				input.open( filename );
				if( ! input ) {
					throw std::runtime_error( "Could not open file." );
				}
				KeyType temp;
				while( input >> temp ) {
					const auto it = key2id.find( temp );
					if( it != key2id.end() ) {
						std::cerr << "Warning: double-defined key found: " << temp
							<< ". Ignoring it.\n";
					} else {
						key2id[ temp ] = counter++;
						if( inverse ) {
							id2key.push_back( temp );
						}
						assert( key2id.size() == counter );
						assert( id2key.size() == counter );
					}
				}
			}

			/**
			 * Translates a key to an index.
			 *
			 * @param[in] query The key to find.
			 * @return If the key is found in this map, its corresponding index.
			 *         Otherwise, the maximum representable index is returned instead.
			 */
			ValueType getIndex( const KeyType query ) const noexcept {
				const auto it = key2id.find( query );
				if( it == key2id.end() ) {
					return std::numeric_limits< ValueType >::max();
				} else {
					return it->second;
				}
			}

			/**
			 * Translates an index to a key. This is the inverse function of #getIndex.
			 *
			 * @param[in] query Which key to return.
			 * @return The requested key, if this instance stores the inverse mapping and
			 *         the \a query is within the valid range.
			 *
			 * @see size()
			 */
			KeyType getKey( const ValueType query ) const {
				assert( inverse );
				if( query < id2key.size() ) {
					return id2key[ query ];
				} else {
					throw std::runtime_error( "Requested out-of-range key." );
				}
			}

			/**
			 * The number of keys stored in this map.
			 *
			 * @return The total number of keys stored in this map.
			 */
			size_t size() const noexcept {
				assert( key2id.size() == id2key.size() );
				return key2id.size();
			}

		}; // end class IndexedMap

	} // end namespace utils

} // end namespace grb

#endif // end flag ``_H_GRB_UTILS_INDEXEDMAP''

