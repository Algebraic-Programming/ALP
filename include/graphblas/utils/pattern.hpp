
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
 * Collection of helper functions to deal with reading in pattern matrices.
 *
 * @author A. N. Yzelman
 * @date 22nd of March, 2017
 */

#ifndef _H_GRB_UTILS_PATTERN
#define _H_GRB_UTILS_PATTERN

#include <iterator> //for iterator_traits
#include <vector>   //for std::vector in #VectorWrapper

namespace grb {

	namespace utils {

		/** Helper function to increment an iterator to a values array. */
		template< typename Iterator >
		static inline void incValueIterator( Iterator & it ) {
			++it; // standard behaviour
		}

		/**
		 * Overload the above standard helper function in case we are reading in a
		 * pattern matrix. In this case, the iterator will never be derefenced, but
		 * may also never be incremented.
		 */
		static inline void incValueIterator( void *& it ) {
			(void)it; // cannot iterate on pattern matrix input
		}

		/**
		 * Wrapper class for caching nonzero values. It is an alias to some of the
		 * std::vector functionalities used while parsing an input matrix in
		 * coordinate format.
		 */
		template< typename T >
		class VectorWrapper {

		private:
			/** The vector to wrap around. */
			std::vector< T > vector;

		public:
			/**
			 * We normally dereference an iterator and push back its value.
			 * @tparam Iterator The iterator type.
			 * @param[in] it The iterator whose dereferenced value is to be pushed back
			 *               into \a vector.
			 */
			template< typename Iterator >
			void push_back( const Iterator it ) {
				vector.push_back( *it );
			}

			/**
			 * @returns The start iterator to the \a vector.
			 */
			typename std::vector< T >::const_iterator begin() const {
				return vector.begin();
			}

			/**
			 * @returns The end iterator to the \a vector.
			 */
			typename std::vector< T >::const_iterator end() const {
				return vector.end();
			}
		};

		/**
		 * Specialisation of the above #VectorWrapper class for use with patterns
		 * matrices, which do not read in any values as there are none. It translates
		 * all functions of #VectorWrapper into no-ops.
		 */
		template<>
		class VectorWrapper< void > {
		public:
			/**
			 * This function does nothing. It expects an iterator of type a pointer to
			 * \a void. It asserts \a it equal to \a NULL as any other use is a highly
			 * probable indicator of a coding failure.
			 *
			 * @param[in] it The \a NULL pointer that corresponds to a `fake' iterator.
			 */
			void push_back( void * const it ) const {
#ifdef NDEBUG
				(void)it;
#else
				assert( it == NULL );
#endif
			}

			/**
			 * @returns An \a NULL pointer to signal there are no nonzero values
			 *          cached.
			 */
			void * begin() const {
				return NULL;
			}

			/**
			 * @returns An \a NULL pointer to signal there are no nonzero values
			 *          cached.
			 */
			void * end() const {
				return NULL;
			}
		};

		/**
		 * Retrieves the \a value_type trait from a given iterator type.
		 *
		 * @tparam _T The type of the given iterator.
		 *
		 * \note This implementation wraps around std::iterator_traits.
		 */
		template< typename _T >
		struct iterator_value_trait : private std::iterator_traits< _T > {
			/** The type of the value an iterator of type \a _T returns */
			typedef typename std::iterator_traits< _T >::value_type type;
		};

		/**
		 * Template specialisation for iterators over elements of type \a void-- i.e.,
		 * for iterators that correspond to an empty set. This specialisation
		 * correctly infers the data type of \a NULL pointers to a \a void.
		 * This is very useful for writing generic input functions that need to be
		 * able to handle pattern input (i.e., store no values; only nonzero
		 * patterns).
		 */
		template<>
		struct iterator_value_trait< void * > {
			/** An `iterator' of type <code>void *</code> can only return nothing; i.e., \a void. */
			typedef void type;
		};

	} // namespace utils

} // namespace grb

#endif // end ``_H_GRB_UTILS_PATTERN''
