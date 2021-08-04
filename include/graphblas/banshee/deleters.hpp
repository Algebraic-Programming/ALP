
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
 * @author Dan Iorga
 */

#ifndef _H_GRB_BANSHEE_DELETERS
#define _H_GRB_BANSHEE_DELETERS

namespace grb {

	namespace utils {

		namespace internal {

			template<>
			class DeleterFunctions< banshee > {
			private:
				/** Prevent instantiation. */
				DeleterFunctions() {}

			public:
				/** Banshee deleter does nothing. */
				template< typename T >
				static void safe_free( T * const pointer ) {
					(void)pointer;
				}
			};

		} // namespace internal

	} // namespace utils

} // namespace grb

// now define user API:
namespace grb {

	namespace utils {

		/**
		 * Banshee does not support dynamic memory allocation
		 */
		template< typename T >
		class AutoDeleter< T, banshee > {

		private:
			/** Where the implementation of the free function(s) reside. */
			typedef typename internal::DeleterFunctions< banshee > functions;

		public:
			/**
			 * Constructs a new AutoDeleter from a pointer. When this instance and all
			 * instances copied from this one are destroyed, the pointer will be freed
			 * if and only if it is not equal to \a NULL.
			 *
			 * The pointer is assumed to be allocated using \a posix_memalign, which is
			 * compitable with the C-standard \a free. Thus pointers that cannot be
			 * free'd in this manner should never be passed to this AutoDeleter
			 * constructor.
			 *
			 * @throws std::bad_alloc If the system cannot allocate enough memory.
			 */
			AutoDeleter( T * const pointer = NULL, const size_t size = 0 ) {
				(void)pointer;
				(void)size;
			};

			/**
			 * Copies an \a other AutoDeleter. The underlying pointer will only be freed
			 * if at least this new AutoDeleter and the \a other AutoDeleter are
			 * destroyed. (The preceding says `at least' because other copies may have
			 * been made previously.)
			 *
			 * @throws std::bad_alloc If the system cannot allocate enough memory.
			 */
			AutoDeleter( const AutoDeleter< T > & other ) {
				(void)other;
			}

			/**
			 * Creates an AutoDeleter from a temporary instance by stealing its
			 * resources.
			 */
			AutoDeleter( AutoDeleter< T > && other ) noexcept {
				(void)other;
			}

			/** Signals auto-deletion no longer is necessary. */
			void clear() noexcept {}

			/**
			 * Relies on std::move. Equals-operator only works on temporary RHS.
			 */
			AutoDeleter< T > & operator=( AutoDeleter< T > && other ) {
				(void)other;
				return *this;
			}

			/**
			 * Relies on copying the underlying shared pointer.
			 */
			AutoDeleter< T > & operator=( const AutoDeleter< T > & other ) {
				(void)other;
				return *this;
			}
		};

	} // namespace utils

} // namespace grb

#endif // end _H_GRB_BANSHEE_DELETERS
