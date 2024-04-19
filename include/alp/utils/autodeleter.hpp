
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
 * Collection of helper functions to deal with deleting shared pointers.
 *
 * @author A. N. Yzelman
 * @date 22nd of April, 2017
 */

#ifndef _H_ALP_UTILS_AUTO_DELETER
#define _H_ALP_UTILS_AUTO_DELETER

#ifndef _ALP_NO_LIBNUMA
#include <numa.h>
#endif

#include <memory>

#include "alp/config.hpp"

namespace alp {

	namespace utils {

		namespace internal {

			template< enum Backend implementation >
			class DeleterFunctions {
			private:
				/** Prevent instantiation. */
				DeleterFunctions() {}

			public:
				/** \todo documentation */
				template< typename T >
				static void safe_free( T * const pointer ) {
					if( pointer != NULL ) {
						free( pointer );
					}
				}

#ifndef _ALP_NO_LIBNUMA
				/** \todo documentation */
				template< typename T >
				class safe_numa_free {
				private:
					size_t size;

				public:
					safe_numa_free( const size_t size_in ) : size( size_in ) {}
					void operator()( T * const pointer ) {
						numa_free( pointer, size );
					}
				};
#endif
			};

		} // namespace internal

	} // namespace utils

} // namespace alp

// now define user API:
namespace alp {

	namespace utils {

		/**
		 * This function is compatible with \a posix_memalign and the standard
		 * practice to allow \a NULL pointers for empty arrays.
		 *
		 * \todo expand documentation
		 *
		 * \warning This function is \em not thread safe!
		 *
		 * \note The difference between this class and the std::shared_ptr< T > is
		 *       the deleter wrapper. This class also provides no way to reference
		 *       the underlying pointer, nor is functionality provided to access
		 *       the underlying type instance. This class thus only provides a
		 *       means to easily delete memory areas that are shared between owners
		 *       (which is in fact quite different from the philosophy of a
		 *        \a shared_ptr which were introduced to `forget' about deletes.)
		 */
		template< typename T, enum Backend implementation = config::default_backend >
		class AutoDeleter {

		private:
			/** Where the implementation of the free function(s) reside. */
			typedef typename internal::DeleterFunctions< implementation > functions;

			/** Functionality is provided by shared pointer. */
			std::shared_ptr< T > _shPtr;

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
#ifdef _ALP_NO_LIBNUMA
				(void)size;
				const auto free_p = &( functions::template safe_free< T > );
				_shPtr = std::shared_ptr< T >( pointer, free_p );
#else
				if( size > 0 ) {
					typedef typename functions::template safe_numa_free< T > FreeFunctor;
					const FreeFunctor free_f( size );
					_shPtr = std::shared_ptr< T >( pointer, free_f );
				} else {
					const auto free_p = &( functions::template safe_free< T > );
					_shPtr = std::shared_ptr< T >( pointer, free_p );
				}
#endif
			};

			/**
			 * Copies an \a other AutoDeleter. The underlying pointer will only be freed
			 * if at least this new AutoDeleter and the \a other AutoDeleter are
			 * destroyed. (The preceding says `at least' because other copies may have
			 * been made previously.)
			 *
			 * @throws std::bad_alloc If the system cannot allocate enough memory.
			 */
			AutoDeleter( const AutoDeleter< T > &other ) : _shPtr( other._shPtr ) {}

			/**
			 * Creates an AutoDeleter from a temporary instance by stealing its
			 * resources.
			 */
			AutoDeleter( AutoDeleter< T > &&other ) noexcept {
				_shPtr = std::move( other._shPtr );
			}

			/** Signals auto-deletion no longer is necessary. */
			void clear() noexcept {
				_shPtr.reset();
			}

			/**
			 * Relies on std::move. Equals-operator only works on temporary RHS.
			 */
			AutoDeleter< T > & operator=( AutoDeleter< T > &&other ) {
				_shPtr = std::move( other._shPtr );
				return *this;
			}

			/**
			 * Relies on copying the underlying shared pointer.
			 */
			AutoDeleter< T > & operator=( const AutoDeleter< T > &other ) {
				_shPtr = other._shPtr;
				return *this;
			}
		};

	} // namespace utils

} // namespace alp

#endif

