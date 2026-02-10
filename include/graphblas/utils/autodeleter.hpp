
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

#ifndef _H_GRB_UTILS_AUTO_DELETER
#define _H_GRB_UTILS_AUTO_DELETER

#ifndef _GRB_NO_LIBNUMA
#include <numa.h>
#endif

#include <memory>

#include "graphblas/config.hpp"


namespace grb {

	namespace utils {

		namespace internal {

			/**
			 * A collection of pre-defined deleter functions and functors.
			 *
			 * @tparam implementation Which backend the deleter functions apply to.
			 *
			 * This is the default implementation for all backends.
			 *
			 * \note Typically this collection only needs be overridden for targets that
			 *       require specialised memory management routines.
			 */
			template< enum Backend implementation >
			class DeleterFunctions {

				private:

					/** Prevent instantiation. */
					DeleterFunctions() = delete;


				public:

					/**
					 * Frees a generic pointer allocated using malloc or posix_memalign.
					 *
					 * @tparam T The type of the pointer.
					 *
					 * @param[in] pointer Pointer to the memory address to be freed.
					 */
					template< typename T >
					static void safe_free( T * const pointer ) {
						if( pointer != nullptr ) {
							free( pointer );
						}
					}

#ifndef _GRB_NO_LIBNUMA
					/**
					 * Functor that frees a pointer to a memory region allocated via libnuma.
					 *
					 * @tparam T The type of the pointer.
					 */
					template< typename T >
					class safe_numa_free {

						private:

							/** The size of the memory region. */
							size_t size;

							safe_numa_free() = delete;

						public:

							/**
							 * Functor constructor.
							 *
							 * @param[in] size_in The size of the memory region.
							 */
							safe_numa_free( const size_t size_in ) : size( size_in ) {}

							/**
							 * Frees the given memory region allocated via libnuma.
							 *
							 * @param[in] pointer Pointer to the memory region.
							 */
							void operator()( T * const pointer ) {
								numa_free( pointer, size );
							}

					};
#endif
			};

		} // namespace internal

	} // namespace utils

} // namespace grb

// now define user API:

namespace grb {

	namespace utils {

		/**
		 * ALP uses memory regions of two types:
		 *  -# memory or buffers tied to a container; or
		 *  -# memory or buffers tied to global and/or thread-local contexts.
		 *
		 * These memory regions may be used by multiple threads simultaneously, and
		 * may be used by any primitive (that takes the container that owns the data
		 * region as argument).
		 *
		 * ALP memory is always \em initially tied to an ALP context, and an ALP
		 * context may consist of multiple user processes. However, containers may be
		 * \em pinned to memory, in order to escape the termination of the ALP context
		 * in which it was created. For this reason there is not always a single owner
		 * of a memory region. Memory once pinned may not be accessed or used in
		 * exactly the same manner as prior to pinning-- consider, for example,
		 * pinning a container in sequential mode in a distributed-memory parallel
		 * context.
		 *
		 * For these reasons, ALP requires something alike a shared pointer in that it
		 * requires keeping track of all contexts in which a memory region is used:
		 * within ALP contexts, and as part of pinning. It is unlike a shared pointer
		 * in that the stored memory region may be interpreted differently. This class
		 * therefore decouples the interpretation cq. storage of the pointer from the
		 * management of the raw memory area: it only provides reference counting, and
		 * once the count reaches zero, executes the given destructor.
		 *
		 * \note This class is compatible with \a posix_memalign.
		 *
		 * \note This class handles <tt>nullptr</tt> OK.
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
				 * if and only if it is not equal to <tt>nullptr</tt>.
				 *
				 * @param[in] pointer The pointer to free if the reference counter reaches
				 *                    zero.
				 * @param[in] size If \a size is zero, uses <tt>free</tt> as the destructor.
				 *                 Otherwise, it uses the free provided by libnuma.
				 *
				 * \note If ALP is configured without libnuma (<tt>_GRB_NO_LIBNUMA</tt>),
				 *       then \a size is ignored.
				 *
				 * \note If \a pointer is <tt>nullptr</tt> then the destructor shall never
				 *       be a no-op.
				 *
				 * @throws std::bad_alloc If the system cannot allocate enough memory.
				 */
				AutoDeleter( T * const pointer = nullptr, const size_t size = 0 ) {
#ifdef _GRB_NO_LIBNUMA
					(void) size;
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
				}

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
				 * Creates an AutoDeleter from a temporary instance.
				 */
				AutoDeleter( AutoDeleter< T > &&other ) noexcept {
					_shPtr = std::move( other._shPtr );
				}

				/**
				 * Forgets the stored pointer (and decreases its reference counter by one).
				 */
				void clear() noexcept {
					_shPtr.reset();
				}

				/**
				 * Relies on std::move. Equals-operator only works on temporary RHS.
				 */
				AutoDeleter< T > & operator=( AutoDeleter< T > &&other ) {
					clear();
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

				/**
				 * Swaps two auto-deleters.
				 */
				void swap( AutoDeleter< T > &other ) {
					_shPtr = std::swap( other._shPtr );
				}

		};

	} // namespace utils

} // namespace grb

// include specialised DeleterFunctions
#ifdef _GRB_WITH_BANSHEE
 #include "graphblas/banshee/deleters.hpp"
#endif

#endif

