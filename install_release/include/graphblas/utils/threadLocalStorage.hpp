
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
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_UTILS_THREADLOCALSTORAGE
#define _H_GRB_UTILS_THREADLOCALSTORAGE

#include <iostream>  //std::cerr
#include <stdexcept> //std::runtime_error

#include <pthread.h> //pthread_key_{t,create,destroy}; pthread_{get,set}specific

/**
 * Internal function that should not emit publicly visible symbols. Used to wrap
 * the standard delete operator which is applied on the supplied pointer, after
 * casting that pointer to a pointer-to-T, with the type name \a T given.
 *
 * @tparam T The type of the value to delete.
 *
 * @param[in] The pointer to delete.
 *
 * @see grb::utils::ThreadLocalStorage This deleter is used as an argument to
 *                                     pthread_key_create when \a autodelete is
 *                                     true.
 */
template< typename T >
static void cpp_deleter( void * data ) {
	delete static_cast< T * >( data );
}

namespace grb {
	namespace utils {

		/**
		 * This is a wrapper around the thread-local storage capabilities provided by
		 * the POSIX Threads standard. It wraps around the type \a pthread_key_t. At
		 * any time while an instance of this class exists, there will be a fully
		 * initialised object associated to that instance. This is guaranteed by the
		 * only way an instance of this class may be constructed.
		 *
		 * To store a new value, see store(). To inspect the currently stored value,
		 * see cload(). To inspect and possibly modify the currently stored value,
		 * see load().
		 *
		 * @tparam T          The type of the value to store. An instance of this
		 *                    type, when deleted, must not throw exceptions.
		 */
		template< typename T >
		class ThreadLocalStorage {

		private:
			/** The POSIX Thread key for the global store. */
			pthread_key_t _key;

			/**
			 * Whether or not on destruction of an instance of this class, whether the
			 * data accessible by load() and cload() should be cleared using the
			 * standard C++ \a delete operator. The default is false, indicating that
			 * the user is responsible for the management of memory given to any
			 * instance of this class.
			 */
			bool autodelete;

			/**
			 * This function deletes the currently stored value, iff \a autodelete is
			 * \a true.
			 */
			void checkDelete() const noexcept {
				if( autodelete ) {
					delete &load();
				}
			}

			/**
			 * Wrapper around setspecific.
			 *
			 * @throws Runtime error whenever the associated call to
			 *         \a pthread_setspecfic fails.
			 */
			void set( T & x ) const {
				const int rc = pthread_setspecific( _key, static_cast< const void * >( &x ) );
				if( rc != 0 ) {
					throw std::runtime_error( "Error during call to "
											  "pthread_setspecific." );
				}
			}

		public:
			/**
			 * The base constructor-- this calls pthread_key_create.
			 *
			 * After calling this constructor, a call to load() or cload() without a
			 * preceding call to store() will lead to undefined behaviour.
			 *
			 * @throws Runtime error whenever the associated call to
			 *         \a pthread_key_create fails.
			 */
			ThreadLocalStorage() : autodelete( false ) {
				const int rc = pthread_key_create( &_key, NULL );
				if( rc != 0 ) {
					throw std::runtime_error( "Error during call to "
											  "pthread_key_create." );
				}
			}

			/**
			 * The base destructor-- this calls pthread_key_delete.
			 *
			 * @throws Runtime error whenever the associated call to
			 *         \a pthread_key_delete fails.
			 */
			~ThreadLocalStorage() noexcept {
				checkDelete();
				const int rc = pthread_key_delete( _key );
				if( rc != 0 ) {
					std::cerr << "[warn] grb::utils::ThreadLocalStorage() "
								 "destructor: could not delete "
								 "pthread_key_t.\n";
				}
			}

			/**
			 * Binds a default value of type \a T to this ThreadLocalStorage.
			 *
			 * The default value is obtained by calling the default constructor of \a T.
			 *
			 * This value is automatically freed on any next call to store() or on
			 * destruction of this ThreadLocalStorage.
			 *
			 * @throws Runtime error whenever the associated call to
			 *         \a pthread_setspecfic fails.
			 */
			void store() {
				checkDelete();
				autodelete = true;
				T * x = new T();
				set( *x );
			}

			/**
			 * Binds a new given value local to this ThreadLocalStorage.
			 *
			 * After a call to this function, calls to load() and cload() are legal.
			 *
			 * @param[in] x The value to store.
			 *
			 * \warning The user must make sure that the data corresponding to the
			 *          stored value \a x remains valid for at least the lifetime
			 *          of this instance of ThreadLocalStorage. In particular, a
			 *          user should never store a temporary.
			 *
			 * @throws Runtime error whenever the associated call to
			 *         \a pthread_setspecfic fails.
			 */
			void store( T & x ) {
				checkDelete();
				autodelete = false;
				set( x );
			}

			/** @return A reference to the value stored at this thread. */
			T & load() const noexcept {
				void * const pointer = pthread_getspecific( _key );
				return *static_cast< T * >( pointer );
			}

			/** @return A const-reference to the value stored at this thread. */
			const T & cload() const noexcept {
				return load();
			}

		}; // end ThreadLocalStorage

	} // namespace utils
} // namespace grb

#endif // _H_GRB_UTILS_THREADLOCALSTORAGE
