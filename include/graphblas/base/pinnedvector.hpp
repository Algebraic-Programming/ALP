
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
 * Contains the API for the PinnedVector class.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_BASE_PINNEDVECTOR
#define _H_GRB_BASE_PINNEDVECTOR

#include <graphblas/backends.hpp>
#include <graphblas/iomode.hpp>

#include "vector.hpp"

namespace grb {

	/** \addtogroup IO
	 * Provides a mechanism to access GraphBLAS containers from outside of any
	 * GraphBLAS context.
	 *
	 * Semantically, an instance of \a PinnedVector caches a container's data and
	 * returns it to user space. The user can operate on the returned data until
	 * such time the \a PinnedVector's instance is destroyed. The container may
	 * not be modified or any derived instance of \a PinnedVector shall become
	 * invalid, or, more precisely: the behaviour of an instance of this class
	 * becomes invalid the moment the underlying container is modified without a
	 * preceding call to PinnendMemory::free. Destroying an instance of this
	 * class automatically calls #free explicitly.
	 * Additionally, further use of the GraphBLAS container this instance was
	 * derived from shall become undefined when the user modifies the vector
	 * contents via this PinnedVector. This is only OK to do when the GraphBLAS
	 * container shall \em not be used after modification of its data via an
	 * instance of a PinnedVector.
	 *
	 * A performant implementation in fact does \em not copy the container's
	 * data, but provides a mechanism to access the underlying GraphBLAS memory
	 * whenever it is possible to do so. This memory should remain valid even
	 * after a call to grb::finalize() is made while an instance of
	 * \a PinnedVector is retained for the user to reference.
	 *
	 * \note Some implementations do not retain a raw vector. In this case, a
	 *       copy is unavoidable.
	 *
	 * \note This mechanism takes some inspiration from Java's Native Interface
	 *       and its JVM-to-C memory sharing API.
	 */
	template< typename IOType, enum Backend implementation >
	class PinnedVector {

		private :

			/** \internal Dummy variable to ensure the spec can compile. */
			static constexpr IOType dummy = IOType();

			/** \internal Dummy variable to ensure the spec can compile. */
			static constexpr bool false_mask = false;

		public :

			/**
			 * Pins a given \a vector to a single memory pointer. The pointer
			 * shall remain valid as long as the lifetime of this instance.
			 * The given \a vector must be in unpinned state or an exception
			 * will be thrown.
			 * Pinning may or may not require a memory copy, depending on the
			 * GraphBLAS implementation. If it does not, then destroying this
			 * instance or calling #free on this vector may or may not result
			 * in memory deallocation, depending on whether the underlying
			 * vector still exists or not.
			 *
			 * If one user process calls this constructor, \em all user
			 * processes must do so-- this is a collective call. All member
			 * functions as well as the default destructor are \em not
			 * collective.
			 *
			 * @param[in] vector The vector to pin the memory of.
			 * @param[in]  mode  The grb::IOMode.
			 *
			 * The \a mode argument is \em optional; its default is PARALLEL.
			 *
			 * \parblock
			 * \par Performance semantics (#IOMode::SEQUENTIAL):
			 *   -# This function contains \f$ \Theta(n) \f$ work, where
			 *      \f$ n \f$ is the global length of \a vector.
			 *   -# This function moves up to \f$ \mathcal{O}(n) \f$ bytes of
			 *      data within its process.
			 *   -# This function incurs an inter-process communication cost
			 *      bounded by \f$ \mathcal{O}(ng+\log(p)l) \f$.
			 *   -# This function may allocate \f$ \mathcal{O}(n) \f$ memory
			 *      and (thus) incur system calls.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics (#IOMode::PARALLEL):
			 *   -# This function contains \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ data within its
			 *      process.
			 *   -# This function has no inter-process communication cost.
			 *   -# This function performs no dynamic memory allocations
			 *      and shall not make system calls.
			 * \endparblock
			 */
			template< typename Coord >
			PinnedVector(
				const Vector< IOType, implementation, Coord > &vector,
				const IOMode mode
			) {
				(void)vector;
				(void)mode;
			}

			/**
			 * Base constructor for this class.
			 *
			 * This corresponds to pinning an empty vector of zero size in PARALLEL mode.
			 * A call to this function inherits the same performance semantics as
			 * described above.
			 *
			 * Unlike the above, calling this constructor need not be a collective
			 * operation.
			 */
			PinnedVector() {}

			/**
			 * Destroying a pinned vector will only remove the underlying vector data if
			 * and only if:
			 *   1) the original grb::Vector has been destroyed;
			 *   2) no other PinnedVector instance to the same vector exists.
			 *
			 * @see free A destructor of PinnedVector shall always call free().
			 */
			~PinnedVector() {}

			/**
			 * Returns a requested nonzero of the pinned vector.
			 *
			 * @param[in] k The nonzero ID to return the value of.
			 *
			 * A nonzero is a tuple of an index and nonzero value. A pinned vector holds
			 * #length nonzeroes. Therefore, \a k must be less than #length.
			 *
			 * This function should only be called when #mask( \a k ) returns
			 * <tt>true</tt>, or otherwise undefined behaviour occurs.
			 *
			 * @return If #mask( \a k ) is <tt>true</tt>, a reference to the \a k-th
			 *         nonzero value.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			inline IOType& operator[]( const size_t k ) noexcept {
				(void)k;
				return dummy;
			}

			/** @see operator[] This is the const version of the above operator[]. */
			inline const IOType& operator[]( const size_t k ) const noexcept {
				(void)k;
				return dummy;
			}

			/**
			 * Whether the k-th nonzero from operator[]() contains a nonzero.
			 *
			 * @param[in] k The nonzero ID of which to return whether a value exists.
			 *
			 * The argument \a k must be smaller than #length.
			 *
			 * @return \a true if the requested nonzero exists, and \a false otherwise.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			const bool& mask( const size_t k ) const noexcept {
				(void)k;
				return false_mask;
			}

			/**
			 * @return The length of this vector, in number of elements.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			size_t length() const noexcept {
				return 0;
			}

			/**
			 * Translates the given nonzero ID \a k into a global index.
			 *
			 * \param[in] k The nonzero ID. Must be smaller than #length().
			 *
			 * \note If #length returns 0, then this function must \em never be called.
			 *
			 * @return The global index corresponding to the \a k-th nonzero.
			 *
			 * This function may be called even if #mask( \a k ) returns <tt>false</tt>.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			size_t index( const size_t k ) const noexcept {
				(void) k;
				return 0;
			}

			/**
			 * Releases any resources tied to this instance either
			 *   1) back to ALP/GraphBLAS, or
			 *   2) back to the system.
			 * Case 1) will happen if the original vector still exists. Case 2) will
			 * happen if and only if the original vector no longer exists while also
			 * no other PinnedVector instances derived from that original vector no
			 * longer exist.
			 *
			 * A subsequent call to any member function except #free will
			 * cause undefined behaviour.
			 *
			 * \note This function may thus safely be called more than once.
			 *
			 * An instance of this class going out of scope will automatically call this
			 * function.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs at most \f$ \mathcal{O}(n) \f$ work.
			 *   -# This function moves at most \f$ \mathcal{O}(n) \f$ bytes of data
			 *      within its process.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function may de-allocate memory areas of (combined) size
			 *      \f$ \mathcal{O}(n) \f$.
			 */
			void free() noexcept {}

	}; // namespace grb

} // end namespace ``grb''

#endif // end _H_GRB_BASE_PINNEDVECTOR

