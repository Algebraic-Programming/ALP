
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
		     * @param[in]  mode  The grb::IOMode. Optional; default is \a parallel.
		     *
		     * \parblock
		     * \par Performance guarantees.
		     *   -# This function moves up to \f$ \mathcal{O}(n) \f$ bytes of
		     *      data.
		     *   -# This function may allocate up to
		     *        \f$ n \mathit{sizeof}(IOType) \f$
		     *      bytes of memory.
		     *   -# If \a mode is grb::SEQUENTIAL, then communication costs of
		     *      up to \f$ \mathcal{O}(gn+l\log(p)) \f$ will be incurred.
		     * \endparblock
		     *
		     * \note A good implementation moves \f$ \Theta(1) \f$ of data and
		     *       performs no new memory allocations.
		     */
			template< typename Coord >
			PinnedVector( const Vector< IOType, implementation, Coord > & vector, const IOMode mode ) { (void)vector; (void)mode; }

	/**
	 * @see free A destructor of PinnedVector shall always call free().
	 */
	~PinnedVector() {}

	/**
	 * Ensures this class can be used as a raw pointer.
	 *
	 * @return A reference to a vector element at local index \a i.
	 *
	 * Whether the reference corresponds to a properly initialised and meaningful
	 * value, the output of the \a mask() function should be checked first.
	 *
	 * \note The only case in which \a mask() may safely never be evaluated is
	 *       when the user is sure the output vector is dense.
	 *
	 * The vector contains \a length elements of type \a IOType, hence \a i must
	 * be greater or equal to zero and strictly smaller than \a length(). The
	 * global index of the returned element with the requested \a i is computed
	 * via a function call to index().
	 *
	 * Only the values stored local to this user process are returned.
	 *
	 * \par Performance guarantees.
	 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
	 *   -# This function does not allocate new memory blocks.
	 *
	 * This function cannot fail.
	 */
	inline IOType & operator[]( const size_t i ) noexcept {
		(void)i;
		return dummy;
	}

	/** @see operator[] This is the const version of the above operator[]. */
	inline const IOType & operator[]( const size_t i ) const noexcept {
		(void)i;
		return dummy;
	}

	/**
	 * Whether the i-th local element from operator[]() contains a nonzero.
	 *
	 * @return \a true if operator[](i) contains a nonzero, \a false otherwise.
	 *
	 * Evaluating \a mask with index \a i returns \a true if and only if there is
	 * a nonzero at the i-th element of operator[](). The index of that nonzero is
	 * given by index(). If \a mask(i) evaluates \a false, the value of
	 * operator[]( i ) will be undefined. Evaluating \a index at that position
	 * \a i, however, remains legal and returns the global index of a zero
	 * element in the pinned vector.
	 *
	 * \par Performance guarantees.
	 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
	 *   -# This function does not allocate new memory blocks.
	 *
	 * This function cannot fail.
	 */
	const bool & mask( const size_t i ) const noexcept {
		(void)i;
		return false_mask;
	}

	/**
	 * @return The length of this vector, in number of elements.
	 *
	 * \par Performance guarantees.
	 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
	 *   -# This function does not allocate new memory blocks.
	 *
	 * This function cannot fail.
	 */
	size_t length() const noexcept {
		return 0;
	}

	/**
	 * Translates the given \a index into #pointer to the index of the global
	 * pinned vector.
	 *
	 * \param[in] local_index A value strictly smaller than \a length().
	 *
	 * @return The global index corresponding to element \a local_index when
	 *         accessing data from \a pointer.
	 *
	 * \warning If \a local_index is invalid, the returned value shall be
	 *          undefined. No error will be thrown, nor will any other type of
	 *          run-time sanity checking be performed.
	 *
	 * \note In particular, this means that if \a length returns 0, then this
	 *       function must \em never be called.
	 *
	 * \par Performance guarantees.
	 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
	 *   -# This function does not allocate new memory blocks.
	 *
	 * This function cannot fail.
	 */
	size_t index( const size_t local_index ) const noexcept {
		(void)local_index;
		return 0;
	}

	/**
	 * Releases any resources tied to this instance either
	 *   1) back to GraphBLAS, or
	 *   2) back to the system.
	 * A subsequent call to any member function except #free will
	 * cause undefined behaviour.
	 *
	 * \note This function may thus safely be called more than once.
	 *
	 * An instance of this class going out of scope will
	 * automatically call this function.
	 *
	 * \par Performance guarantees.
	 *   -# This function moves at most \f$ \mathcal{O}(n) \f$ bytes of data.
	 *   -# This function may de-allocate a memory area of size \f$ \mathcal{O}(n) \f$.
	 *
	 * This function never fails.
	 */
	void free() noexcept {}

}; // namespace grb

} // end namespace ``grb''

#endif // end _H_GRB_BASE_PINNEDVECTOR
