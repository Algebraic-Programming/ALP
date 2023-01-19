
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

#include <limits>

#include <graphblas/backends.hpp>
#include <graphblas/iomode.hpp>

#include "vector.hpp"


namespace grb {

	/** \addtogroup IO
	 *
	 * Provides a mechanism to access GraphBLAS containers from outside of any
	 * GraphBLAS context.
	 *
	 * An instance of \a PinnedVector caches a container's data and returns it
	 * to the user. The user can refer to the returned data until such time the
	 * \a PinnedVector's instance is destroyed, regardless of whether a call to
	 * #grb::finalize occurs, and regardless whether the ALP/GraphBLAS program
	 * executed through the #grb::Launcher had already returned.
	 *
	 * The original container may not be modified or any derived instance of
	 * \a PinnedVector shall become invalid.
	 *
	 * \note It would be strange if an ALP/GraphBLAS container a pinned vector is
	 *       derived from persists-- pinned vectors are designed to be used
	 *       precisely when the original container no longer is in scope.
	 *       Therefore this last remark on invalidation should not matter.
	 *
	 * The PinnedVector abstracts a container over nonzeroes. A nonzero is a pair
	 * of indices and values. One may query for the number of nonzeroes and use
	 *   1. #PinnedVector::getNonzeroValue to retrieve a nonzero value, or
	 *   2. #PinnedVector::getNonzeroIndex to retrieve a nonzero index.
	 *
	 * An instance of the PinnedVector cannot modify the underlying nonzero
	 * structure nor its values.
	 *
	 * \note A performant implementation in fact does \em not copy the container
	 *       data, but provides a mechanism to access the underlying GraphBLAS
	 *       memory whenever it is possible to do so. This memory should remain
	 *       valid even after a call to grb::finalize() is made, and for as long
	 *       as the PinnedVector instance remains valid.
	 *
	 * \note Some implementations may not retain a raw vector. In this case, a
	 *       copy is unavoidable.
	 */
	template< typename IOType, enum Backend implementation >
	class PinnedVector {

		private :

			/**
			 * \internal Dummy false bool with a descriptive name for assertion
			 * failures.
			 */
			static const constexpr bool
				function_was_not_implemented_in_the_selected_backend = false;


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
				assert( function_was_not_implemented_in_the_selected_backend );
			}

			/**
			 * Base constructor for this class.
			 *
			 * This corresponds to pinning an empty vector of zero size in PARALLEL mode.
			 * A call to this function inherits the same performance semantics as
			 * described above.
			 *
			 * Unlike the above, and exceptionally, calling this constructor need not be
			 * a collective operation.
			 */
			PinnedVector() {
				assert( function_was_not_implemented_in_the_selected_backend );
			}

			/**
			 * Destroying a pinned vector will only remove the underlying vector data if
			 * and only if:
			 *   1) the original grb::Vector has been destroyed;
			 *   2) no other PinnedVector instance derived from the same source container
			 *      exists.
			 */
			~PinnedVector() {
				assert( function_was_not_implemented_in_the_selected_backend );
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
			inline size_t size() const noexcept {
				assert( function_was_not_implemented_in_the_selected_backend );
				return 0;
			}

			/**
			 * @returns The number of nonzeroes this pinned vector contains.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			inline size_t nonzeroes() const noexcept {
				assert( function_was_not_implemented_in_the_selected_backend );
				return 0;
			}

			/**
			 * Returns a requested nonzero of the pinned vector.
			 *
			 * @tparam OutputType The value type returned by this function. If this
			 *                    differs from \a IOType and \a IOType is not
			 *                    <tt>void</tt>, then nonzero values will be cast to
			 *                    \a OutputType.
			 *
			 * \warning If \a OutputType and \a IOType is not compatible, then this
			 *          function should not be used.
			 *
			 * @param[in] k   The nonzero ID to return the value of.
			 * @param[in] one (Optional.) In case \a IOType is <tt>void</tt>, which value
			 *                should be returned in lieu of a vector element value. By
			 *                default, this will be a default-constructed instance of
			 *                \a OutputType.
			 *
			 * If \a OutputType cannot be default-constructed, then \a one no longer is
			 * optional.
			 *
			 * A nonzero is a tuple of an index and nonzero value. A pinned vector holds
			 * #nonzeroes() nonzeroes. Therefore, \a k must be less than #nonzeroes().
			 *
			 * @return The requested value.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			template< typename OutputType >
			inline OutputType getNonzeroValue(
				const size_t k, const OutputType one = OutputType()
			) const noexcept {
				(void)k;
				assert( function_was_not_implemented_in_the_selected_backend );
				return one;
			}

			/**
			 * Direct access variation of the general #getNonzeroValue function.
			 *
			 * This variant is only defined when \a IOType is not <tt>void</tt>.
			 *
			 * \warning If, in your application, \a IOType is templated and can be
			 *          <tt>void</tt>, then robust code should use the general
			 *          #getNonzeroValue variant.
			 *
			 * For semantics, including performance semantics, see the general
			 * specification of #getNonzeroValue.
			 *
			 * \note By providing this variant, implementations may avoid the
			 *       requirement thatensure that that \a IOType must be default-
			 *       constructable.
			 */
			inline IOType getNonzeroValue(
				const size_t k
			) const noexcept {
				IOType ret;
				(void)k;
				assert( function_was_not_implemented_in_the_selected_backend );
				return ret;
			}

			/**
			 * Retrieves a nonzero index.
			 *
			 * @param[in] k The nonzero ID to return the index of.
			 *
			 * A nonzero is a tuple of an index and nonzero value. A pinned vector holds
			 * #nonzeroes() nonzeroes. Therefore, \a k must be less than #nonzeroes().
			 *
			 * @return The requested index.
			 *
			 * \par Performance semantics.
			 *   -# This function incurs \f$ \Theta(1) \f$ work.
			 *   -# This function moves \f$ \Theta(1) \f$ bytes of data.
			 *   -# This function does not incur inter-process communication.
			 *   -# This function does not allocate new memory nor makes any other system
			 *      calls.
			 */
			inline size_t getNonzeroIndex(
				const size_t k
			) const noexcept {
				(void)k;
				assert( function_was_not_implemented_in_the_selected_backend );
				return std::numeric_limits< size_t >::max();
			}


	}; // end class grb::PinnedVector

} // end namespace ``grb''

#endif // end _H_GRB_BASE_PINNEDVECTOR

