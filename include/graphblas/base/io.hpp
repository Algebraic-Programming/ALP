
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
 * @date 21st of February, 2017
 */

#ifndef _H_GRB_IO_BASE
#define _H_GRB_IO_BASE

#include <graphblas/iomode.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/SynchronizedNonzeroIterator.hpp>

#include "matrix.hpp"
#include "vector.hpp"

#include <assert.h>


namespace grb {

	/**
	 * \defgroup IO Data Ingestion and Extraction.
	 *
	 * Provides functions for putting user data into opaque ALP/GraphBLAS
	 * containers, provides functions for extracting data from such containers,
	 * and provides query as well resizing functionalities.
	 *
	 * ALP/GraphBLAS operates on opaque data objects. Users can input data using
	 * grb::buildVector and/or grb::buildMatrix.
	 *
	 * The standard output methods are provided by grb::Vector::cbegin and
	 * grb::Vector::cend, and similarly for grb::Matrix. Iterators provide
	 * parallel output (see #IOMode for a discussion on parallel versus
	 * sequential IO).
	 *
	 * Sometimes it is desired to have direct access to ALP/GraphBLAS memory
	 * area, and to have that memory available even after the ALP/GraphBLAS context
	 * has been destroyed. This functionality is provided by the concept of
	 * <em>pinned containers</em> such as provided by #PinnedVector.
	 *
	 * Containers may be instantiated with default or given requested capacities.
	 * Implementations may reserve a higher capacity, but must allocate at least
	 * the requested amount or otherwise raise an out-of-memory error.
	 *
	 * Capacities are always expressed in terms of number of nonzeroes that the
	 * container can hold. Current capacities of container instances can be queried
	 * using grb::capacity. At any point in time, the actual number of nonzeroes
	 * held within a container is given by grb::nnz and must be less than the
	 * reported capacity.
	 *
	 * To remove all nonzeroes from a container, see grb::clear. The use of this
	 * function does not affect a container's capacity.
	 *
	 * Capacities can be resized after a container has been instantiated by use of
	 * grb::resize. Smaller capacities may or may not yield a reduction of memory
	 * used -- this depends on the implementation, and specifically on the memory
	 * usage semantics it defines.
	 *
	 * After instantiation, the size of a container cannot be modified. The size
	 * is retrieved through grb::size for vectors, and through grb::nrows as well
	 * as grb::ncols for matrices.
	 *
	 * In the above, implementation can also be freely substituted with backend,
	 * in that a single implementation can provide multiple backends that define
	 * different performance and memory semantics.
	 *
	 * @{
	 */

	/**
	 * Request the size of a given vector.
	 *
	 * The dimension is set at construction of the given vector and cannot be
	 * changed after instantiation.
	 *
	 * A call to this function shall always succeed.
	 *
	 * @tparam DataType The type of elements contained in the vector \a x.
	 * @tparam backend  The backend of the vector \a x.
	 *
	 * \internal
	 *    @tparam Coords How sparse coordinates are stored.
	 * \endinternal
	 *
	 * @param[in] x The vector of which to retrieve the size.
	 *
	 * @returns The size of the vector \a x.
	 *
	 * This function shall not raise exceptions.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a x unchanged.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note This specification forces implementations and backends to cache the
	 *       size of a vector so that it can be immediately returned. By RAII
	 *       principles, given containers, on account of being instantiated and
	 *       passed by reference, indeed must have a size that can be immediately
	 *       returned.
	 */
	template<
		typename DataType,
		Backend backend, typename Coords
	>
	size_t size( const Vector< DataType, backend, Coords > &x ) noexcept {
#ifndef NDEBUG
		const bool may_not_call_base_size = false;
#endif
		(void) x;
		assert( may_not_call_base_size );
		return SIZE_MAX;
	}

	/**
	 * Requests the row size of a given matrix.
	 *
	 * The row size is set at construction of the given matrix and cannot be
	 * changed after instantiation.
	 *
	 * A call to this function shall always succeed.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * @param[in] A The matrix of which to retrieve the row size.
	 *
	 * @returns The number of rows of \a A.
	 *
	 * This function shall not raise exceptions.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a A unchanged.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note This specification forces implementations and backends to cache the
	 *       row size of a matrix so that it can be immediately returned. By RAII
	 *       principles, given containers, on account of being instantiated and
	 *       passed by reference, indeed must have a size that can be immediately
	 *       returned.
	 */
	template< typename InputType, Backend backend >
	size_t nrows( const Matrix< InputType, backend > &A ) noexcept {
#ifndef NDEBUG
		const bool may_not_call_base_nrows = false;
#endif
		(void) A;
		assert( may_not_call_base_nrows );
		return SIZE_MAX;
	}

	/**
	 * Requests the column size of a given matrix.
	 *
	 * The column size is set at construction of the given matrix and cannot be
	 * changed after instantiation.
	 *
	 * A call to this function shall always succeed.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * @param[in] A The matrix of which to retrieve the column size.
	 *
	 * @returns The number of columns of \a A.
	 *
	 * This function shall not raise exceptions.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a A unchanged.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note This specification forces implementations and backends to cache the
	 *       column size of a matrix so that it can be immediately returned. By
	 *       RAII principles, given containers, on account of being instantiated
	 *       and passed by reference, indeed must have a size that can be
	 *       immediately returned.
	 */
	template< typename InputType, Backend backend >
	size_t ncols( const Matrix< InputType, backend > &A ) noexcept {
#ifndef NDEBUG
		const bool may_not_call_base_ncols = false;
#endif
		(void) A;
		assert( may_not_call_base_ncols );
		return SIZE_MAX;
	}

	/**
	 * Queries the capacity of the given ALP/GraphBLAS container.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * \internal
	 *    @tparam Coords How sparse coordinates are stored.
	 * \endinternal
	 *
	 * @param[in] x The vector whose capacity is requested.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a x unchanged.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note Backends thus are forced to cache current cacacities and immediately
	 *       return those. By RAII principles, given containers on account of
	 *       being instantiated, must have a capacity that can be immediately
	 *       returned.
	 */
	template< typename InputType, Backend backend, typename Coords >
	size_t capacity( const Vector< InputType, backend, Coords > &x ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_vector_capacity = false;
#endif
		assert( should_not_call_base_vector_capacity );
		(void) x;
		return SIZE_MAX;
	}

	/**
	 * Queries the capacity of the given ALP/GraphBLAS container.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend   The backend of the matrix \a A.
	 *
	 * @param[in] A The matrix whose capacity is requested.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions.
	 *
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a A untouched.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note Backends thus are forced to cache current cacacities and immediately
	 *       return those. By RAII principles, given containers on account of
	 *       being instantiated, must have a capacity that can be immediately
	 *       returned.
	 */
	template< typename InputType, Backend backend >
	size_t capacity( const Matrix< InputType, backend > &A ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_matrix_capacity = false;
#endif
		assert( should_not_call_base_matrix_capacity );
		(void) A;
		return SIZE_MAX;
	}

	/**
	 * Request the number of nonzeroes in a given vector.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * \internal
	 *    @tparam Coords How sparse coordinates are stored.
	 * \endinternal
	 *
	 * @param[in] x The vector whose current number of nonzeroes is requested.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions.
	 *
	 * @returns The number of nonzeroes in \a x.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a A untouched.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note Backends thus are forced to cache the current number of nonzeroes and
	 *       immediately return that cached value.
	 */
	template< typename DataType, Backend backend, typename Coords >
	size_t nnz( const Vector< DataType, backend, Coords > &x ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_vector_nnz = false;
#endif
		(void) x;
		assert( should_not_call_base_vector_nnz );
		return SIZE_MAX;
	}

	/**
	 * Retrieve the number of nonzeroes contained in this matrix.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * @param[in] A The matrix whose current number of nonzeroes is requested.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions.
	 *
	 * @returns The number of nonzeroes that \a A contains.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * A call to this function:
	 *    -# completes in \f$ \Theta(1) \f$ work.
	 *    -# moves \f$ \Theta(1) \f$ intra-process data.
	 *    -# moves \f$ 0 \f$ inter-process data.
	 *    -# does not require inter-process reduction.
	 *    -# leaves memory requirements of \a A untouched.
	 *    -# does not make system calls, and in particular shall not allocate or
	 *       free any dynamic memory.
	 * \endparblock
	 *
	 * \note This is a getter function which has strict performance semantics that
	 *       are \em not backend-specific.
	 *
	 * \note Backends thus are forced to cache the current number of nonzeroes and
	 *       immediately return that cached value.
	 */
	template< typename InputType, Backend backend >
	size_t nnz( const Matrix< InputType, backend > &A ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_matrix_nnz = false;
#endif
		(void) A;
		assert( should_not_call_base_matrix_nnz );
		return SIZE_MAX;
	}

	/**
	 * Clears a given vector of all nonzeroes.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * \internal
	 *    @tparam Coords How sparse coordinates are stored.
	 * \endinternal
	 *
	 * @param[in,out] x The vector of which to remove all values.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions. That clearing a container should never fail is also an implied
	 * requirement of the specification of #grb::resize.
	 *
	 * On function exit, this vector contains zero nonzeroes. The vector size
	 * as well as its nonzero capacity remain unchanged.
	 *
	 * @return grb::SUCCESS This function cannot fail.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * The backend must:
	 *    -# define cost in terms of work
	 *    -# define intra-process data movement costs
	 *    -# define inter-process data movement costs
	 *    -# define inter-process synchronisation requirements
	 *    -# define memory storage requirements and may define
	 *       this in terms of \a new_nz.
	 *    -# define whether system calls may be made and in particular whether
	 *       dynamic memory management may occor.
	 * \endparblock
	 *
	 * \warning Calling clear shall not clear any dynamically allocated
	 *          memory associated with \a x.
	 *
	 * \note Even #grb::resize may or may not free dynamically allocated memory
	 *       associated with \a x-- depending on the memory usage semantics defined
	 *       on a per-backend basis, this is optional.
	 *
	 * \note Only the destruction of \a x would ensure all corresponding memory is
	 *       freed, for all backends.
	 */
	template< typename DataType, Backend backend, typename Coords >
	RC clear( Vector< DataType, backend, Coords > &x ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_vector_clear = false;
#endif
		(void) x;
		assert( should_not_call_base_vector_clear );
		return UNSUPPORTED;
	}

	/**
	 * Clears a given matrix of all nonzeroes.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * @param[in,out] A The matrix of which to remove all nonzero values.
	 *
	 * A call to this function shall always succeed and shall never throw
	 * exceptions. That clearing a container should never fail is also an implied
	 * requirement of the specification of #grb::resize.
	 *
	 * On function exit, this matrix contains zero nonzeroes. The matrix
	 * dimensions (i.e., row and column sizes) as well as the nonzero capacity
	 * remains unchanged.
	 *
	 * @return grb::SUCCESS This function cannot fail.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * The backend must:
	 *    -# define cost in terms of work
	 *    -# define intra-process data movement costs
	 *    -# define inter-process data movement costs
	 *    -# define inter-process synchronisation requirements
	 *    -# define memory storage requirements and may define
	 *       this in terms of \a new_nz.
	 *    -# define whether system calls may be made and in particular whether
	 *       dynamic memory management may occor.
	 * \endparblock
	 *
	 * \warning Calling clear may not clear any dynamically allocated
	 *          memory associated with \a A.
	 *
	 * \note Depending on the memory usage semantics defined on a per-backend
	 *       basis, grb::resize may or may not free dynamically allocated memory
	 *       associated with \a A.
	 *
	 * \note Only the destruction of \a A would ensure all corresponding memory is
	 *       freed, for all backends.
	 */
	template< typename InputType, Backend backend >
	RC clear( Matrix< InputType, backend > &A ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_matrix_clear = false;
#endif
		(void) A;
		assert( should_not_call_base_matrix_clear );
		return UNSUPPORTED;
	}

	/**
	 * Resizes the nonzero capacity of this vector. Any current contents of the
	 * vector are \em not retained.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * \internal
	 *    @tparam Coords How sparse coordinates are stored.
	 * \endinternal
	 *
	 * @param[out]   x   The vector whose capacity is to be resized.
	 * @param[in] new_nz The number of nonzeroes this vector is to contain. After
	 *                   a successful call, the container has, at minimum, space
	 *                   for \a new_nz nonzeroes.
	 *
	 * The requested \a new_nz must be smaller than or equal to the size of \a x.
	 *
	 * Even for non-successful calls to this function, the vector after the call
	 * shall not contain any nonzeroes; only if #grb::PANIC is returned shall the
	 * resulting state of \a x be undefined.
	 *
	 * The size of this vector is fixed. By a call to this function, only the
	 * maximum number of nonzeroes that the vector may contain can be adapted.
	 *
	 * If the vector has size zero, all calls to this function will be equivalent
	 * to a call to grb::clear. In particular, any value for \a new_nz shall be
	 * ignored, even ones that would normally be considered illegal (which would
	 * be any nonzero value in the case of an empty container).
	 *
	 * A request for less capacity than currently already may be allocated, may
	 * or may not be ignored. A backend
	 *   1. must define memory usage semantics that may be proportional
	 *      to the requested capacity, and therefore must free any memory that the
	 *      user has deemed unnecessary. However, a backend
	 *   2. could define memory usage semantics that are \em not proportional to
	 *      the requested capacity, and in that case a performant implementation
	 *      may choose not to free memory that the user has deemed unnecessary.
	 *
	 * @returns ILLEGAL  When \a new_nz is larger than admissable and \a x was
	 *                   non-empty. The vector \a x is cleared, but its capacity
	 *                   remains unchanged.
	 * @returns OUTOFMEM When the required memory memory could not be allocated.
	 *                   The vector \a x is cleared, but its capacity remains
	 *                   unchanged.
	 * @returns SUCCESS  If \a x is empty (i.e., has #grb::size zero).
	 * @returns PANIC    When allocation fails for any other reason. The vector
	 *                   \a x, as well as ALP/GraphBLAS, enters an undefined
	 *                   state.
	 * @returns SUCCESS  If \a x is non-empty and when sufficient capacity for
	 *                   the resize operation was available. The vector \a x has
	 *                   obtained a capacity of at least \a new_nz \em while all
	 *                   nonzeroes it previously contained, if any, are cleared.
	 *
	 * \parblock
	 * \par Performance semantics.
	 * The backend must:
	 *    -# define cost in terms of work
	 *    -# define intra-process data movement costs
	 *    -# define inter-process data movement costs
	 *    -# define inter-process synchronisation requirements
	 *    -# define memory storage requirements and may define
	 *       this in terms of \a new_nz.
	 *    -# define whether system calls may be made and in particular whether
	 *       dynamic memory management may occor.
	 * \endparblock
	 *
	 * \warning For most implementations, this function will indeed imply system
	 *          calls, as well as \f$ \Theta( \mathit{new\_nz} ) \f$ work and data
	 *          movement costs. It is thus to be considered an expensive function,
	 *          and should be used sparingly and only when absolutely necessary.
	 */
	template<
		typename InputType,
		Backend backend, typename Coords
	>
	RC resize( Vector< InputType, backend, Coords > &x, const size_t new_nz ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_vector_resize = false;
#endif
		(void) x;
		(void) new_nz;
		assert( should_not_call_base_vector_resize );
		return UNSUPPORTED;
	}

	/**
	 * Resizes the nonzero capacity of this matrix. Any current contents of the
	 * matrix are \em not retained.
	 *
	 * @tparam InputType The type of elements contained in the matrix \a A.
	 * @tparam backend  The backend of the matrix \a A.
	 *
	 * @param[out]   A   The matrix whose capacity is to be resized.
	 * @param[in] new_nz The number of nonzeroes this matrix is to contain. After
	 *                   a successful call, the container will have space for <em>
	 *                   at least</em> \a new_nz nonzeroes.
	 *
	 * The requested \a new_nz must be smaller or equal to product of the number
	 * of rows and columns.
	 *
	 * After a call to this function, the matrix shall not contain any nonzeroes.
	 * This is the case even after an unsuccessful call, with the exception for
	 * cases where #grb::PANIC is returned-- see below.
	 *
	 * The size of this matrix is fixed. By a call to this function, only the
	 * maximum number of nonzeroes that the matrix may contain can be adapted.
	 *
	 * If the matrix has size zero, meaning either zero rows or zero columns (or,
	 * as the preceding implies, both), then all calls to this function will be
	 * equivalent to a call to grb::clear. In particular, any value of \a new_nz
	 * shall be ignored, even ones that would normally be considered illegal
	 * (which would be any nonzero value in the case of an empty container).
	 *
	 * A request for less capacity than currently already may be allocated,
	 * may or may not be ignored. A backend
	 *   1. must define memory usage semantics that may be proportional to the
	 *      requested capacity, and therefore must free any memory that the user
	 *      has deemed unnecessary. However, a backend
	 *   2. could define memory usage semantics that are \em not proportional to
	 *      the requested capacity, and in that case a performant implementation
	 *      may choose not to free memory that the user has deemed unnecessary.
	 *
	 * \note However, useful implementations will almost surely define storage
	 *       costs that are proportional to \a new_nz, and in such cases resizing
	 *       to smaller capacity must indeed free up unused memory.
	 *
	 * @returns ILLEGAL  When \a new_nz is larger than admissable and \a A was
	 *                   non-empty. The capacity of \a A remains unchanged while
	 *                   its contents have been cleared.
	 * @returns OUTOFMEM When the required memory memory could not be allocated.
	 *                   The capacity of \a A remains unchanged while its contents
	 *                   have been cleared.
	 * @returns PANIC    When allocation fails for any other reason. The matrix
	 *                   \a A as well as ALP/GraphBLAS, enters an undefined state.
	 * @returns SUCCESS  If \a A is non-empty and when sufficient capacity for
	 *                   resizing was available. The matrix \a A has obtained the
	 *                   requested (or a larger) capacity. Its previous contents,
	 *                   if any, have been cleared.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *    -# the backend must define cost in terms of work
	 *    -# the backend must define intra-process data movement costs
	 *    -# the backend must define inter-process data movement costs
	 *    -# the backend must define memory storage requirements and may define
	 *       this in terms of \a new_nz.
	 *    -# the backend must define whether system calls may be made.
	 * \endparblock
	 *
	 * \warning For useful backends, this function will indeed imply system calls
	 *          and incur \f$ \Theta( \mathit{new\_nz} ) \f$ work and data movement
	 *          costs. It is thus to be considered an expensive function, and
	 *          should be used sparingly and only when absolutely necessary.
	 */
	template<
		typename InputType,
		Backend backend
	>
	RC resize( Matrix< InputType, backend > &A, const size_t new_nz ) noexcept {
#ifndef NDEBUG
		const bool should_not_call_base_matrix_resize = false;
#endif
		(void) A;
		(void) new_nz;
		assert( should_not_call_base_matrix_resize );
		return UNSUPPORTED;
	}

	/**
	 * Constructs a dense vector from a container of exactly grb::size(x)
	 * elements. This function aliases to the buildVector routine that takes
	 * an accumulator, using grb::operators::right_assign (thus overwriting
	 * any old contents).
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator,
		Backend backend, typename Coords
	>
	RC buildVector(
		Vector< InputType, backend, Coords > &x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		operators::right_assign< InputType > accum;
		return buildVector< descr >( x, accum, start, end, mode );
	}

	/**
	 * Ingests possibly sparse input from a container to which iterators are
	 * provided. This function dispatches to the buildVector routine that
	 * includes an accumulator, here set to grb::operators::right_assign.
	 * Any existing values in \a x that overlap with newer values will hence
	 * be overwritten.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		class Merger = operators::right_assign< InputType >,
		typename fwd_iterator1, typename fwd_iterator2,
		Backend backend, typename Coords
	>
	RC buildVector( Vector< InputType, backend, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode, const Merger & merger = Merger()
	) {
		operators::right_assign< InputType > accum;
		return buildVector< descr >( x, accum, ind_start, ind_end, val_start, val_end, mode, merger );
	}

	/**
	 * Ingests a set of nonzeroes into a given vector \a x.
	 *
	 * Old values will be overwritten. The given set of nonzeroes must not contain
	 * duplicate nonzeroes that should be stored at the same index.
	 *
	 * \warning Inputs with duplicate nonzeroes when passed into this function will
	 *          invoke undefined behaviour.
	 *
	 * @param[in,out] x     The vector where to ingest nonzeroes into.
	 * @param[in] ind_start Start iterator to the nonzero indices.
	 * @param[in] ind_end   End iterator to the nonzero indices.
	 * @param[in] val_start Start iterator to the nonzero values.
	 * @param[in] val_end   End iterator to the nonzero values.
	 * @param[in] mode      Whether sequential or parallel ingestion is requested.
	 *
	 * The containers the two iterator pairs point to must contain an equal number
	 * of elements. Any pre-existing nonzeroes that do not overlap with any nonzero
	 * between \a ind_start and \a ind_end will remain unchanged.
	 *
	 * \parblock
	 * \par Performance semantics:
	 * A call to this function
	 *   -# comprises \f$ \mathcal{O}( n ) \f$ work where \a n is the number of
	 *      elements pointed to by the given iterator pairs. This work may be
	 *      distributed over multiple user processes.
	 *   -# results in at most \f$   n \mathit{sizeof}( T ) +
	 *                               n \mathit{sizeof}( U ) +
	 *                               n \mathit{sizeof}( \mathit{InputType} ) +
	 *                             2 n \mathit{sizeof}( \mathit{bool} ) \f$
	 *      bytes of data movement, where \a T and \a U are the underlying data
	 *      types of the input iterators. These costs may be distributed over
	 *      multiple user processes.
	 *   -# inter-process communication costs are \f$ \mathcal{O}(n) g + l \f$.
	 *   -# if the capacity of this vector is not large enough to hold \a n
	 *      elements, a call to this function may allocate
	 *         \f$ \mathcal{O}( n ) \f$
	 *      new bytes of memory which \em may be distributed over multiple user
	 *      processes.
	 *   -# if the capacity of this vector is not large enough to hold \a n
	 *      elements, a call to this function may result in system calls at any of
	 *      the user processes.
	 *   -# If the IOMode is sequential, then the work and data movement costs are
	 *      incurred <em>per user process</em> and will not be distributed. In this
	 *      case the inter-process communication costs will, however, be zero.
	 *   -# if the IOMode is parallel, then a good implementation under a uniformly
	 *      randomly distributed input incurs an inter-process communication cost
	 *      of expected value \f$ n/p g + l \f$. The best-case inter-process cost
	 *      is \f$ (p-1)g + l \f$.
	 * \endparblock
	 *
	 * @returns grb::SUCCESS When ingestion has completed successfully.
	 * @returns grb::ILLEGAL When a nonzero has an index larger than grb::size(x).
	 * @returns grb::PANIC   If an unmitigable error has occured during ingestion.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		class Merger = operators::right_assign< InputType >,
		typename fwd_iterator1, typename fwd_iterator2,
		Backend backend, typename Coords
	>
	RC buildVectorUnique( Vector< InputType, backend, Coords > &x,
		fwd_iterator1 ind_start, const fwd_iterator1 ind_end,
		fwd_iterator2 val_start, const fwd_iterator2 val_end,
		const IOMode mode
	) {
		return buildVector< descr | descriptors::no_duplicates >( x,
			ind_start, ind_end,
			val_start, val_end,
			mode );
	}

	/**
	 * Assigns nonzeroes to the matrix from a coordinate format.
	 *
	 * Invalidates any prior existing content. Disallows different nonzeroes
	 * to have the same row and column coordinates; input must consist out of
	 * unique triples. See #buildMatrix for an alternate function that does
	 * not have these restrictions-- at the cost of lower performance.
	 *
	 * \warning Calling this function with duplicate input coordinates will
	 *          lead to undefined behaviour.
	 *
	 * @tparam descr         The descriptor used. The default is
	 *                       #grb::descriptors::no_operation, which means that
	 *                       no pre- or post-processing of input or input is
	 *                       performed.
	 * @tparam fwd_iterator1 The type of the row index iterator.
	 * @tparam fwd_iterator2 The type of the column index iterator.
	 * @tparam fwd_iterator3 The type of the nonzero value iterator.
	 * @tparam length_type   The type of the number of elements in each iterator.
	 *
	 * The iterators will only be used to read from, never to assign to.
	 *
	 * @param[in] I  A forward iterator to \a cap row indices.
	 * @param[in] J  A forward iterator to \a cap column indices.
	 * @param[in] V  A forward iterator to \a cap nonzero values.
	 * @param[in] nz The number of items pointed to by \a I, \a J, \em and \a V.
	 *
	 * @return grb::MISMATCH -# when an element from \a I dereferences to a value
	 *                          larger than the row dimension of this matrix, or
	 *                       -# when an element from \a J dereferences to a value
	 *                          larger than the column dimension of this matrix.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return grb::OVERFLW  When the internal data type used for storing the
	 *                       number of nonzeroes is not large enough to store
	 *                       the number of nonzeroes the user wants to assign.
	 *                       When this error code is returned the state of this
	 *                       container will be as though this function was never
	 *                       called; however, the given forward iterators may
	 *                       have been copied and the copied iterators may have
	 *                       incurred multiple increments and dereferences.
	 * @return grb::SUCCESS  When the function completes successfully.
	 *
	 * \parblock
	 * \par Performance semantics.
	 *        -# This function contains
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ amount of work.
	 *        -# This function may dynamically allocate
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of memory.
	 *        -# A call to this function will use \f$ \mathcal{O}(m+n) \f$ bytes
	 *           of memory beyond the memory in use at the function call entry.
	 *        -# This function will copy each input forward iterator at most
	 *           \em once; the three input iterators \a I, \a J, and \a V thus
	 *           may have exactly one copyeach, meaning that all input may be
	 *           traversed only once.
	 *        -# Each of the at most three iterator copies will be incremented
	 *           at most \f$ \mathit{nz} \f$ times.
	 *        -# Each position of the each of the at most three iterator copies
	 *           will be dereferenced exactly once.
	 *        -# This function moves
	 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of data.
	 *        -# This function will likely make system calls.
	 * \endparblock
	 *
	 * \warning This is an expensive function. Use sparingly and only when
	 *          absolutely necessary.
	 *
	 * \note Streaming input can be implemented by supplying buffered
	 *       iterators to this GraphBLAS implementation.
	 *
	 * \note The functionality herein described is exactly that of buildMatrix,
	 *       though with stricter input requirements. These requirements allow
	 *       much faster construction.
	 *
	 * \note No masked version of this variant is provided. The use of masks in
	 *       matrix construction is costly and the user is referred to the
	 *       costly buildMatrix() function instead.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1 = const size_t * __restrict__,
		typename fwd_iterator2 = const size_t * __restrict__,
		typename fwd_iterator3 = const InputType * __restrict__,
		typename length_type = size_t,
		Backend implementation = config::default_backend >
	RC buildMatrixUnique(
		Matrix< InputType, implementation > &A,
		fwd_iterator1 I, fwd_iterator1 I_end,
		fwd_iterator2 J, fwd_iterator2 J_end,
		fwd_iterator3 V, fwd_iterator3 V_end,
		const IOMode mode
	) {
		// derive synchronized iterator
		auto start = utils::makeSynchronized( I, J, V, I_end, J_end, V_end );
		const auto end = utils::makeSynchronized( I_end, J_end, V_end, I_end, J_end, V_end );

		// defer to other signature
		return buildMatrixUnique< descr >( A, start, end, mode );
	}

	/**
	 * Alias that transforms a set of pointers and an array length to the
	 * buildMatrixUnique variant based on iterators.
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1 = const size_t * __restrict__,
		typename fwd_iterator2 = const size_t * __restrict__,
		typename fwd_iterator3 = const InputType * __restrict__,
		typename length_type = size_t,
		Backend implementation = config::default_backend >
	RC buildMatrixUnique( Matrix< InputType, implementation > &A,
		fwd_iterator1 I, fwd_iterator2 J, fwd_iterator3 V,
		const size_t nz, const IOMode mode
	) {
		return buildMatrixUnique< descr >( A,
			I, I + nz,
			J, J + nz,
			V, V + nz,
			mode
		);
	}

	/** Version of the above #buildMatrixUnique that handles \a NULL value pointers. */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1 = const size_t * __restrict__,
		typename fwd_iterator2 = const size_t * __restrict__,
		typename length_type = size_t,
		Backend implementation = config::default_backend >
	RC buildMatrixUnique( Matrix< InputType, implementation > & A, fwd_iterator1 I, fwd_iterator2 J, const length_type nz, const IOMode mode ) {
		// derive synchronized iterator
		auto start = utils::makeSynchronized( I, J, I + nz, J + nz );
		const auto end = utils::makeSynchronized( I + nz, J + nz, I + nz, J + nz );

		// defer to other signature
		return buildMatrixUnique< descr >( A, start, end, mode );
	}

	/**
	 * Version of buildMatrixUnique that works by supplying a single iterator
	 * (instead of three).
	 *
	 * This is useful in cases where the input is given as a single struct per
	 * nonzero, whatever this struct may be exactly, as opposed to multiple
	 * containers for row indices, column indices, and nonzero values.
	 *
	 * This GraphBLAS implementation provides both input modes since which one is
	 * more appropriate (and performant!) depends mostly on how the data happens
	 * to be stored in practice.
	 *
	 * @tparam descr          The currently active descriptor.
	 * @tparam InputType      The value type the output matrix expects.
	 * @tparam fwd_iterator   The iterator type.
	 * @tparam implementation For which backend a matrix is being read.
	 *
	 * The iterator \a fwd_iterator, in addition to being STL-compatible, must
	 * support the following three public functions:
	 *  -# <tt>S fwd_iterator.i();</tt> which returns the row index of the current
	 *     nonzero;
	 *  -# <tt>S fwd_iterator.j();</tt> which returns the columnindex of the
	 *     current nonzero;
	 *  -# <tt>V fwd_iterator.v();</tt> which returns the nonzero value of the
	 *     current nonzero.
	 *
	 * It also must provide the following public typedefs:
	 *  -# <tt>fwd_iterator::row_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::column_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::nonzero_value_type</tt>
	 *
	 * This means a specialised iterator is required for use with this function.
	 * See, for example, grb::utils::internal::MatrixFileIterator.
	 *
	 * @param[out]   A   The matrix to be filled with nonzeroes from \a start to
	 *                   \a end.
	 * @param[in]  start Iterator pointing to the first nonzero to be added.
	 * @param[in]   end  Iterator pointing past the last nonzero to be added.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator,
		Backend implementation = config::default_backend
	>
	RC buildMatrixUnique( Matrix< InputType, implementation > & A,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		(void)A;
		(void)start;
		(void)end;
		(void)mode;
		return PANIC;
	}

	/**
	 * Function that returns a unique ID for a given non-empty container.
	 *
	 * \note An empty container is either a vector of size 0 or a matrix with one
	 *       of its dimensions equal to 0.
	 *
	 * The ID is unique across all currently valid container instances. If
	 * \f$ n \f$ is the number of such valid instances, the returned ID
	 * may \em not be strictly smaller than \f$ n \f$ -- i.e., implementations
	 * are not required to maintain consecutive IDs (nor would this be possible
	 * if IDs are to be reused).
	 *
	 * The use of <tt>uintptr_t</tt> to represent IDs guarantees that, at any time
	 * during execution, there can never be more initialised containers than can be
	 * assigned an ID. Therefore this specification demands that a call to this
	 * function never fails.
	 *
	 * An ID, once given, may never change during the life-time of the given
	 * container. I.e., multiple calls to this function using the same argument
	 * must return the same ID.
	 *
	 * If the program calling this function is deterministic, then it must assign
	 * the exact same IDs across different runs.
	 *
	 * If the backend supports multiple user processes, the IDs obtained for the
	 * same containers but across different processes, may differ. However, across
	 * the same run of a deterministic program, the IDs returned within any single
	 * user process must, as per the preceding requirement, be the same across
	 * different runs that are executed using the same number of user processes.
	 *
	 * @param[in] x A valid non-empty ALP container to retrieve a unique ID for.
	 *
	 * \note If \a x is invalid or empty then a call to this function results in
	 *       undefined behaviour.
	 *
	 * @returns The unique ID corresponding to \a x.
	 *
	 * \warning The returned ID is not the same as a pointer to \a x, since, for
	 *          example, two containers may be swapped via <tt>std::swap</tt>. In
	 *          such a case, the IDs of the two containers are swapped also.
	 *
	 * \note Another example is when move semantics are invoked, e.g., when a
	 *       temporary container is copied into another just before it would be
	 *       destroyed. Via move semantics the remaining container is in fact not a
	 *       copy of the temporary one, which would have caused their IDs to be
	 *       different. Instead, the remaining container has taken over the
	 *       ownership of the to-be destroyed one, retaining its ID.
	 *
	 * \note For the purposes of defining determinism of ALP programs, and perhaps
	 *       superfluously, two program which only differ by one constructing a
	 *       matrix instead of the other constructing a vector, are not considered
	 *       to be the same program; i.e., implementations are allowed to assign
	 *       vector IDs differently from matrix IDs.
	 *       However, as per the guarantee in the preceding, implementations are
	 *       not allowed to run out of IDs to assign by any use of such mechanism.
	 */
	template<
		typename ElementType, typename Coords,
		Backend implementation = config::default_backend
	>
	uintptr_t getID( const Vector< ElementType, implementation, Coords > &x ) {
#ifndef NDEBUG
		const bool this_is_an_invalid_default_implementation = false;
#endif
		assert( this_is_an_invalid_default_implementation );
		return static_cast< uintptr_t >(-1);
	}

	/**
	 * Specialisation of #getID for matrix containers. The same specification
	 * applies.
	 *
	 * @see getID
	 */
	template<
		typename ElementType,
		Backend implementation = config::default_backend
	>
	uintptr_t getID( const Matrix< ElementType, implementation > &x ) {
#ifndef NDEBUG
		const bool this_is_an_invalid_default_implementation = false;
#endif
		assert( this_is_an_invalid_default_implementation );
		return static_cast< uintptr_t >(-1);
	}

	/** @} */

} // namespace grb

#endif // end _H_GRB_IO_BASE

