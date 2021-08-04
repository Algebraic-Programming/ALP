
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

namespace grb {

	/**
	 * \defgroup IO Data Ingestion and Extraction.
	 * Provides functions for putting user data into opaque GraphBLAS objects,
	 * and provides functions for extracting data from opaque GraphBLAS objects.
	 *
	 * The GraphBLAS operates on opaque data objects. Users can input data using
	 * grb::buildVector and/or grb::buildMatrixUnique. This group provides free
	 * functions that automatically dispatch to those variants.
	 *
	 * The standard output methods are provided by grb::Vector::cbegin and
	 * grb::Vector::cend, and similarly for grb::Matrix. Iterators provide
	 * parallel output (see #IOMode for a discussion on parallel versus
	 * sequential IO).
	 *
	 * Sometimes it is desired to have direct access to a GraphBLAS memory
	 * area, and to have that memory available even after the GraphBLAS
	 * context has been closed (via grb::finalize). This functionality is
	 * provided by grb::pin_memory.
	 *
	 * @{
	 */

	/** TODO add documentation */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator, Backend backend, typename Coords >
	RC buildVector( Vector< InputType, backend, Coords > & x, fwd_iterator start, const fwd_iterator end, const IOMode mode ) {
		operators::right_assign< InputType > accum;
		return buildVector< descr >( x, accum, start, end, mode );
	}

	/** TODO add documentation */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		class Merger = operators::right_assign< InputType >,
		typename fwd_iterator1,
		typename fwd_iterator2,
		Backend backend,
		typename Coords >
	RC buildVector( Vector< InputType, backend, Coords > & x,
		fwd_iterator1 ind_start,
		const fwd_iterator1 ind_end,
		fwd_iterator2 val_start,
		const fwd_iterator2 val_end,
		const IOMode mode,
		const Merger & merger = Merger() ) {
		operators::right_assign< InputType > accum;
		return buildVector< descr >( x, accum, ind_start, ind_end, val_start, val_end, mode, merger );
	}

	/** TODO add documentation */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		class Merger = operators::right_assign< InputType >,
		typename fwd_iterator1,
		typename fwd_iterator2,
		Backend backend,
		typename Coords >
	RC buildVectorUnique( Vector< InputType, backend, Coords > & x, fwd_iterator1 ind_start, const fwd_iterator1 ind_end, fwd_iterator2 val_start, const fwd_iterator2 val_end, const IOMode mode ) {
		return buildVector< descr | descriptors::no_duplicates >( x, ind_start, ind_end, val_start, val_end, mode );
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
	 * \note By default, the iterator types are raw, unaliased, pointers.
	 *
	 * \warning This means that by default, input arrays are \em not
	 *          allowed to overlap.
	 *
	 * Forward iterators will only be used to read from, never to assign to.
	 *
	 * \note It is therefore both legal and preferred  to pass constant forward
	 *       iterators, as opposed to mutable ones as \a I, \a J, and \a V.
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
	 * \par Performance guarantees.
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
	 *          absolutely necessary
	 *
	 * \note Streaming input can be implemented by supplying buffered
	 *       iterators to this GraphBLAS implementation.
	 *
	 * \note The functionality herein described is exactly that of buildMatrix,
	 *       though with stricter input requirements. These requirements allow
	 *       much faster construction, however.
	 *
	 * \note No masked version of this variant is provided. The use of masks in
	 *       matrix construction is extremely costly and thus the user is
	 *       referred to the equally costly buildMatrix() function instead.
	 *
	 * \todo update documentation
	 */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1 = const size_t * __restrict__,
		typename fwd_iterator2 = const size_t * __restrict__,
		typename fwd_iterator3 = const InputType * __restrict__,
		typename length_type = size_t,
		Backend implementation = config::default_backend >
	RC
	buildMatrixUnique( Matrix< InputType, implementation > & A, fwd_iterator1 I, fwd_iterator1 I_end, fwd_iterator2 J, fwd_iterator2 J_end, fwd_iterator3 V, fwd_iterator3 V_end, const IOMode mode ) {
		// derive synchronized iterator
		auto start = utils::makeSynchronized( I, J, V, I_end, J_end, V_end );
		const auto end = utils::makeSynchronized( I_end, J_end, V_end, I_end, J_end, V_end );

		// defer to other signature
		return buildMatrixUnique< descr >( A, start, end, mode );
	}

	/** TODO documentation */
	template< Descriptor descr = descriptors::no_operation,
		typename InputType,
		typename fwd_iterator1 = const size_t * __restrict__,
		typename fwd_iterator2 = const size_t * __restrict__,
		typename fwd_iterator3 = const InputType * __restrict__,
		typename length_type = size_t,
		Backend implementation = config::default_backend >
	RC buildMatrixUnique( Matrix< InputType, implementation > & A, fwd_iterator1 I, fwd_iterator2 J, fwd_iterator3 V, const size_t nz, const IOMode mode ) {
		return buildMatrixUnique< descr >( A, I, I + nz, J, J + nz, V, V + nz, mode );
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
	 * It also must provide the following public typedefs:
	 *  -# <tt>fwd_iterator::row_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::column_coordinate_type</tt>
	 *  -# <tt>fwd_iterator::nonzero_value_type</tt>
	 * Note that the regular STL-mandated <tt>fwd_iterator</tt> could refer to any
	 * underlying user-defined value, including, for example,
	 *   <tt>std::pair< std::pair< S, S >, V ></tt>
	 * as used by grb::utils::internal::MatrixFileIterator.
	 *
	 * @param[out]   A   The matrix to be filled with nonzeroes from \a start to
	 *                   \a end.
	 * @param[in]  start Iterator pointing to the first nonzero to be added.
	 * @param[in]   end  Iterator pointing past the last nonzero to be added.
	 *
	 * @see buildMatrixUnique for performance guarantees, valid descriptors, and
	 *                        other information not specific to this version only.
	 */
	template< Descriptor descr = descriptors::no_operation, typename InputType, typename fwd_iterator, Backend implementation = config::default_backend >
	RC buildMatrixUnique( Matrix< InputType, implementation > & A, fwd_iterator start, const fwd_iterator end, const IOMode mode ) {
		(void)A;
		(void)start;
		(void)end;
		(void)mode;
		return PANIC;
	}

	/** @} */

} // namespace grb

#endif // end _H_GRB_IO_BASE
