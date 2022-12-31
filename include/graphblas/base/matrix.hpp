
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
 * Specifies the ALP/GraphBLAS matrix container.
 *
 * @author A. N. Yzelman
 * @date 10th of August, 2016
 */

#ifndef _H_GRB_MATRIX_BASE
#define _H_GRB_MATRIX_BASE

#include <iterator>

#include <stddef.h>

#include <utility>

#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>


namespace grb {

	/**
	 * An ALP/GraphBLAS matrix.
	 *
	 * This is an opaque data type that implements the below constructors, member
	 * functions, and destructors.
	 *
	 * @tparam D The type of a nonzero element.
	 *
	 * The given type \a D shall not be an ALP/GraphBLAS object.
	 *
	 * @tparam implementation Allows multiple backends to implement different
	 *                        versions of this data type.
	 *
	 * \internal
	 * @tparam RowIndexType The type used for row indices
	 * @tparam ColIndexType The type used for column indices
	 * @tparam NonzeroIndexType The type used for nonzero indices
	 * \endinternal
	 *
	 * \warning Creating a grb::Matrix of other ALP/GraphBLAS types is not allowed.
	 */
	template<
		typename D, enum Backend implementation,
		typename RowIndexType,
		typename ColIndexType,
		typename NonzeroIndexType
	>
	class Matrix {

		public :

			/** The type of this container. */
			typedef Matrix<
				D, implementation,
				RowIndexType, ColIndexType, NonzeroIndexType
			> self_type;

			/**
			 * A standard iterator for an ALP/GraphBLAS matrix.
			 *
			 * This iterator is used for data extraction only. Hence only this const
			 * version is specified.
			 *
			 * Dereferencing an iterator of this type that is not in end position yields
			 * a pair \f$ (c,v) \f$. The value \a v is of type \a D and corresponds to
			 * the value of the dereferenced nonzero.
			 *
			 * The value \a c is another pair \f$ (i,j) \f$. The values \a i and \a j
			 * are of type <code>size_t</code> and correspond to the coordinate of the
			 * dereferenced nonzero.
			 *
			 * \note `Pair' here corresponds to the regular <code>std::pair</code>.
			 *
			 * \warning Comparing two const iterators corresponding to different
			 *          containers leads to undefined behaviour.
			 *
			 * \warning Advancing an iterator past the end iterator of the container
			 *          it corresponds to, leads to undefined behaviour.
			 *
			 * \warning Modifying the contents of a container makes any use of any
			 *          iterator derived from it incur invalid behaviour.
			 *
			 * \note These are standard limitations of STL iterators.
			 *
			 * In terms of STL, the returned iterator is an <em>forward iterator</em>.
			 * Its performance semantics match that defined by the STL. Backends are
			 * encouraged to specify additional performance semantics as long as they
			 * do not conflict with those of a forward iterator.
			 *
			 * Backends are allowed to return bi-directional or random access iterators
			 * instead of forward iterators.
			 */
			class const_iterator : public std::iterator<
				std::forward_iterator_tag,
				std::pair< std::pair< const size_t, const size_t >, const D >,
				size_t
			> {

				public :

					/**
					 * Standard equals operator
					 *
					 * @returns Whether this iterator and the given \a other iterator are the
					 *          same.
					 */
					bool operator==( const const_iterator &other ) const {
						(void)other;
						return false;
					}

					/**
					 * @returns The negation of operator==()
					 */
					bool operator!=( const const_iterator &other ) const {
						(void)other;
						return true;
					}

					/**
					 * Dereferences the current position of this iterator.
					 *
					 * @returns If this iterator is valid and not in end position, this returns
					 *          an std::pair with in its first field the position of the
					 *          nonzero value, and in its second field the value of the nonzero.
					 *          The position of a nonzero is another std::pair with both the
					 *          first and second field of type <code>size_t</code>.
					 *
					 * \note If this iterator is invalid or in end position, the result is
					 *       undefined.
					 */
					std::pair< const size_t, const D > operator*() const {
						return std::pair< const size_t, const D >();
					}

					/**
					 * Advances the position of this iterator by one.
					 *
					 * If the current position corresponds to the last element in the
					 * container, the new position of this iterator will be its end
					 * position.
					 *
					 * If the current position of this iterator is already the end
					 * position, this iterator will become invalid; any use of invalid
					 * iterators will lead to undefined behaviour.
					 *
					 * @return A reference to this iterator.
					 */
					const_iterator & operator++() {
						return *this;
					}

			};

			/** The value type of elements stored in this matrix. */
			typedef D value_type;

			/**
			 * ALP/GraphBLAS matrix constructor that sets an initial capacity.
			 *
			 * @param[in] rows    The number of rows of the matrix to be instantiated.
			 * @param[in] columns The number of columns of the matrix to be instantiated.
			 * @param[in] nz      The minimum initial capacity of the matrix to be
			 *                    instantiated.
			 *
			 * After successful construction, the resulting matrix has a capacity of
			 * <em>at least</em> \a nz nonzeroes. If either \a rows or \a columns is 0,
			 * then the capacity may instead be 0 as well.
			 *
			 * On errors such as out-of-memory, this constructor may throw exceptions.
			 *
			 * \parblock
			 * \par Performance semantics.
			 * Implementations must define cost semantics across the following
			 * dimensions:
			 *   -# work;
			 *   -# intra-process data movement;
			 *   -# inter-process data movement;
			 *   -# inter-process synchronisations;
			 *   -# memory usage; and
			 *   -# whether system calls, in particular dynamic memory management calls,
			 *      could occur.
			 * \endparblock
			 *
			 * \warning Avoid the use of this constructor within performance critical
			 *          code sections.
			 */
			Matrix( const size_t rows, const size_t columns, const size_t nz ) {
				(void) rows;
				(void) columns;
				(void) nz;
			}

			/**
			 * ALP/GraphBLAS matrix constructor that sets a default initial capacity.
			 *
			 * @param[in] rows        The number of rows in the new matrix.
			 * @param[in] columns     The number of columns in the new matrix.
			 *
			 * The default capacity is the maximum of \a rows and \a columns.
			 *
			 * On errors such as out-of-memory, this constructor may throw exceptions.
			 *
			 * For the full specification, please see the full constructor signature.
			 *
			 * \warning Avoid the use of this constructor within performance critical
			 *          code sections.
			 */
			Matrix( const size_t rows, const size_t columns ) {
				(void)rows;
				(void)columns;
			}

			/**
			 * Copy constructor.
			 *
			 * @param[in] other The matrix to copy.
			 *
			 * This performs a deep copy; a new matrix is allocated with the same
			 * (or larger) capacity as \a other, after which the contents of \a other
			 * are copied into the new instance.
			 *
			 * The use of this constructor is semantically the same as:
			 *
			 *     grb::Matrix< T > newMatrix(
			 *         grb::nrows( other ), grb::ncols( other ),
			 *         grb::capacity( other )
			 *     );
			 *     grb::set( newMatrix, other );
			 *
			 * (Under the condition that all calls are successful.)
			 *
			 * \parblock
			 * \par Performance semantics.
			 * Implementations must define cost semantics across the following
			 * dimensions:
			 *   -# work;
			 *   -# intra-process data movement;
			 *   -# inter-process data movement;
			 *   -# inter-process synchronisations;
			 *   -# memory usage; and
			 *   -# whether system calls, in particular dynamic memory management calls,
			 *      could occur.
			 * \endparblock
			 *
			 * \warning Avoid the use of this constructor within performance critical
			 *          code sections.
			 */
			Matrix( const Matrix<
				D, implementation,
				RowIndexType, ColIndexType, NonzeroIndexType > &other
			) {
				(void)other;
			}

			/**
			 * Move constructor. This will take over the resources of the given \a other
			 * matrix, invalidating the contents of \a other while its contents are now
			 * moved into this instance instead.
			 *
			 * @param[in] other The matrix to move to this new instance.
			 *
			 * \parblock
			 * \par Performance semantics.
			 * This constructor:
			 *        -# entails \f$ \Theta(1) \f$ amount of work;
			 *        -# moves \f$ \Theta(1) \f$ bytes of data within its user process;
			 *        -# moves \f$ 0 \f$ bytes of data between user processes;
			 *        -# shall \em not require synchronisations between user processes;
			 *        -# inherit the memory usage of \a other;
			 *        -# will \em not make system calls and in particular will not free
			 *           nor allocate dynamic memory.
			 * \endparblock
			 */
			Matrix( self_type &&other ) {
				(void)other;
			}

			/**
			 * Move-assignment. This will take over the resources of the given \a other
			 * matrix, invalidating the contents of \a other while its contents are now
			 * moved into this instance instead.
			 *
			 * This will destroy any current contents in this container.
			 *
			 * @param[in] other The matrix contents to move into this instance.
			 *
			 * \parblock
			 * \par Performance semantics.
			 * This constructor:
			 *        -# entails \f$ \Theta(1) \f$ amount of work;
			 *        -# moves \f$ \Theta(1) \f$ bytes of data within its user process;
			 *        -# moves \f$ 0 \f$ bytes of data between user processes;
			 *        -# shall \em not require synchronisations between user processes;
			 *        -# inherit the memory usage of \a other;
			 *        -# will \em not make system calls and in particular will not free
			 *           nor allocate dynamic memory.
			 *
			 * Additionally, the backend-specific cost of the matrix destructor apply.
			 * \endparblock
			 */
			self_type& operator=( self_type &&other ) noexcept {
				*this = std::move( other );
				return *this;
			}

			/**
			 * Matrix destructor.
			 *
			 * \parblock
			 * \par Performance semantics.
			 * Beckends must define cost semantics across the following
			 * dimensions:
			 *   -# work;
			 *   -# intra-process data movement;
			 *   -# inter-process data movement;
			 *   -# inter-process synchronisations;
			 *   -# memory usage; and
			 *   -# whether system calls, in particular dynamic memory management calls,
			 *      could occur.
			 * \endparblock
			 *
			 * \warning Avoid calling destructors from within performance critical
			 *          code sections.
			 */
			~Matrix() {}

			/**
			 * Provides the only mechanism to extract data from a GraphBLAS matrix.
			 *
			 * The order in which nonzero elements are returned is undefined.
			 *
			 * @return An iterator pointing to the first element of this matrix, if any;
			 *         \em or an iterator in end position if this vector contains no
			 *         nonzeroes.
			 *
			 * \note An `iterator in end position' compares equal to the const_iterator
			 *       returned by cend().
			 *
			 * \parblock
			 * \par Performance semantics.
			 * Beckends must define cost semantics across the following
			 * dimensions:
			 *   -# work;
			 *   -# intra-process data movement;
			 *   -# inter-process data movement;
			 *   -# inter-process synchronisations;
			 *   -# memory usage; and
			 *   -# whether system calls, in particular dynamic memory management calls,
			 *      could occur.
			 * \endparblock
			 *
			 * \note This function may make use of a const_iterator that is buffered,
			 *       hence possibly causing its implicitly called constructor to
			 *       allocate dynamic memory.
			 *
			 * \warning Avoid the use of this function within performance critical code
			 *          sections.
			 */
			const_iterator cbegin() const {}

			/**
			 * Same as cbegin().
			 *
			 * Since iterators are only supplied as a data extraction mechanism, there
			 * is no overloaded version of this function that returns a non-const
			 * iterator.
			 */
			const_iterator begin() const {}

			/**
			 * Indicates the end to the elements in this container.
			 *
			 * @return An iterator at the end position of this container.
			 *
			 * \parblock
			 * \par Performance semantics.
			 * Beckends must define cost semantics across the following
			 * dimensions:
			 *   -# work;
			 *   -# intra-process data movement;
			 *   -# inter-process data movement;
			 *   -# inter-process synchronisations;
			 *   -# memory usage; and
			 *   -# whether system calls, in particular dynamic memory management calls,
			 *      could occur.
			 * \endparblock
			 *
			 * \note Even if cbegin() returns a buffered const_iterator that may require
			 *       dynamic memory allocation and additional data movement, this
			 *       specification disallows the same to happen for the construction of
			 *       an iterator in end position.
			 *
			 * \warning Avoid the use of this function within performance critical code
			 *          sections.
			 */
			const_iterator cend() const {}

			/**
			 * Same as cend().
			 *
			 * Since iterators are only supplied as a data extraction mechanism, there
			 * is no overloaded version of this function that returns a non-const
			 * iterator.
			 */
			const_iterator end() const {}

	};

} // end namespace ``grb''

#endif // end _H_GRB_MATRIX_BASE

