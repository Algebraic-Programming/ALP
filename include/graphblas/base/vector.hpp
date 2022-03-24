
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
 * @date 10th of August, 2016
 */

#ifndef _H_GRB_VECTOR_BASE
#define _H_GRB_VECTOR_BASE

#include <cstdlib>  //size_t
#include <iterator> //std::iterator
#include <stdexcept>
#include <utility> //pair

#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>

namespace grb {

	/**
	 * A GraphBLAS vector. This is an opaque data type that can be provided to
	 * any GraphBLAS function, such as, grb::eWiseMulAdd, for example.
	 *
	 * @tparam D The type of an element of this vector. \a D shall not be a
	 *           GraphBLAS type.
	 * @tparam implementation Allows different backends to implement different
	 *         versions of this data type.
	 * @tparam C The type of the class that keeps track of sparsity structure.
	 *
	 * \warning Creating a grb::Vector of other GraphBLAS types is
	 *                <em>not allowed</em>.
	 *          Passing a GraphBLAS type as template parameter will lead to
	 *          undefined behaviour.
	 *
	 * \note The implementation found in the same file as this documentation
	 *       catches invalid backends only. This class should never compile.
	 *
	 * @see grb::Vector< D, reference, C > for an actual implementation example.
	 */
	template< typename D, enum Backend implementation, typename C >
	class Vector {

		public :

			/** The type of elements stored in this vector. */
			typedef D value_type;

			/**
			 * Defines a reference to a value of type \a D. This reference is only valid
			 * when used inside a lambda function that is passed to grb::eWiseLambda().
			 *
			 * \warning Any other use of this reference incurs undefined behaviour.
			 *
			 * \par Example.
			 * An example valid use:
			 * \code
			 * void f(
			 *      Vector< D >::lambda_reference x,
			 *      const Vector< D >::lambda_reference y,
			 *      const Vector< D > &v
			 * ) {
			 *      grb::eWiseLambda( [x,y](const size_t i) {
			 *          x += y;
			 *      }, v );
			 * }
			 * \endcode
			 * This code adds \a y to \a x for every element in \a v. For a more useful
			 * example, see grb::eWiseLambda.
			 *
			 * \warning Note that, unlike the above, this below code is illegal since it
			 *          does not evaluate via a lambda passed to any of the above
			 *          GraphBLAS lambda functions (such as grb::eWiseLambda).
			 *          \code{.cpp}
			 *              void f(
			 *                   Vector< D >::lambda_reference x,
			 *                   const Vector< D >::lambda_reference y
			 *              ) {
			 *                   x += y;
			 *              }
			 *          \endcode
			 *          Also this usage is illegal since it does not rely on any
			 *          GraphBLAS-approved function listed above:
			 *          \code{.cpp}
			 *              void f(
			 *                   Vector< D >::lambda_reference x,
			 *                   const Vector< D >::lambda_reference y
			 *              ) {
			 *                   std::functional< void() > f =
			 *                       [x,y](const size_t i) {
			 *                           x += y;
			 *                       };
			 *                   f();
			 *              }
			 *          \endcode
			 *
			 * \warning There is no similar concept in the official GraphBLAS specs.
			 *
			 * @see grb::Vector::operator[]()
			 * @see grb::eWiseLambda
			 */
			typedef D& lambda_reference;

			/**
			 * A standard iterator for the Vector< D > class.
			 *
			 * This iterator is used for data extraction only. Hence only this const
			 * version is supplied.
			 *
			 * \warning Comparing two const iterators corresponding to different
			 *          containers leads to undefined behaviour.
			 * \warning Advancing an iterator past the end iterator of the container
			 *          it corresponds to leads to undefined behaviour.
			 * \warning Modifying the contents of a container makes any use of any
			 *          iterator derived from it incur invalid behaviour.
			 * \note    These are standard limitations of STL iterators.
			 */
			class const_iterator :
				public std::iterator<
					std::forward_iterator_tag,
					std::pair< const size_t, const D >,
					size_t
				>
			{

				public :

					/** Standard equals operator. */
					bool operator==( const const_iterator & other ) const {
						(void)other;
						return false;
					}

					/** @returns The negation of operator==(). */
					bool operator!=( const const_iterator & other ) const {
						(void)other;
						return true;
					}

					/**
					 * Dereferences the current position of this iterator.
					 *
					 * @return If this iterator is valid and not in end position,
					 *         this returns a new std::pair with in its first
					 *         field the position of the nonzero value, and in its
					 *         second field the value of the nonzero.
					 *
					 * \note If this iterator is invalid or in end position, the result is,
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

			/**
			 * Creates an ALP/GraphBLAS vector. The given dimension will be fixed
			 * throughout the lifetime of this container. After instantiation, the
			 * vector will contain no nonzeroes.
			 *
			 * @param[in] n  The dimension of this vector.
			 * @param[in] nz The minimal initial capacity of this vector.
			 *
			 * The argument \a nz is \em optional. Its default value is \a n.
			 *
			 * \parblock
			 * \par Performance semantics
			 * A backend must:
			 *    -# define cost in terms of work,
			 *    -# define intra-process data movement costs,
			 *    -# define inter-process data movement costs,
			 *    -# define whether inter-process synchronisations occur,
			 *    -# define memory storage requirements and may define
			 *       this in terms of \a n and/or \a nz, and
			 *    -# must define whether system calls may be made, and in particular
			 *       whether allocation or freeing of dynamic memory occurs or may
			 *       occur.
			 * \endparblock
			 *
			 * \warning Most backends will require work, intra-process data movement, and
			 *          system calls for the dynamic allocation of memory areas, all of
			 *          (at least the complexity of) \f$ \Omega( \mathit{nz} ) \f$. Hence
			 *          avoid the use of this constructor within performance-critical
			 *          code sections.
			 */
			Vector( const size_t n, const size_t nz ) {
				(void)n;
				(void)nz;
			}

			/**
			 * Creates an ALP/GraphBLAS vector. This constructor is specified as per the
			 * above where \a nz is to taken equal to \a n.
			 */
			Vector( const size_t n ) {
				(void)n;
			}

			/**
			 * Move constructor.
			 *
			 * This will make the new vector equal the given GraphBLAS vector while
			 * destroying the supplied GraphBLAS vector.
			 *
			 * This function always succeeds and will not throw exceptions.
			 *
			 * @param[in] x The GraphBLAS vector to move to this new container.
			 *
			 * \parblock
			 * \par Performance semantics
			 *        -# This constructor completes in \f$ \Theta(1) \f$ time.
			 *        -# This constructor does not allocate new data on the heap.
			 *        -# This constructor uses \f$ \mathcal{O}(1) \f$ more memory than
			 *           already used by this application at constructor entry.
			 *        -# This constructor incurs at most \f$ \mathcal{O}(1) \f$ bytes of
			 *           data movement.
			 * \endparblock
			 */
			Vector( Vector< D, implementation, C > &&x ) noexcept {
				(void)x;
			}

			/**
			 * Move-from-temporary assignment.
			 *
			 * @param[in,out] x The temporary instance from which this instance shall
			 *                  take over its resources.
			 *
			 * After a call to this function, \a x shall correspond to an empy vector.
			 *
			 * \parblock
			 * \par Performance semantics
			 *         -# This move assignment completes in \f$ \Theta(1) \f$ time.
			 *         -# This move assignment may not make system calls.
			 *         -# this move assignment moves \f$ \Theta(1) \f$ data only.
			 * \endparblock
			 */
			Vector< D, implementation, C >& operator=( Vector< D, implementation, C > &&x ) noexcept {
				(void)x;
				return *this;
			}

			/**
			 * Default destructor. Frees all associated memory areas.
			 *
			 * \parblock
			 * \par Performance semantics
			 *        -# This destructor contains \f$ \mathcal{O}(n) \f$ work, where
			 *           \f$ n \f$ is the capacity of this vector.
			 *        -# This destructor is only allowed to free memory, not allocate.
			 *        -# This destructor uses \f$ \mathcal{O}(1) \f$ more memory than
			 *           already used by this application at entry.
			 *        -# This destructor shall move at most \f$ \mathcal{O}(n) \f$ bytes
			 *           of data.
			 *        -# This destructor will make system calls.
			 * \endparblock
			 *
			 * \warning Avoid the use of this destructor within performance critical
			 *          code sections.
			 *
			 * \note Destruction of this GraphBLAS container is the only way to
			 *       guarantee that any underlying dynamically allocated memory is
			 *       freed.
			 */
			~Vector() {}

			//@{
			/**
			 * Provides the only mechanism to extract data from this GraphBLAS vector.
			 *
			 * The order in which nonzero elements are returned is undefined.
			 *
			 * @return An iterator pointing to the first element of this vector, if any;
			 *         \em or an iterator in end position if this vector contains no
			 *         nonzeroes.
			 *
			 * \note An `iterator in end position' compares equal to the const_iterator
			 *       returned by cend().
			 *
			 * \parblock
			 * \par Performance semantics
			 *        -# This function contains \f$ \mathcal{O}(1) \f$ work.
			 *        -# This function is allowed allocate dynamic memory.
			 *        -# This function uses up to \f$ \mathcal{O}(1) \f$ more memory
			 *           than already used by this application at entry.
			 *        -# This function shall move at most \f$ \mathcal{O}(1) \f$ bytes
			 *           of data.
			 *        -# This function may make system calls.
			 * \endparblock
			 *
			 * \warning Avoid the use of this function within performance critical code
			 *          sections.
			 *
			 * \note This function may make use of a const_iterator that is buffered,
			 *       hence possibly causing its implicitly called constructor to
			 *       allocate dynamic memory.
			 */
			const_iterator cbegin() const {}

			/**
			 * Same as cbegin().
			 * Since iterators are only supplied as a data extraction mechanism, there
			 * is no overloaded version of this function that returns a non-const
			 * iterator.
			 */
			const_iterator begin() const {}
			//@}

			//@{
			/**
			 * Indicates the end to the elements in this container.
			 *
			 * @return An iterator at the end position of this container.
			 *
			 * \parblock
			 * \par Performance semantics
			 *        -# This function contains \f$ \mathcal{O}(1) \f$ work.
			 *        -# This function is not allowed allocate dynamic memory.
			 *        -# This function uses up to \f$ \mathcal{O}(1) \f$ more memory
			 *           than already used by this application at entry.
			 *        -# This function shall move at most \f$ \mathcal{O}(1) \f$ bytes
			 *           of data.
			 *        -# This function shall \em not induce any system calls.
			 * \endparblock
			 *
			 * \note Even if cbegin() returns a buffered const_iterator that may require
			 *       dynamic memory allocation and additional data movement, this
			 *       specification disallows the same to happen for the construction of
			 *       an iterator in end position.
			 */
			const_iterator cend() const {}

			/**
			 * Same as cend().
			 * Since iterators are only supplied as a data extraction mechanism, there
			 * is no overloaded version of this function that returns a non-const
			 * iterator.
			 */
			const_iterator end() const {}
			//@}

			/**
			 * Copy from raw user-supplied data into a vector.
			 *
			 * This is the dense unmasked variant.
			 *
			 * @tparam descr        The pre-processing descriptor to use.
			 * @tparam fwd_iterator The type of input iterator. By default, this will be
			 *                      a raw \em unaliased pointer.
			 * @tparam Accum        The accumulator type used to merge incoming new
			 *                      elements with existing contents, if any.
			 *
			 * @param[in] accum The accumulator used to merge incoming new elements with
			 *                  existing content, if any.
			 * @param[in] start The iterator to the first element that should be copied
			 *                  into this GraphBLAS vector.
			 * @param[in] end   Iterator shifted exactly one past the last element that
			 *                  should be copied into this GraphBLAS vector.
			 * @param[out] npos The last iterator position after exiting this function.
			 *                  In most cases this will equal \a end. This parameter is
			 *                  optional.
			 *
			 * The first element from \a it will be copied into the element with index
			 * \f$ 0 \f$ in this vector. The \f$ k \f$-th element will be copied into
			 * the element with index \f$ k - 1 \f$. The iterator \a start will be
			 * incremented along with \f$ k \f$ until it compares equal to \a end, or
			 * until it has been incremented \a n times, where \a n is the dimension of
			 * this vector. In the latter case, any remaining values are ignored.
			 *
			 * @return grb::SUCCESS This function always succeeds.
			 *
			 * \note The default accumulator expects \a val to be of the same type
			 *       as nonzero elements in this function, and will cause old
			 *       values to be overwritten by the incoming new values.
			 *
			 * \note Previous contents of the vector are retained. If these are to be
			 *       cleared first, see clear(). The default accumulator is NOT an
			 *       alternative since any pre-existing values corresponding to entries
			 *       in the mask that evaluate to false will be retained.
			 *
			 * \note The parameter \a n can be used to ingest only a subset of a larger
			 *       data structure pointed to by \a start. At the end of the call, \a
			 *       start will then not be equal to \a end, but instead point to the
			 *       first element of the remainder of the larger data structure.
			 *
			 * \par Valid descriptors
			 * grb::descriptors::no_operation, grb::descriptors::no_casting.
			 *
			 * \note Invalid descriptors will be ignored.
			 *
			 * If grb::descriptors::no_casting is specified, then 1) the first domain of
			 * \a accum must match the type of \a val, 2) the second domain must match
			 * the type \a D of nonzeroes in this vector, and 3) the third domain must
			 * match \a D. If one of these is not true, the code shall not compile.
			 *
			 * \parblock
			 * \par Performance semantics
			 *      If the capacity of this container is sufficient to perform the
			 *      requested operation, then:
			 *        -# This function contains \f$ \Theta(n) \f$ work.
			 *        -# This function will take at most \f$ \Theta(1) \f$ memory beyond
			 *           the memory already used by the application before the call to
			 *           this function.
			 *        -# This function moves at most \f$ n ( 2\mathit{sizeof}(D) +
			 *           \mathit{sizeof}(\mathit{bool}) ) + \mathcal{O}(1) \f$ bytes of
			 *           data.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance exceptions
			 *      If the capacity of this container at function entry is insufficient
			 *      to perform the requested operation, then, in addition to the above:
			 *        -# this function allocates \f$ \Theta(n) \f$ bytes of memory .
			 *        -# this function frees \f$ \mathcal{O}(n) \f$ bytes of memory.
			 *        -# this function will make system calls.
			 * \endparblock
			 *
			 * \note An implementation may ensure that at object construction the
			 *       capacity is maximised. In that case, the above performance
			 *       exceptions will never come to pass.
			 *
			 * @see grb::buildVector for the GraphBLAS standard dispatcher to this
			 *                       function.
			 */
			template< Descriptor descr = descriptors::no_operation, class Accum = typename operators::right_assign< D, D, D >, typename fwd_iterator = const D * __restrict__ >
			RC build( const Accum & accum, const fwd_iterator start, const fwd_iterator end, fwd_iterator npos ) {
				(void)accum;
				(void)start;
				(void)end;
				(void)npos;
				return PANIC;
			}

			/**
			 * Copy from raw user-supplied data into a vector.
			 *
			 * This is the sparse non-masked variant.
			 *
			 * @tparam descr        The pre-processing descriptor to use.
			 * @tparam Accum        The type of the operator used to combine newly input
			 *                      data with existing data, if any.
			 * @tparam ind_iterator The type of index input iterator. By default, this
			 *                      will be a raw \em unaliased pointer to elements of
			 *                      type \a size_t.
			 * @tparam nnz_iterator The type of nonzero input iterator. By default, this
			 *                      will be a raw \em unaliased pointer to elements of
			 *                      type \a D.
			 * @tparam Dup          The type of operator used to combine any duplicate
			 *                      input values.
			 *
			 * @param[in] accum     The operator to be used when writing back the result
			 *                      of data that was already in this container prior to
			 *                      calling this function.
			 * @param[in] ind_start The iterator to the first index value that should be
			 *                      added to this GraphBLAS vector.
			 * @param[in] ind_end   Iterator corresponding to the end position of
			 *                      \a ind_start.
			 * @param[in] nnz_start The iterator to the first nonzero value that should
			 *                      be added to this GraphBLAS vector.
			 * @param[in] nnz_end   Iterator corresponding to the end position of
			 *                      \a nnz_start.
			 * @param[in] dup       The operator to be used when handling multiple
			 *                      nonzero values that are to be mapped to the same
			 *                      index position.
			 *
			 * The first element from \a nnz_start will be copied into this vector at
			 * the index corresponding to the first element from \a ind_start. Then,
			 * both nonzero and index value iterators advance to add the next input
			 * element and the process repeats until either of the input iterators
			 * reach \a nnz_end or \a ind_end, respectively.
			 * If at that point one of the iterators still has remaining elements, then
			 * those elements are ignored.
			 *
			 * @return grb::MISMATCH When attempting to insert a nonzero value at an
			 *                       index position that is larger or equal to the
			 *                       dimension of this vector. When this code is
			 *                       returned, the contents of this container are
			 *                       undefined.
			 * @return grb::SUCCESS  When all elements are successfully assigned.
			 *
			 * \note The default accumulator expects \a D to be of the same type
			 *       as nonzero elements of this operator, and will cause old
			 *       values to be overwritten by the incoming new values.
			 *
			 * \note The default \a dup expects \a D to be of the same type as nonzero
			 *       elements of this operator, and will cause duplicate values to be
			 *       discarded in favour of the last seen value.
			 *
			 * \note Previous contents of the vector are retained. If these are to be
			 *       cleared first, see clear(). The default accumulator is NOT an
			 *       alternative since any pre-existing values corresponding to entries
			 *       in the mask that evaluate to false will be retained.
			 *
			 * \par Valid descriptors
			 * grb::descriptors::no_operation, grb::descriptors::no_casting,
			 * grb::descriptors::no_duplicates.
			 *
			 * \note Invalid descriptors will be ignored.
			 *
			 * If grb::descriptors::no_casting is specified, then 1) the first domain of
			 * \a accum must match the type of \a D, 2) the second domain must match
			 * nnz_iterator::value_type, and 3) the third domain must \a D. If one of
			 * these is not true, the code shall not compile.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This function contains \f$ \Theta(n) \f$ work.
			 *        -# This function will take at most \f$ \Theta(1) \f$ memory beyond
			 *           the memory already used by the application before the call to
			 *           this function.
			 *        -# This function moves at most \f$ n ( 2\mathit{sizeof}(D) +
			 *           \mathit{sizeof}(\mathit{bool}) ) + \mathcal{O}(1) \f$ bytes of
			 *           data.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance exceptions
			 *      If the capacity of this container at function entry is insufficient
			 *      to perform the requested operation, then, in addition to the above:
			 *        -# this function allocates \f$ \Theta(n) \f$ bytes of memory .
			 *        -# this function frees \f$ \mathcal{O}(n) \f$ bytes of memory.
			 *        -# this function will make system calls.
			 * \endparblock
			 *
			 * \note An implementation may ensure that at object construction the
			 *       capacity is maximised. In that case, the above performance
			 *       exceptions will never come to pass.
			 *
			 * @see grb::buildVector for the GraphBLAS standard dispatcher to this
			 *                       function.
			 */
			template< Descriptor descr = descriptors::no_operation,
				class Accum = operators::right_assign< D, D, D >,
				typename ind_iterator = const size_t * __restrict__,
				typename nnz_iterator = const D * __restrict__,
				class Dup = operators::right_assign< D, D, D > >
			RC build( const Accum & accum, const ind_iterator ind_start, const ind_iterator ind_end, const nnz_iterator nnz_start, const nnz_iterator nnz_end, const Dup & dup = Dup() ) {
				(void)accum;
				(void)ind_start;
				(void)ind_end;
				(void)nnz_start;
				(void)nnz_end;
				(void)dup;
				return PANIC;
			}

			/**
			 * Copy from raw user-supplied data into a vector.
			 *
			 * This is the sparse masked variant.
			 *
			 * @tparam descr        The pre-processing descriptor to use.
			 * @tparam mask_type    The value type of the \a mask vector. This type is
			 *                      \em not required to be \a bool.
			 * @tparam Accum        The type of the operator used to combine newly input
			 *                      data with existing data, if any.
			 * @tparam ind_iterator The type of index input iterator. By default, this
			 *                      will be a raw \em unaliased pointer to elements of
			 *                      type \a size_t.
			 * @tparam nnz_iterator The type of nonzero input iterator. By default, this
			 *                      will be a raw \em unaliased pointer to elements of
			 *                      type \a D.
			 * @tparam Dup          The type of operator used to combine any duplicate
			 *                      input values.
			 *
			 * @param[in] mask      An element is only added to this container if its
			 *                      index \f$ i \f$ has a nonzero at the same position
			 *                      in \a mask that evaluates true.
			 * @param[in] accum     The operator to be used when writing back the result
			 *                      of data that was already in this container prior to
			 *                      calling this function.
			 * @param[in] ind_start The iterator to the first index value that should be
			 *                      added to this GraphBLAS vector.
			 * @param[in] ind_end   Iterator corresponding to the end position of
			 *                      \a ind_start.
			 * @param[in] nnz_start The iterator to the first nonzero value that should
			 *                      be added to this GraphBLAS vector.
			 * @param[in] nnz_end   Iterator corresponding to the end position of
			 *                      \a nnz_start.
			 * @param[in] dup       The operator to be used when handling multiple
			 *                      nonzero values that are to be mapped to the same
			 *                      index position.
			 *
			 * The first element from \a nnz_start will be copied into this vector at
			 * the index corresponding to the first element from \a ind_start. Then,
			 * both nonzero and index value iterators advance to add the next input
			 * element and the process repeats until either of the input iterators
			 * reach \a nnz_end or \a ind_end, respectively.
			 * If at that point one of the iterators still has remaining elements, then
			 * those elements are ignored.
			 *
			 * @return grb::MISMATCH When attempting to insert a nonzero value at an
			 *                       index position that is larger or equal to the
			 *                       dimension of this vector. When this code is
			 *                       returned, the contents of this container are
			 *                       undefined.
			 * @return grb::SUCCESS  When all elements are successfully assigned.
			 *
			 * \note The default accumulator expects \a D to be of the same type
			 *       as nonzero elements of this operator, and will cause old
			 *       values to be overwritten by the incoming new values.
			 *
			 * \note The default \a dup expects \a D to be of the same type as nonzero
			 *       elements of this operator, and will cause duplicate values to be
			 *       discarded in favour of the last seen value.
			 *
			 * \note Previous contents of the vector are retained. If these are to be
			 *       cleared first, see clear(). The default accumulator is NOT an
			 *       alternative since any pre-existing values corresponding to entries
			 *       in the mask that evaluate to false will be retained.
			 *
			 * \par Valid descriptors
			 * grb::descriptors::no_operation, grb::descriptors::no_casting,
			 * grb::descriptors::invert_mask, grb::descriptors::no_duplicates.
			 *
			 * \note Invalid descriptors will be ignored.
			 *
			 * If grb::descriptors::no_casting is specified, then 1) the first domain of
			 * \a accum must match the type of \a D, 2) the second domain must match
			 * nnz_iterator::value_type, and 3) the third domain must \a D. If one of
			 * these is not true, the code shall not compile.
			 *
			 * \parblock
			 * \par Performance semantics.
			 *        -# This function contains \f$ \Theta(n) \f$ work.
			 *        -# This function will take at most \f$ \Theta(1) \f$ memory beyond
			 *           the memory already used by the application before the call to
			 *           this function.
			 *        -# This function moves at most \f$ n ( 2\mathit{sizeof}(D) +
			 *           \mathit{sizeof}(\mathit{bool}) ) + \mathcal{O}(1) \f$ bytes of
			 *           data.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance exceptions
			 *      If the capacity of this container at function entry is insufficient
			 *      to perform the requested operation, then, in addition to the above:
			 *        -# this function allocates \f$ \Theta(n) \f$ bytes of memory .
			 *        -# this function frees \f$ \mathcal{O}(n) \f$ bytes of memory.
			 *        -# this function will make system calls.
			 * \endparblock
			 *
			 * \note An implementation may ensure that at object construction the
			 *       capacity is maximised. In that case, the above performance
			 *       exceptions will never come to pass.
			 *
			 * @see grb::buildVector for the GraphBLAS standard dispatcher to this
			 *                       function.
			 */
			template< Descriptor descr = descriptors::no_operation,
				typename mask_type,
				class Accum,
				typename ind_iterator = const size_t * __restrict__,
				typename nnz_iterator = const D * __restrict__,
				class Dup = operators::right_assign< D, typename nnz_iterator::value_type, D > >
			RC build( const Vector< mask_type, implementation, C > mask,
				const Accum & accum,
				const ind_iterator ind_start,
				const ind_iterator ind_end,
				const nnz_iterator nnz_start,
				const nnz_iterator nnz_end,
				const Dup & dup = Dup() ) {
				(void)mask;
				(void)accum;
				(void)ind_start;
				(void)ind_end;
				(void)nnz_start;
				(void)nnz_end;
				(void)dup;
				return PANIC;
			}

			/**
			 * Return the dimension of this vector.
			 *
			 * @tparam T The integral output type.
			 *
			 * @param[out] size Where to store the size of this vector.
			 *                  The initial value is ignored.
			 *
			 * @returns grb::SUCCESS When the function call completes successfully.
			 *
			 * \note This function cannot fail.
			 *
			 * \parblock
			 * \par Performance semantics
			 *      This function
			 *        -# contains \f$ \Theta(1) \f$ work,
			 *        -# will not allocate new dynamic memory,
			 *        -# will take at most \f$ \Theta(1) \f$ memory beyond the memory
			 *           already used by the application before the call to this
			 *           function.
			 *        -# will move at most \f$ \mathit{sizeof}(T) +
			 *          \mathit{sizeof}(\mathit{size\_t}) \f$ bytes of data.
			 * \endparblock
			 */
			template< typename T >
			RC size( T & size ) const {
				(void)size;
				return PANIC;
			}

			/**
			 * Return the number of nonzeroes in this vector.
			 *
			 * @tparam T The integral output type.
			 *
			 * @param[out] nnz Where to store the number of nonzeroes contained in this
			 *                 vector. Its initial value is ignored.
			 *
			 * @returns grb::SUCCESS When the function call completes successfully.
			 *
			 * \note This function cannot fail.
			 *
			 * \parblock
			 * \par Performance semantics
			 *      This function
			 *        -# contains \f$ \Theta(1) \f$ work,
			 *        -# will not allocate new dynamic memory,
			 *        -# will take at most \f$ \Theta(1) \f$ memory beyond the memory
			 *           already used by the application before the call to this
			 *           function.
			 *        -# will move at most \f$ \mathit{sizeof}(T) +
			 *           \mathit{sizeof}(\mathit{size\_t}) \f$ bytes of data.
			 * \endparblock
			 */
			template< typename T >
			RC nnz( T & nnz ) const {
				(void)nnz;
				return PANIC;
			}

			/**
			 * Returns a lambda reference to an element of this sparse vector.
			 *
			 * A lambda reference to an element of this vector is only valid when used
			 * inside a lambda function evaluated via grb::eWiseLambda. The lambda
			 * function is called for specific indices only-- that is, the GraphBLAS
			 * implementation decides at which elements to dereference this container.
			 * Outside this scope the returned reference incurs undefined behaviour.
			 *
			 * \warning In particular, for the given index \a i by the lambda function,
			 *          it shall be \em illegal to refer to indices relative to that
			 *          \a i; including, but not limited to, \f$ i+1 \f$, \f$ i-1 \f$, et
			 *          cetera.
			 *
			 * \note    As a consequence, this function cannot be used to perform stencil
			 *          or halo based operations.
			 *
			 * If a previously non-existing entry of the vector is requested, a new
			 * nonzero is added at position \a i in this vector. The new element will
			 * have its initial value equal to the \em identity corresponding to the
			 * given monoid.
			 *
			 * \warning In parallel contexts the use of a returned lambda reference
			 *          outside the context of an eWiseLambda will incur at least one of
			 *          the following ill effects: it may
			 *            -# fail outright,
			 *            -# work on stale data,
			 *            -# work on incorrect data, or
			 *            -# incur high communication costs to guarantee correctness.
			 *          In short, such usage causes undefined behaviour. Implementers are
			 *          \em not advised to provide GAS-like functionality through this
			 *          interface, as it invites bad programming practices and bad
			 *          algorithm design decisions. This operator is instead intended to
			 *          provide for generic BLAS1-type operations only.
			 *
			 * \note    For I/O, use the iterator retrieved via cbegin() instead of
			 *          relying on a lambda_reference.
			 *
			 * @param[in] i      Which element to return a lambda reference of.
			 * @param[in] monoid Under which generalised monoid to interpret the
			 *                   requested \f$ i \f$th element of this vector.
			 *
			 * \note The \a monoid (or a ring) is required to be able to interpret a
			 *       sparse vector. A user who is sure this vector is dense, or otherwise
			 *       is able to ensure that the a lambda_reference will only be requested
			 *       at elements where nonzeroes already exists, may refer to
			 *       Vector::operator[],
			 *
			 * @return A lambda reference to the element \a i of this vector.
			 *
			 * \par Example.
			 * See grb::eWiseLambda() for a practical and useful example.
			 *
			 * \warning There is no similar concept in the official GraphBLAS specs.
			 *
			 * @see lambda_reference For more details on the returned reference type.
			 * @see grb::eWiseLambda For one legal way in which to use the returned
			 *      #lambda_reference.
			 */
			template< class Monoid >
			lambda_reference operator()( const size_t i, const Monoid & monoid = Monoid() ) {
				(void)i;
				(void)monoid;
				return PANIC;
			}

			/**
			 * Returns a lambda reference to an element of this vector. The user
			 * ensures that the requested reference only corresponds to a pre-existing
			 * nonzero in this vector, <em>or undefined behaviour will occur</em>.
			 *
			 * A lambda reference to an element of this vector is only valid when used
			 * inside a lambda function evaluated via grb::eWiseLambda. The lambda
			 * function is called for specific indices only-- that is, the GraphBLAS
			 * implementation decides at which elements to dereference this container.
			 * Outside this scope the returned reference incurs undefined behaviour.
			 *
			 * \warning In particular, for the given index \a i by the lambda function,
			 *          it shall be \em illegal to refer to indices relative to that
			 *          \a i; including, but not limited to, \f$ i+1 \f$, \f$ i-1 \f$, et
			 *          cetera.
			 *
			 * \note    As a consequence, this function cannot be used to perform stencil
			 *          or halo based operations.
			 *
			 * If a previously non-existing entry of the vector is requested, undefined
			 * behaviour will occur. Functions that are defined to work with references
			 * of this kind, such as grb::eWiseLambda, define exactly which elements are
			 * dereferenced.
			 *
			 * \warning In parallel contexts the use of a returned lambda reference
			 *          outside the context of an eWiseLambda will incur at least one of
			 *          the following ill effects: it may
			 *            -# fail outright,
			 *            -# work on stale data,
			 *            -# work on incorrect data, or
			 *            -# incur high communication costs to guarantee correctness.
			 *          In short, such usage causes undefined behaviour. Implementers are
			 *          \em not advised to provide GAS-like functionality through this
			 *          interface, as it invites bad programming practices and bad
			 *          algorithm design decisions. This operator is instead intended to
			 *          provide for generic BLAS1-type operations only.
			 *
			 * \note    For I/O, use the iterator retrieved via cbegin() instead of
			 *          relying on a lambda_reference.
			 *
			 * @param[in] i    Which element to return a lambda reference of.
			 * @param[in] ring Under which generalised semiring to interpret the
			 *                 requested \f$ i \f$th element of this vector.
			 *
			 * \note The \a ring is required to be able to interpret a sparse vector. A
			 *       user who is sure this vector is dense, or otherwise is able to
			 *       ensure that the a lambda_reference will only be requested at
			 *       elements where nonzeroes already exists, may refer to
			 *       Vector::operator[],
			 *
			 * @return A lambda reference to the element \a i of this vector.
			 *
			 * \par Example.
			 * See grb::eWiseLambda() for a practical and useful example.
			 *
			 * \warning There is no similar concept in the official GraphBLAS specs.
			 *
			 * @see lambda_reference For more details on the returned reference type.
			 * @see grb::eWiseLambda For one legal way in which to use the returned
			 *      #lambda_reference.
			 */
			lambda_reference operator[]( const size_t i ) {
				(void)i;
#ifndef _GRB_NO_EXCEPTIONS
				assert( false ); // Requesting lambda reference of unimplemented Vector backend.
#endif
			}
};

	template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
	class VectorView { };

}

#endif // _H_GRB_VECTOR_BASE
