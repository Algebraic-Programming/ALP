
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
 * @date 10th of August
 */

#ifndef _H_GRB_MATRIX_BASE
#define _H_GRB_MATRIX_BASE

#include <iterator>
#include <stddef.h>
#include <type_traits>
#include <utility>
#include <memory>

#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/storage.hpp>
#include <graphblas/structures.hpp>
#include <graphblas/utils.hpp>
#include <graphblas/views.hpp>
#include <graphblas/imf.hpp>

namespace grb {

	/**
	 * A GraphBLAS matrix.
	 *
	 * This is an opaque data type that implements the below functions.
	 *
	 * @tparam D  The type of a nonzero element. \a D shall not be a GraphBLAS
	 *            type.
	 * @tparam implementation Allows multiple backends to implement different
	 *         versions of this data type.
	 *
	 * \warning Creating a grb::Matrix of other GraphBLAS types is
	 *                <em>not allowed</em>.
	 *          Passing a GraphBLAS type as template parameter will lead to
	 *          undefined behaviour.
	 */
	template< typename D, enum Backend implementation >
	class Matrix {

		typedef Matrix< D, implementation > self_type;

		public :

			/**
		     * A standard iterator for a GraphBLAS aatrix.
		     *
		     * This iterator is used for data extraction only. Hence only this const
		     * version is specified.
		     *
		     * Dereferencing an iterator of this type that is not in end position yields
		     * a pair \f$ (c,v) \f$. The value \a v is of type \a D and corresponds to
		     * the value of the dereferenced nonzero.
		     * The value \a c is another pair \f$ (i,j) \f$. The values \a i and \a j
		     * are of type <code>size_t</code> and correspond to the coordinate of the
		     * dereferenced nonzero.
		     *
		     * \note `Pair' here corresponds to the regular <code>std::pair</code>.
		     *
		     * \warning Comparing two const iterators corresponding to different
		     *          containers leads to undefined behaviour.
		     * \warning Advancing an iterator past the end iterator of the container
		     *          it corresponds to, leads to undefined behaviour.
		     * \warning Modifying the contents of a container makes any use of any
		     *          iterator derived from it incur invalid behaviour.
		     * \note    These are standard limitations of STL iterators.
		     */
			class const_iterator : public std::iterator< std::forward_iterator_tag, std::pair< std::pair< const size_t, const size_t >, const D >, size_t > {

				public :

					/** Standard equals operator. */
					bool
					operator==( const const_iterator & other ) const { (void)other; return false; }

	/** @returns The negation of operator==(). */
	bool operator!=( const const_iterator & other ) const {
		(void)other;
		return true;
	}

	/**
	 * Dereferences the current position of this iterator.
	 *
	 * @return If this iterator is valid and not in end position, this returns
	 *         an std::pair with in its first field the position of the
	 *         nonzero value, and in its second field the value of the nonzero.
	 *         The position of a nonzero is another std::pair with both the
	 *         first and second field of type <code>size_t</code>.
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

}; // namespace grb

/** The value type of elements stored in this matrix. */
typedef D value_type;

/**
 * The main GraphBLAS matrix constructor.
 *
 * Matrix nonzeroes will be uninitalised after successful construction.
 *
 * Requesting a matrix with zero \a rows or \a columns will yield an empty
 * matrix; i.e., it will be useless but will not result in an error.
 *
 * @param rows        The number of rows in the new matrix.
 * @param columns     The number of columns in the new matrix.
 *
 * @return SUCCESS This function never fails.
 *
 * \parblock
 * \par Performance semantics.
 *        -# This constructor completes in \f$ \Theta(1) \f$ time.
 *        -# This constructor will not allocate any new dynamic memory.
 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
 *           memory beyond that at constructor entry.
 *        -# This constructor incurs \f$ \Theta(1) \f$ data movement.
 *        -# This constructor \em may make system calls.
 * \endparblock
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
 * @param other The matrix to copy.
 *
 * \parblock
 * \par Performance semantics.
 *      Allocates the same capacity as the \a other matrix, even if the
 *      actual number of nonzeroes contained in \a other is less.
 *        -# This constructor entails \f$ \Theta(\mathit{nz}) \f$ amount of
 *           work.
 *        -# This constructor allocates \f$ \Theta(\mathit{nz}) \f$ bytes
 *           of dynamic memory.
 *        -# This constructor incurs \f$ \Theta(\mathit{nz}) \f$ of data
 *           movement.
 *        -# This constructor \em may make system calls.
 * \endparblock
 *
 * \warning Avoid the use of this constructor within performance critical
 *          code sections.
 */
Matrix( const Matrix< D, implementation > & other ) {
	(void)other;
}

/**
 * Move constructor. This will make the new matrix equal to the given
 * GraphBLAS matrix while destroying the given matrix.
 *
 * @param[in] other The GraphBLAS matrix to move to this new instance.
 *
 * \parblock
 * \par Performance semantics.
 *        -# This constructor entails \f$ \Theta(1) \f$ amount of work.
 *        -# This constructor will not allocate any new dynamic memory.
 *        -# This constructor will use \f$ \Theta(1) \f$ extra bytes of
 *           memory beyond that at constructor entry.
 *        -# This constructor will move \f$ \Theta(1) \f$ bytes of data.
 * \endparblock
 */
Matrix( self_type && other ) {
	(void)other;
}

/**
 * Matrix destructor.
 *
 * \parblock
 * \par Performance semantics.
 *        -# This destructor entails \f$ \Theta(1) \f$ amount of work.
 *        -# This destructor will not perform any memory allocations.
 *        -# This destructor will use \f$ \mathcal{O}(1) \f$ extra bytes of
 *           memory beyond that at constructor entry.
 *        -# This destructor will move \f$ \Theta(1) \f$ bytes of data.
 *        -# This destructor makes system calls.
 * \endparblock
 *
 * \warning Avoid calling destructors from within performance critical
 *          code sections.
 */
~Matrix() {}

/**
 * Assigns nonzeroes to the matrix from a coordinate format.
 *
 * Any prior content may be combined with new input according to the
 * user-supplied accumulator operator (\a accum).
 *
 * Input triplets need not be unique. Input triplets that are written to the
 * same row and column coordinates will be combined using the supplied
 * duplicate operator (\a dup).
 *
 * \note Note that \a dup and \a accum may differ. The duplicate operator is
 *       \em not applied to any pre-existing nonzero values.
 *
 * \note The order of application of the operators is undefined.
 *
 * The number of nonzeroes, after reduction by duplicate removals and after
 * merger with the existing nonzero structure, must be equal or less than the
 * space reserved during the construction of this matrix. The nonzeroes will
 * not be stored in a fully sorted fashion-- it will be sorted column-wise,
 * but within each column the order can be arbitrary.
 *
 * @tparam accum         How existing entries of this matrix should be
 *                       treated.
 *                       The default is #grb::operators::right_assign, which
 *                       means that any existing values are overwritten with
 *                       the new values.
 * @tparam dup           How to handle duplicate entries. The default is
 *                       #grb::operators::add, which means that duplicated
 *                       values are combined by addition.
 * @tparam descr         The descriptor used. The default is
 *                       #grb::descriptors::no_operation, which means that
 *                       no pre- or post-processing of input or input is
 *                       performed.
 * @tparam fwd_iterator1 The type of the row index iterator.
 * @tparam fwd_iterator2 The type of the column index iterator.
 * @tparam fwd_iterator3 The type of the nonzero value iterator.
 * @tparam length_type   The type of the number of elements in each iterator.
 * @tparam T             The type of the supplied mask.
 *
 * \note By default, the iterator types are raw, unaliased, pointers.
 *
 * \warning This means that by default, input arrays are \em not
 *          allowed to overlap.
 *
 * Forward iterators will only be used to read from, never to assign to.
 *
 * \note It is therefore both legal and preferred to pass constant forward
 *       iterators, as opposed to mutable ones as \a I, \a J, and \a V.
 *
 * @param[in]  I   A forward iterator to \a cap row indices.
 * @param[in]  J   A forward iterator to \a cap column indices.
 * @param[in]  V   A forward iterator to \a cap nonzero values.
 * @param[in]  nz  The number of items pointed to by \a I, \a J, \em and
 *                 \a V.
 * @param[in] mask An input element at coordinate \f$ (i,j) \f$ will only be
 *                 added to this matrix if there exists a matching element
 *                 \f$ \mathit{mask}_{ij} \f$ in the given \a mask that
 *                 eveluates <tt>true</tt>. The matrix in \a mask must be
 *                 of the same dimension as this matrix.
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
 *           \f$ \Theta(\mathit{nz}\log\mathit{nz})+\mathcal{O}(m+n)) \f$
 *           amount of work.
 *        -# This function may dynamically allocate
 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of memory.
 *        -# A call to this function will use \f$ \mathcal{O}(m+n) \f$ bytes
 *           of memory beyond the memory in use at the function call entry.
 *        -# This function will copy each input forward iterator at most
 *           \em twice; the three input iterators \a I, \a J, and \a V thus
 *           may have exactly two copies each, meaning that all input may be
 *           traversed \em twice.
 *        -# Each of the at most six iterator copies will be incremented at
 *           most \f$ \mathit{nz} \f$ times.
 *        -# Each position of the each of the at most six iterator copies
 *           will be dereferenced exactly once.
 *        -# This function moves
 *           \f$ \Theta(\mathit{nz})+\mathcal{O}(m+n)) \f$ bytes of data.
 *        -# If the mask is nonempty, the performance costs of grb::eWiseMul
 *           on two matrix arguments must be added to the above costs.
 *        -# This function will likely make system calls.
 * \endparblock
 *
 * \warning This is an extremely expensive function. Use sparingly and only
 *          when absolutely necessary
 *
 * \note Streaming input can be implemented by supplying buffered
 *       iterators to this GraphBLAS implementation.
 */
template< Descriptor descr = descriptors::no_operation,
	template< typename, typename, typename > class accum = operators::right_assign,
	template< typename, typename, typename > class dup = operators::add,
	typename fwd_iterator1 = const size_t * __restrict__,
	typename fwd_iterator2 = const size_t * __restrict__,
	typename fwd_iterator3 = const D * __restrict__,
	typename length_type = size_t,
	typename T >
RC buildMatrix( const fwd_iterator1 I, const fwd_iterator2 J, const fwd_iterator3 V, const length_type nz, const Matrix< T, implementation > & mask ) {
	(void)I;
	(void)J;
	(void)V;
	(void)nz;
	(void)mask;
	return PANIC;
}

//@{
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
}
;

template< typename InputType, Backend backend >
RC clear( Matrix< InputType, backend > & A ) noexcept {
	// this is the generic stub implementation
	return UNSUPPORTED;
}

/**
 * \brief An ALP structured matrix.
 *
 * This is an opaque data type for structured matrices. 
 * This container allows to maintain the interface of grb::Matrix and grb::Vector 
 * unaltered enabling back-compatibility while building on them to create
 * semantically reacher algebraic objects.
 * A structured matrix is generalized over five parameters further described 
 * below: its data type, 
 * its structure, whether it is stored using a dense or sparse storage scheme, 
 * a static view and the backend for which it is implemented.
 * At a high level of abstraction a structured matrix exposes a mathematical 
 * \em logical layout which allows to express implementation-oblivious concepts 
 * (e.g., the transpose of a symmetric matrix).
 * At the lowest level, the logical layout maps to its physical counterpart via 
 * a particular choice of a storage scheme within those exposed by the chosen 
 * backend. grb::Matrix and grb::Vector are used as interfaces to the physical
 * layout.
 * To visualize this, you could think of a band matrix. Using either the 
 * \a storage::Dense:full or \a storage::Dense:band storage schemes would require
 * the use of a \a grb::Matrix container (see include/graphblas/storage.hpp for
 * more details about the two storage schemes). However, the interpration of its 
 * content would differ in the two cases being a function of both the Structure 
 * information and the storage scheme combined.
 * 
 * Views can be used to create logical \em perspectives on top of a container. 
 * For example, I could decide to refer to the transpose of a matrix or to see 
 * a for limited part of my program a square matrix as symmetric. 
 * If a view can be expressed as concept invariant of specific runtime features,
 * such views can be defined statically (for example I can always refer to the 
 * transpose or the diagonal of a matrix irrespective of features such as its 
 * size). Other may depend on features such as the size of a matrix
 * and can be expressed as linear transformations via operations such as \a mxm 
 * (e.g., gathering/scattering the rows/columns of a matrix or permuting them).
 * Structured matrices defined as views on other matrices do not instantiate a
 * new container but refer to the one used by their targets. (See the 
 * documentation of StructuredMatrix for both scenarios within the \em denseref 
 * backend folder).
 *
 * @tparam T The type of the matrix elements. \a T shall not be a GraphBLAS
 *            type.
 * @tparam Structure  One of the matrix structures in \a grb::structures.
 * @tparam StorageSchemeType Either \em enum \a storage::Dense or \em enum 
 * 	       \a storage::Sparse.
 * 		   A StructuredMatrix will be allowed to pick among the storage schemes 
 *         within their specified \a StorageSchemeType.
 * @tparam View  One of the matrix views in \a grb::view.
 * 		   All static views except for \a view::Identity (via 
 *         \a view::identity<void> cannot instantiate a new container and only 
 *         allow to refer to an existing \a StructuredMatrix.  
 *         The \a View parameter should not be used directly by the user but 
 *         can be set using specific member types appropriately 
 *         defined by each StructuredMatrix. (See examples of StructuredMatrix 
 *         definitions within \a include/graphblas/denseref/matrix.hpp and the
 *         \a dense_structured_matrix.cpp unit test).
 * @tparam backend Allows multiple backends to implement different versions 
 *         of this data type.
 *
 * \note The presence of different combination of structures and views could 
 *       produce many specialization with lots of logic replication. We might
 *       could use some degree of inheritence to limit this.
 */
template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
class StructuredMatrix {


	/** 
	 * Whether the container presently is initialized or not. 
	 * We differentiate the concept of empty matrix (matrix of size \f$0\times 0\f$)
	 * from the one of uninitialized (matrix of size \f$m\times n\f$ which was never set)
	 * and that of zero matrix (matrix with all zero elements).
	 * \note in sparse format a zero matrix result in an ampty data structure. Is this
	 * used to refer to uninitialized matrix in ALP/GraphBLAS?
	 **/
	bool initialized;

	/**
	 * The two following members define the \em logical layout of a structured matrix:
	 * Its structure and access relations. This is enabled only if the structured matrix
	 * does not define a View on another matrix.
	 */
	using structure = Structure;
	/**
	 * A pair of pointers to index mapping functions (see imf.hpp) that express the
	 * logical access to the structured matrix.
	 */
	std::shared_ptr<imf::IMF> imf_l, imf_r;

	/**
	 * When a structured matrix instanciate a \em container it defines a new \em physical
	 * (concrete?) layout. This is characterized by an ALP container (aka a \a Matrix) and a 
	 * storage scheme that defines a unique interpretation of its content.
	 * The combination of the logical and physical layout of a structured matrix enables to
	 * identify a precise mapping between an element in the structured matrix and a position
	 * wihtin one or more 1/2D-arrays that store it.
	 */
    Matrix< T, reference_dense > * _container;

	/**
	 * A container's storage scheme. \a storage_scheme is not exposed to the user as an option
	 * but can defined by ALP at different points in the execution depending on the \a backend choice.
	 * For example, if the container is associated to an I/O matrix, with a reference backend
	 * it might be set to reflect the storage scheme of the user data as specified at buildMatrix.
	 * If \a backend is set to \a mlir then the scheme could be fixed by the JIT compiler to effectively
	 * support its optimization strategy.
	 * At construction time and until the moment the scheme decision is made it may be set to
	 * an appropriate default choice, e.g. if \a StorageSchemeType is \a storage::Dense then
	 * \a storage::Dense::full could be used.
	 */
	StorageSchemeType storage_scheme;

	/**
	 * When a structured matrix defines a View over another matrix, it contains a pointer
	 * to the latter. Its type can be identified via the View parameter.
	 */
	using target_type = typename std::enable_if<! std::is_same<View, view::Identity<void> >::value, typename View::applied_to>::type;
	target_type * ref;

	public :

	StructuredMatrix( const size_t m, const size_t n );

	StructuredMatrix( const StructuredMatrix< T, Structure, StorageSchemeType, View, backend > & other );

	StructuredMatrix( StructuredMatrix< T, Structure, StorageSchemeType, View, backend > && other );

	~StructuredMatrix();

}; // class StructuredMatrix

/**
 * Check if type \a T is a StructuredMatrix.
 */
template< typename T >
struct is_structured_matrix : std::false_type {};
template< typename T, typename Structure, typename StorageSchemeType, typename View, enum Backend backend >
struct is_structured_matrix< StructuredMatrix< T, Structure, StorageSchemeType, View, backend > > : std::true_type {};

} // end namespace ``grb''

#endif // end _H_GRB_MATRIX_BASE
