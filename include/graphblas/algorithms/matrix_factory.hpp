
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @file matrix_factory.hpp
 *
 * Implements matrix factory methods.
 *
 * @author Benjamin Lozes
 * @date 7th of August, 2023
 */

#ifndef _H_GRB_MATRIX_FACTORY
#define _H_GRB_MATRIX_FACTORY

#include <random>
#include <iostream>
#include <algorithm>

#include <graphblas/utils/iterators/adapter.hpp>
#include <graphblas/utils/iterators/regular.hpp>

#include <graphblas.hpp>


namespace grb::algorithms {

	/**
	 * Factories for creating ALP/GraphBLAS matrices with standard patterns such as
	 * identity matrices.
	 *
	 * Just as with constructing any ALP container, the use of factory functions,
	 * when an error is encountered, results in C++ exceptions being thrown.
	 *
	 * The capacity of returned matrices is requested at the exact minimum that is
	 * required. As per the core ALP/GraphBLAS specification, this means that the
	 * returned matrices have a capacity of <em>at least</em> the minimum required;
	 * i.e., <code>assert( grb::nnz(R) <= grb::capacity(R) );</code> holds, with
	 * <code>R</code> the returned matrix.
	 *
	 * In the case of an ALP process with multiple user processes, calling any
	 * factory method is a \em collective call. ALP guarantees that if a call
	 * fails at one user process, the call also fails at all other user processes,
	 * with matching exceptions. The matrix factory is scalable in the number of
	 * threads as well as the number of user processes.
	 *
	 * The following matrix factory methods are supported:
	 *  -# #diag,
	 *  -# #empty,
	 *  -# #eye,
	 *  -# #identity,
	 *  -# #full,
	 *  -# #dense,
	 *  -# #ones, and
	 *  -# #zeros.
	 *
	 * All these methods are implemented on top of core ALP primitives.
	 *
	 * @tparam D       The type of a non-zero element of the returned matrices.
	 *
	 * When passing <tt>void</tt> to \a D, the returned matrices will be pattern
	 * matrices.
	 *
	 * @tparam mode    The I/O mode to be used when constructing matrices.
	 *                 Optional; default is #grb::PARALLEL.
	 *
	 * \warning At present, #diag for non-pattern matrices is only supported for
	 *          sequential I/O. Please append to GitHub issue #238 if you require
	 *          this functionality so that we may prioritise it higher.
	 *
	 * The remainder template parameters are automatically inferred and should
	 * only be overridden by an expert user:
	 *
	 * @tparam backend The selected backend.
	 * @tparam RIT     The type used for row indices.
	 * @tparam CIT     The type used for column indices.
	 * @tparam NIT     The type used for non-zero indices.
	 *
	 * \par Performance semantics
	 * \parblock
	 * While the implementation of all factory methods guarantee their construction
	 * within \f$ \Theta(1) \f$ work-space, the involved work and data movement are
	 * \f$ \mathcal{O}(n+m) \f$ instead, where \f$ n \f$ is the maximum matrix
	 * dimension and \f$ m \f$ the number of nonzeroes in the produced matrix.
	 *
	 * In case of a shared-memory parallel backend with \f$ T \f$ threads, the
	 * thread-local work and data movement become \f$ \mathcal{O}((n+m)/T+T) \f$.
	 * System-wide compute costs thus are proportional to
	 * \f$ \mathcal{O}(m+n)/T+T \f$ while system-wide data movement costs remain
	 * proportional to \f$ \mathcal{O}(m+n+T) \f$, with usually \f$ m+n \gg T \f$.
	 * The work-space cost remains \f$ \Theta(1) \f$.
	 * 
	 * In case of a distributed-memory parallel backend and use of this factory
	 * class in #grb::PARALLEL I/O \a mode over \f$ P \f$ user processes with
	 * \f$ T_s \f$ threads at user process \f$ s \f$, thread-local work and data
	 * movement become \f$ \mathcal{O}((n+m)/(T_sP)+T_s+P) \f$.
	 * System-wide compute costs thus are proportional to
	 *   \f$ \mathcal{O}(\min_s (m+n)/(T_sP) + \max_s T_s + P), \f$
	 * while system-wide data movement costs are proportional to
	 *   \f$ \mathcal{O}((m+n)/P + \max_s T_s + P). \f$
	 * The work-space costs are \f$ \Theta( P ) \f$.
	 *
	 * In sequential I/O mode, we give only the system-wide costing for brevity:
	 *   -# \f$ \mathcal{O}( \min m+n / T_s + \max T_s + P ) \f$ work;
	 *   -# \f$ \mathcal{O}( m + n + \max T_s + P ) \f$ data movement;
	 *   -# \f$ \Theta( P ) \f$ work-space.
	 *
	 * \warning Thus, the use of the sequential I/O mode is never scalable in
	 *          \f$ P \f$ and discouraged always.
	 *
	 * @see grb::IOMode for a more in-depth description of sequential (versus
	 *                  parallel) I/O semantics.
	 *
	 * \note This analysis assumes bandwidth is shared amongst all threads in a
	 *       shared-memory system, whereas each user process has exclusive use
	 *       of its memory controllers. These assumptions require properly
	 *       configured execution environments.
	 * \endparblock
	 *
	 * \par A note on aliases
	 * Different from core primitives, the goal for this factory class is to
	 * provide productive tools. This goal differs from the core primitives which
	 * aim to provide some minimal set of primitives that allow efficient
	 * implementation of the widest possible range of algorithms.
	 *
	 * Given this difference in end-goal, maintaining multiple aliases for this
	 * factory class -- or indeed for algorithms in general -- are hence
	 * encouraged, as long as they indeed increase productivity.
	 */
	template<
		typename D,
		grb::IOMode mode = grb::PARALLEL,
		grb::Backend backend = grb::config::default_backend,
		typename RIT = grb::config::RowIndexType,
		typename CIT = grb::config::ColIndexType,
		typename NIT = grb::config::NonzeroIndexType
	>
	class matrices {

		friend class matrices< void, mode, backend >;

		private:

			/** Short-hand typedef for the matrix return type. */
			typedef Matrix< D, backend, RIT, CIT, NIT > MatrixType;

			/**
			 * @returns The number of user processes
			 *
			 * This number is used to establish in how many pieces input containers must
			 * be cut.
			 *
			 * If the number is larger than one, then #getPID establishes the unique ID
			 * of the calling user process.
			 */
			static size_t getP() {
				return mode == grb::SEQUENTIAL ? 1 : grb::spmd<backend>::nprocs();
			}

			/**
			 * @returns The unique user process ID.
			 *
			 * This value is stricly less than what #getP returns.
			 */
			static size_t getPID() {
				return mode == grb::SEQUENTIAL ? 0 : grb::spmd<backend>::pid();
			}

			/**
			 * Given a matrix size as well as a diagonal offset, computes the length
			 * (in number of elements) of that diagonal.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 * @param[in] k The diagonal offset (may be a negative integer).
			 *
			 * @returns The diagonal length, in number of elements.
			 */
			static size_t compute_diag_length(
				const size_t m, const size_t n,
				const long k
			) {
				constexpr const long zero = static_cast< long >( 0 );
				const size_t k_abs = static_cast< size_t >(
					(k < zero) ? -k : k );
				// catch out-of-bounds offsets
				if( k < zero && k_abs >= m ) {
					return 0;
				} else if( k > zero && k_abs >= n ) {
					return 0;
				}
				return std::min( std::min( m, n ), k < zero
						? m - k_abs
						: n - k_abs
					);
			}

			/**
			 * Generic implementation for creating an identity matrix.
			 *
			 * @tparam IteratorV The iterator type for diagonal values. This is
			 *                   strongly recommended to be a random access iterator so
			 *                   as to facilitate shared-memory parallel ingestion, but
			 *                   any forward iterator is accepted as well(!).
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 * @param[in] k The diagonal offset (may be a negative integer).
			 *
			 * @param[in] V_iter Start iterator to a collection of values to be put on the
			 *                   diagonal.
			 * @param[in] V_end  Iterator to the same collection as \a V_iter, at its end
			 *                   position.
			 *
			 * The \a V_iter and \a V_end iterator pair is assumed to match the I/O mode
			 * of the matrix factory class this method is part of. If it does not match,
			 * undefined behaviour will occur.
			 *
			 * The \a V_iter and \a V_end iterator pair must contain exactly the number
			 * of elements that should appear on the requested diagonal.
			 *
			 * \warning In parallel I/O mode, the given iterator pair must match in
			 *          terms of both contents and order of returned elements, to those
			 *          of the row- and column-indices returned by the \em internally-
			 *          defined iterators, which (currently) are given by:
			 *            - #grb::utils::iterators::Range .
			 *          Therefore, the input values iterators should always remain hidden
			 *          from the humble user.
			 */
			template< class IteratorV >
			static MatrixType createIdentity_generic(
				const size_t m, const size_t n, const long k,
				const IteratorV V_iter, const IteratorV V_end
			) {
#ifdef _DEBUG
				std::cout << "createIdentity_generic called with m = " << m << ", n = "
					<< n << ", k = " << k << " (non-void variant)\n";
#endif
				// some useful scalars
				constexpr const long s_zero = static_cast< long >( 0 );
				constexpr const size_t u_zero = static_cast< size_t >( 0 );
				const size_t diag_length = compute_diag_length( m, n, k );
#ifdef NDEBUG
				(void) V_end;
#endif
#ifdef _DEBUG
				std::cout << "Computed diag_length = " << diag_length << "\n";
#endif

				// declare matrix-to-be-returned
				MatrixType matrix( m, n, diag_length );

				// get row- and column-wise starting positions
				const RIT k_i_incr = static_cast< RIT >(
					(k < s_zero) ? std::abs( k ) : u_zero );
				const CIT k_j_incr = static_cast< CIT >(
					(k < s_zero) ? u_zero : std::abs( k ) );

				// translate it to a range so we can get iterators
				grb::utils::containers::Range< RIT > I( k_i_incr, diag_length + k_i_incr );
				grb::utils::containers::Range< CIT > J( k_j_incr, diag_length + k_j_incr );

				// construct the matrix from the given iterators
				const size_t s = getPID();
				const size_t P = getP();
				assert( static_cast< size_t >(std::distance( V_iter, V_end )) ==
					static_cast< size_t >(std::distance( I.begin( s, P ), I.end( s, P ) )) );
				assert( static_cast< size_t >(std::distance( V_iter, V_end )) ==
					static_cast< size_t >(std::distance( J.begin( s, P ), J.end( s, P ) )) );
				const RC rc = buildMatrixUnique(
					matrix,
					I.begin( s, P ), I.end( s, P ),
					J.begin( s, P ), J.end( s, P ),
					V_iter, V_end,
					mode
				);

				if( rc != SUCCESS ) {
					throw std::runtime_error(
						"Error: createIdentity_generic failed: rc = " + grb::toString( rc )
					);
				}
				return matrix;
			}


		public:

			/**
			 * Builds an empty matrix, without any values.
			 *
			 * @param[in] m The requested number of rows of the returned matrix.
			 * @param[in] n The requested number of columns of the returned matrix.
			 *
			 * @returns An empty matrix of the requested type.
			 */
			static MatrixType empty(
				const size_t m, const size_t n
			) {
				return MatrixType( m, n, 0 );
			}

			/**
			 * Builds a diagonal matrix with the given values.
			 *
			 * This function takes as input an iterator-pair over any STL-style container
			 * of diagonal values.
			 *
			 * @tparam ValueIterator The type of the iterator used to provide the values.
			 *                       This iterator must be a forward iterator.
			 *
			 * \warning If \a ValueIterator is not also a random access iterator, then
			 *          construction of the requested matrix cannot be parallelised.
			 *
			 * The output matrix will contain \f$ q = \min\{ m, n \} \f$ non-zeroes or less
			 * if \a k is not zero.
			 *
			 * @param[in] m      The number of rows of the matrix.
			 * @param[in] n      The number of columns of the matrix.
			 * @param[in] values The iterator over the diagonal values.
			 * @param[in] valEnd Iterator in end-position matching \a values.
			 * @param[in] k      The diagonal offset (default = 0). A positive value
			 *                   indicates an offset above the main diagonal, while a
			 *                   negative value indicates an offset below the main
			 *                   diagonal.
			 *
			 * The distance between \a values and \a valEnd must be at least \f$ q \f$, or
			 * less if \a k is not zero.
			 *
			 * If \a mode is #grb::PARALLEL, then the \em union of elements spanned by
			 * \a values and \a valEnd across all user processes making a collective call
			 * to this function, comprises the set of all input values. If \a mode is
			 * #grb::SEQUENTIAL, then the elements spanned at each user process comprises
			 * the set of all input values, and the complete set of inputs must match
			 * over all user processes. See also #grb::IOMode.
			 *
			 * @returns The requested diagonal matrix.
			 *
			 * \warning This function is currently only implemented for sequential I/O.
			 */
			template< class ValueIterator >
			static MatrixType diag(
				const size_t m, const size_t n,
				const ValueIterator values, const ValueIterator valEnd,
				const long k = static_cast< long >( 0 )
			) {
				// static sanity checks
				static_assert( mode == grb::SEQUENTIAL,
					"matrices<>::diag is currently only supported with sequential I/O" );
				static_assert(
					std::is_convertible<
						typename std::iterator_traits< ValueIterator >::value_type,
						D
					>::value,
					"grb::factory::diag called with value types that are not convertible to the "
					"matrix nonzero type."
				);

				// check trivial dispatch
				if( m == 0 || n == 0 ) {
					return empty( n, n );
				}

				// call generic implementation with given iterator
				return createIdentity_generic( m, n, k, values, valEnd );
			}

			/**
			 * Builds a diagonal matrix.
			 *
			 * The output matrix will contain \f$ \min\{ m, n \} \f$ non-zero elements
			 * or less, if \a k is not zero.
			 *
			 * @param[in] m     The number of rows of the matrix.
			 * @param[in] n     The number of columns of the matrix.
			 * @param[in] value The value of each non-zero element.
			 * @param[in] k     The diagonal offset. A positive value indicates an offset
			 *                  above the main diagonal, while a negative value indicates
			 *                  an offset below the main diagonal.
			 *
			 * Providing \a value equal to one, \a k equal to zero, and \a n equal to
			 * \a m, is equivalent to a default call to #identity. Therefore, \a value,
			 * \a k, and \a n are defined as optional arguments to this function as well,
			 * with defaults one and zero, respectively.
			 *
			 * @returns The requested diagonal matrix.
			 */
			static MatrixType eye(
				const size_t m,
				const size_t n = m,
				const D value = static_cast< D >( 1 ),
				const long k = static_cast< long >( 0 )
			) {
				// check for possible trivial dispatch
				if( m == 0 || n == 0 ) {
					return empty( m, n );
				}

				// we are in the non-pattern case, so we need a repeated iterator for the
				// values-- determine worst-case length (cheaper than computing the actual
				// diagonal length)
				const size_t diag_length = compute_diag_length( m, n, k );
				const grb::utils::containers::ConstantVector< D > V( value,
					diag_length );

				// call generic implementation
				const size_t s = getPID();
				const size_t P = getP();
				return createIdentity_generic( m, n, k, V.cbegin( s, P ), V.cend( s, P ) );
			}

			/**
			 * Builds an identity matrix.
			 *
			 * \note This is an alias for #eye. It differs only in that this function
			 *       does not allow for rectangular output, nor for non-identity values
			 *       on the diagonal, nor for diagonal offsets.
			 *
			 * See #eye for detailed documentation.
			 *
			 * @param[in] n    The size of the identity matrix to be returned.
			 * @param[in] ring The semiring under which an identity matrix should be
			 *                 formed (optional -- the default simply puts ones on the
			 *                 diagonal).
			 *
			 * \note For non-numerical \a D, the semiring default cannot be applied and
			 *       \a ring becomes a mandatory argument.
			 *
			 * @returns The requested identity matrix.
			 */
			template<
				typename Semiring = grb::Semiring<
					grb::operators::add< D >, grb::operators::mul< D >,
					grb::identities::zero, grb::identities::one
				>
			>
			static MatrixType identity(
				const size_t n,
				const Semiring &ring = Semiring()
			) {
				const D identity_value = ring.template getOne< D >();
				return eye( n, n, identity_value, 0 );
			}

			/**
			 * Builds a dense matrix filled with a given value.
			 *
			 * \note ALP/GraphBLAS does not have efficient support for dense matrices--
			 *       the dense data will be stored in a sparse format, which will not
			 *       lead to efficient computations. See ALP/Dense for efficient dense
			 *       linear algebra support.
			 *
			 * Output matrix will contain \f$ mn \f$ non-zero elements.
			 *
			 * @param[in] m     The number of rows of the matrix.
			 * @param[in] n     The number of columns of the matrix.
			 * @param[in] value The value of each non-zero element.
			 *
			 * @returns The requested full matrix.
			 */
			static MatrixType full(
				const size_t m, const size_t n,
				const D value
			) {
				if( m == 0 || n == 0 ) {
					return empty( m, n );
				}

				const size_t nz = m * n;
				if( nz / m != n ) {
					throw std::runtime_error( "Requested dense matrix overflows in number of "
						"nonzeroes." );
				}

				const size_t s = getPID();
				const size_t P = getP();

				// Initialise rows indices container with a range from 0 to nrows,
				// each value repeated ncols times.
				grb::utils::containers::Range< RIT > I( 0, m, 1, n );

				// Initialise columns values container with a range from 0 to ncols
				// repeated nrows times. There are two ways of doing this:
				//  1) using InterleavedIterators, or
				//  2) using the iterator::Adapter.
				// We select here way #2, and disable way #1:
#if 0
				grb::utils::containers::InterleavedIteratorsVector<
						typename grb::utils::containers::Range< CIT >::const_iterator
					> J( m );
				for( size_t i = 0; i < m; ++i ) {
					J.emplace_back( grb::utils::containers::Range< CIT >( 0, n ) );
				}
#endif
				const grb::utils::containers::Range< size_t > J_raw( 0, nz );
				const auto entryInd2colInd = [&m] (const CIT k) -> CIT {
						return k / m;
					};
				auto J_begin = grb::utils::iterators::make_adapter_iterator(
					J_raw.cbegin( s, P ), J_raw.cend( s, P ), entryInd2colInd );
				const auto J_end = grb::utils::iterators::make_adapter_iterator(
					J_raw.cend( s, P ), J_raw.cend( s, P ), entryInd2colInd );

				// Initialise values container with the given value.
				const size_t local_nz = std::distance( I.begin( s, P ), I.end( s, P ) );
#ifndef NDEBUG
				const size_t Isz = local_nz;
				const size_t Jsz = std::distance( J_begin, J_end );
				assert( Isz == Jsz );
#endif
				grb::utils::containers::ConstantVector< D > V( value, local_nz );
#ifndef NDEBUG
				const size_t Vsz = std::distance( V.begin(), V.end() );
				assert( Isz == Vsz );
#endif

				// allocate and build
				MatrixType matrix( m, n, nz );
				const RC rc = buildMatrixUnique(
					matrix,
					I.begin( s, P ), I.end( s, P ),
					J_begin, J_end,
					V.begin(), V.end(),
					mode
				);

				if( rc != SUCCESS ) {
					throw std::runtime_error(
						"Error: factory::full<void> failed: rc = " + grb::toString( rc )
					);
				}

				return matrix;
			}

			/**
			 * Builds a dense matrix filled with a given value.
			 *
			 * \note This is an alias for #full -- see that function for complete
			 *       documentation.
			 *
			 * @param[in] m     The number of rows of the matrix.
			 * @param[in] n     The number of columns of the matrix.
			 * @param[in] value The value of each non-zero element.
			 *
			 * @returns The requested dense matrix.
			 */
			static MatrixType dense(
				const size_t m, const size_t n, const D value
			) {
				return full( m, n, value );
			}

			/**
			 * Builds a dense matrix from a given matrix.
			 *
			 * Indended for converting sparse input matrices to a dense format. Assumes
			 * numerical data types, and that entries previously not set by the input
			 * matrix, will equal zero in the returned matrix.
			 *
			 * While not strictly a factory constructor, the definition of the above
			 * #dense function does most likely raise the user expectation that the
			 * functionality provided by this method is available.
			 *
			 * @param[in] A The (sparse) input matrix.
			 *
			 * @param[in] ring The semiring in case 'zero' is not intended to be the
			 *                 standard numerical zero. This argument is optional; by
			 *                 default, the numerical zero will be used.
			 *
			 * \note In case of non-numerical \a D, \a ring becomes a mandatory
			 *       argument.
			 *
			 * @returns The matrix \a A converted to a dense format. More precisely,
			 *          entries \f$ a_{ij} \in A \f$ will equal that of the returned
			 *          matrix at position \f$ (i, j) \f$. Coordinates $(k, l)$ for
			 *          which no entry \f$ a_{kl} \f$ existed in \f$ A \f$ will have
			 *          value 0 in the returned matrix.
			 */
			template<
				typename Semiring = grb::Semiring<
					grb::operators::add< D >, grb::operators::mul< D >,
					grb::identities::zero, grb::identities::one
				>
			>
			static MatrixType dense(
				const MatrixType &A,
				const Semiring &ring = Semiring()
			) {
				const size_t m = grb::nrows( A );
				const size_t n = grb::ncols( A );
				if( n == 0 || m == 0 ) {
					return empty( m, n );
				}

				if( grb::nnz( A ) == 0 ) {
					return zeros( m, n, ring );
				}

				const size_t nz = m * n;
				if( nz / m != n ) {
					throw std::runtime_error( "Requested dense matrix overflows in number of "
						"nonzeroes." );
				}

				const auto addMon = ring.getAdditiveMonoid();
				const D zero = ring.template getZero< D >();
				MatrixType matrix( m, n, nz );
				grb::RC rc = grb::set( matrix, zero );
				rc = rc ? rc : grb::foldl( matrix, A, addMon );

				if( rc != grb::SUCCESS) {
					throw std::runtime_error( "Could not promote input matrix to a dense one: "
						+ grb::toString( rc ) );
				}

				return matrix;
			}

			/**
			 * Builds a matrix filled with zeros.
			 *
			 * \note This is an alias for factory::full( m, n, 0 )
			 *
			 * @see #full for complete documentation.
			 *
			 * @param[in] m    The number of rows of the matrix.
			 * @param[in] n    The number of columns of the matrix.
			 * @param[in] ring The semiring under which a matrix of zeroes should be
			 *                 formed (optional -- the default simply produces a matrix
			 *                 of numerical zeroes).
			 *
			 * \note For non-numerical \a D, the semiring default cannot be applied and
			 *       \a ring becomes a mandatory argument.
			 *
			 * @returns A dense matrix of zeroes.
			 */
			template<
				typename Semiring = grb::Semiring<
					grb::operators::add< D >, grb::operators::mul< D >,
					grb::identities::zero, grb::identities::one
				>
			>
			static MatrixType zeros(
				const size_t m, const size_t n,
				const Semiring &ring = Semiring()
			) {
				const D zero = ring.template getZero< D >();
				return full( m, n, static_cast< D >( zero ) );
			}

			/**
			 * Builds a matrix filled with ones.
			 *
			 * \note This is an alias for <tt>full( m, n, 1 )</tt>.
			 *
			 * @see #full for complete documentation.
			 *
			 * @param[in] m    The number of rows of the matrix.
			 * @param[in] n    The number of columns of the matrix.
			 * @param[in] ring The semiring under which a matrix of ones should be
			 *                 formed (optional -- the default simply produces a matrix
			 *                 of numerical ones).
			 *
			 * \note For non-numerical \a D, the semiring default cannot be applied and
			 *       \a ring becomes a mandatory argument.
			 *
			 * @returns Returns a dense matrix with entries one.
			 */
			template<
				typename Semiring = grb::Semiring<
					grb::operators::add< D >, grb::operators::mul< D >,
					grb::identities::zero, grb::identities::one
				>
			>
			static MatrixType ones(
				const size_t m, const size_t n,
				const Semiring &ring = Semiring()
			) {
				const D one = ring.template getOne< D >();
				return full( m, n, static_cast< D >( one ) );
			}

	}; // end class matrices

	/**
	 * Factories for creating ALP/GraphBLAS pattern matrices.
	 *
	 * This is the specialisation for <tt>void</tt> data types; see the
	 * non-pattern, generic, class for the complete documentation.
	 */
	template<
		grb::IOMode mode, grb::Backend backend,
		typename RIT, typename CIT, typename NIT
	>
	class matrices< void, mode, backend, RIT, CIT, NIT > {

		private:

			/** Short-hand typedef for the matrix return type. */
			typedef Matrix< void, backend, RIT, CIT, NIT > MatrixType;

			/**
			 * Short-hand type to the base implementation that defines some useful
			 * helper functions.
			 */
			typedef matrices< int, mode, backend, RIT, CIT, NIT > BaseType;

			/**
			 * Generic implementation for creating an identity matrix.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 * @param[in] k The diagonal offset (may be a negative integer).
			 *
			 * This is the variant for void matrices.
			 */
			static MatrixType createIdentity_generic(
				const size_t m, const size_t n, const long k
			) {
				// pattern matrix variant of the above
				constexpr const long s_zero = static_cast< long >( 0 );
				constexpr const size_t u_zero = static_cast< size_t >( 0 );
				const size_t diag_length = BaseType::compute_diag_length( m, n, k );
				MatrixType matrix( m, n, diag_length );
				const RIT k_i_incr = static_cast< RIT >(
					(k < s_zero) ? std::abs( k ) : u_zero );
				const CIT k_j_incr = static_cast< CIT >(
					(k < s_zero) ? u_zero : std::abs( k ) );
				grb::utils::containers::Range< RIT > I( k_i_incr, diag_length + k_i_incr );
				grb::utils::containers::Range< CIT > J( k_j_incr, diag_length + k_j_incr );
				const size_t s = BaseType::getPID();
				const size_t P = BaseType::getP();
				const RC rc = buildMatrixUnique(
					matrix,
					I.begin( s, P ), I.end( s, P ),
					J.begin( s, P ), J.end( s, P ),
					mode
				);
				if( rc != SUCCESS ) {
					throw std::runtime_error(
						"Error: createIdentity_generic<void> failed: rc = " + grb::toString( rc )
					);
				}
				return matrix;
			}


		public:

			/**
			 * Builds an empty matrix, without any values.
			 *
			 * @param[in] m The requested number of rows of the returned matrix.
			 * @param[in] n The requested number of columns of the returned matrix.
			 *
			 * @returns An empty matrix of the requested type.
			 */
			static MatrixType empty(
				const size_t m, const size_t n
			) {
				return MatrixType( m, n, 0 );
			}

			/**
			 * Builds a diagonal pattern matrix.
			 *
			 * The output matrix will contain \f$ q = \min\{ m, n \} \f$ non-zeroes or less
			 * if \a k is not zero.
			 *
			 * \note This method is specialised for pattern matrices (void non-zero type).
			 *
			 * For pattern matrices, a call to this function is an alias of a call to
			 * #eye. (For general matrices, the functions are \em not equivalent.)
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 * @param[in] k The diagonal offset (default = 0). A positive value indicates
			 *              an offset above the main diagonal, while a negative value
			 *              indicates an offset below the main diagonal.
			 *
			 * @returns The requested diagonal matrix.
			 */
			MatrixType diag(
				const size_t m, const size_t n,
				const long k = static_cast< long >( 0 )
			) {
				// check trivial dispatch
				if( m == 0 || n == 0 ) {
					return empty( n, n );
				}

				// call generic implementation with given iterator
				return createIdentity_generic( m, n, k );
			}

			/**
			 * Builds an identity pattern matrix.
			 *
			 * Output matrix will contain \f$ \min\{ m, n \} \f$ non-zero elements or less
			 * if \a k is not zero.
			 *
			 * \note This method is specialised for pattern matrices (void non-zero type).
			 *
			 * For pattern matrices, a call to this function is an alias of a call to
			 * #diag.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 * @param[in] k The diagonal offset (default = 0). A positive value indicates
			 *              an offset above the main diagonal, while a negative value
			 *              indicates an offset below the main diagonal.
			 *
			 * @returns The requested identity matrix.
			 */
			static MatrixType eye(
				const size_t m,
				const size_t n,
				const long k = static_cast< long >( 0 )
			) {
				// check trivial case
				if( m == 0 || n == 0 ) {
					return empty( m, n );
				}

				// dispatch to generic function
				return createIdentity_generic( m, n, k );
			}

			/**
			 * Builds an identity pattern matrix.
			 *
			 * \note This is the pattern matrix variant -- see the non-pattern variant
			 *       for complete documentation.
			 *
			 * @param[in] n The size of the identity matrix to be returned.
			 *
			 * \note A semiring is not requested as what value represents one is not
			 *       relevant for pattern matrices.
			 *
			 * @returns The requested identity pattern matrix.
			 */
			static MatrixType identity(
				const size_t n
			) {
				return eye( n, n, 0 );
			}

			/**
			 * Build a dense pattern matrix.
			 *
			 * \note ALP/GraphBLAS does not have efficient support for dense matrices--
			 *       the dense data will be stored in a sparse format, which will not
			 *       lead to efficient computations. See ALP/Dense for efficient dense
			 *       linear algebra support.
			 *
			 * Output matrix will contain \f$ mn \f$ non-zero elements.
			 *
			 * \note This method is specialised for pattern matrices (void non-zero type).
			 *
			 * @param[in] m     The number of rows of the matrix.
			 * @param[in] n     The number of columns of the matrix.
			 *
			 * @returns The requested full matrix.
			 */
			static MatrixType full( const size_t m, const size_t n ) {
				if( m == 0 || n == 0 ) {
					return empty( m, n );
				}

				const size_t nz = m * n;
				if( nz / m != n ) {
					throw std::runtime_error( "Requested dense matrix overflows in number of "
						"nonzeroes." );
				}

				const size_t s = BaseType::getPID();
				const size_t P = BaseType::getP();

				// Initialise rows indices container with a range from 0 to nrows,
				// each value repeated ncols times.
				grb::utils::containers::Range< RIT > I( 0, m, 1, n );

				// Initialise columns values container with a range from 0 to ncols
				// repeated nrows times. As mentioned above, there are two ways to provide
				// iterators, we disable the first way (via InterleavedIterators) and enable
				// the iterators::adaptor way:
#if 0
				grb::utils::containers::InterleavedIteratorsVector<
						typename grb::utils::containers::Range< CIT >::const_iterator
					> J( m );
				for( size_t i = 0; i < m; ++i ) {
					J.emplace_back( grb::utils::containers::Range< CIT >( 0, n ) );
				}
#endif
				const grb::utils::containers::Range< size_t > J_raw( 0, nz );
				const auto nonzeroInd2colInd = [&m] (const size_t k) -> size_t {
						return k / m;
					};
				auto J_begin = grb::utils::iterators::make_adapter_iterator(
					J_raw.cbegin( s, P ), J_raw.cend( s, P ), nonzeroInd2colInd );
				auto J_end = grb::utils::iterators::make_adapter_iterator(
					J_raw.cend( s, P ), J_raw.cend( s, P ), nonzeroInd2colInd );

				assert( std::distance( I.begin( s, P ), I.end( s, P ) ) ==
					std::distance( J_begin, J_end ) );

				// construct and populate matrix
				MatrixType matrix( m, n, nz );
				const RC rc = buildMatrixUnique(
					matrix,
					I.begin( s, P ), I.end( s, P ),
					J_begin, J_end,
					mode
				);

				if( rc != SUCCESS ) {
					throw std::runtime_error(
						"Error: factory::full<void> failed: rc = " + grb::toString( rc )
					);
				}

				return matrix;
			}

			/**
			 * Builds a dense pattern matrix.
			 *
			 * \note This is an alias for #full -- see that function for complete
			 *       documentation.
			 *
			 * \note This is the specialisation for pattern matrices.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 *
			 * @returns The requested dense matrix.
			 */
			static MatrixType dense( const size_t m, const size_t n ) {
				return full( m, n );
			}

			/**
			 * Builds a dense pattern matrix from a given matrix.
			 *
			 * \note This is the pattern (void matrix) variant, for which this function
			 *       ignores the input sparse matrix and behaves exactly the same as
			 *       a call to #full with the dimensions of \a A.
			 *
			 * @param[in] A The (sparse) input matrix.
			 *
			 * \note A semiring is not requested as what value represents one is not
			 *       relevant for pattern matrices.
			 *
			 * @returns A dense pattern matrix.
			 */
			static MatrixType dense( const MatrixType &A ) {
				(void) A;
				return full( grb::nrows( A ), grb::ncols( A ) );
			}

			/**
			 * Builds a matrix filled with zeros.
			 *
			 * \note This is an alias for factory::full( m, n, 0 ). Since a pattern
			 *       matrix is requested, however, the numerical value 0 is meaningless;
			 *       this function simply returns a dense pattern matrix.
			 *
			 * @see #full for complete documentation.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 *
			 * \note A semiring is not requested as what value represents zero is not
			 *       relevant for pattern matrices.
			 *
			 * @returns A dense pattern matrix.
			 */
			static MatrixType zeros( const size_t m, const size_t n ) {
				return full( m, n );
			}

			/**
			 * Builds a dense pattern matrix filled with ones.
			 *
			 * \note This is an alias for factory::full( m, n, 1 ). Since a pattern
			 *       matrix is requested, however, the numerical value 1 is meaningless;
			 *       the result is simply a fully dense pattern matrix.
			 *
			 * @see #full for complete documentation.
			 *
			 * @param[in] m The number of rows of the matrix.
			 * @param[in] n The number of columns of the matrix.
			 *
			 * \note A semiring is not requested as what value represents one is not
			 *       relevant for pattern matrices.
			 *
			 * @returns Returns a dense pattern matrix.
			 */
			static MatrixType ones( const size_t m, const size_t n ) {
				return full( m, n );
			}

	}; // end class matrices (pattern specialisation)

} // namespace grb::algorithms

#endif // end _H_GRB_MATRIX_FACTORY

