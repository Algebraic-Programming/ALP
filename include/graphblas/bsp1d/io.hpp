
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
 * @date 16th of February, 2017
 */

#ifndef _H_GRB_BSP1D_IO
#define _H_GRB_BSP1D_IO

#include <memory>

#include "graphblas/blas1.hpp"                 // for grb::size
#include "graphblas/utils/NonzeroIterator.hpp" // for transforming an std::vector::iterator
                                               // into an ALP/GraphBLAS-compatible iterator
#include "graphblas/utils/pattern.hpp"         // for handling pattern input
#include <graphblas/base/io.hpp>

#include "lpf/core.h"
#include "matrix.hpp" //for BSP1D matrix
#include "vector.hpp" //for BSP1D vector

#define NO_CAST_ASSERT( x, y, z )                                                  \
	static_assert( x,                                                              \
		"\n\n"                                                                     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"*     ERROR      | " y " " z ".\n"                                        \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n"                                                 \
		"* Possible fix 1 | Remove no_casting from the template parameters in "    \
		"this call to " y ".\n"                                                    \
		"* Possible fix 2 | Provide a value input iterator with element types "    \
		"that match the output vector element type.\n"                             \
		"* Possible fix 3 | If applicable, provide an index input iterator with "  \
		"element types that are integral.\n"                                       \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );


namespace grb {

	/**
	 * \defgroup IO Data Ingestion -- BSP1D backend
	 * @{
	 */

	/** \internal No implementation details. */
	template< typename InputType, typename Coords >
	uintptr_t getID( const Vector< InputType, BSP1D, Coords > &x ) {
		return x._id;
	}

	/** \internal No implementation details. */
	template< typename InputType >
	uintptr_t getID( const Matrix< InputType, BSP1D > &A ) {
		return A._id;
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename Coords >
	size_t size( const Vector< DataType, BSP1D, Coords > &x ) noexcept {
		return x._n;
	}

	/** \internal No implementation notes. */
	template< typename DataType >
	size_t nrows( const Matrix< DataType, BSP1D > &A ) noexcept {
		return A._m;
	}

	/** \internal No implementation notes. */
	template< typename DataType >
	size_t ncols( const Matrix< DataType, BSP1D > &A ) noexcept {
		return A._n;
	}

	/** \internal No implementation notes. */
	template< typename DataType, typename Coords >
	size_t capacity( const Vector< DataType, BSP1D, Coords > &x ) noexcept {
		return x._cap;
	}

	/** \internal No implementation notes. */
	template< typename DataType >
	size_t capacity( const Matrix< DataType, BSP1D > &A ) noexcept {
		return A._cap;
	}

	/**
	 * \internal Uses grb::collectives::alreduce. Can throw exceptions.
	 *
	 * \todo Internal issue #200 -- this function should not be able to throw
	 *       exceptions.
	 */
	template< typename DataType, typename Coords >
	size_t nnz( const Vector< DataType, BSP1D, Coords > &x ) {
		// first update number of nonzeroes (and _became_dense flag)
		if( x.updateNnz() != SUCCESS ) {
			throw std::runtime_error( "Unrecoverable error during update of "
				"the global nonzero count."
			);
		}
		// done
		return x._nnz;
	}

	/**
	 * Implementation details: relies on grb::collectives::allreduce.
	 *
	 * \todo internal issue #200 -- allreduce could fail, which is not acceptable.
	 *
	 * @see grb::nnz for the user-level specification.
	 */
	template< typename DataType >
	size_t nnz( const Matrix< DataType, BSP1D > &A ) noexcept {
#ifdef _DEBUG
		std::cout << "Called grb::nnz (matrix, BSP1D).\n";
#endif
		// get local number of nonzeroes
		size_t ret = nnz( A._local );
#ifdef _DEBUG
		std::cout << "\t local number of nonzeroes: " << ret << std::endl;
#endif
		// call allreduce on it
		collectives< BSP1D >::allreduce<
			descriptors::no_casting,
			operators::add< size_t >
		>( ret );
#ifdef _DEBUG
		std::cout << "\t global number of nonzeroes: " << ret << std::endl;
#endif
		// after allreduce, return sum of the local nonzeroes
		return ret;
	}

	/**
	 * Clears a given vector of all values.
	 *
	 * \parblock
	 * \par Performance semantics
	 * This primitive inherits the performance semantics of #grb::clear of the
	 * underlying process-local backend, which is the reference backend by default.
	 * It adds to those:
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ work,
	 *   -# \f$ \Theta( P ) \f$ intra-process data movement,
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ inter-process
	 *      data movement,
	 *   -# one inter-process synchronisation step.
	 * Here, \f$ P \f$ is the number of user processes.
	 * \endparblock
	 *
	 * \internal No implementation notes.
	 */
	template< typename DataType, typename Coords >
	RC clear( Vector< DataType, BSP1D, Coords > &x ) noexcept {
		const RC ret = clear( internal::getLocal( x ) );
		if( ret == SUCCESS ) {
			x._cleared = true;
			internal::signalLocalChange( x );
		}
		return ret;
	}

	/**
	 * Clears a given matrix of all values.
	 *
	 * \parblock
	 * \par Performance semantics
	 * This primitive inherits the performance semantics of #grb::clear of the
	 * underlying process-local backend, which is the reference backend by default.
	 * It does not add any costs beyond those.
	 * \endparblock
	 *
	 * \internal No implementation notes.
	 */
	template< typename IOType >
	RC clear( grb::Matrix< IOType, BSP1D > &A ) noexcept {
		return grb::clear( internal::getLocal( A ) );
	}

	/**
	 * Resizes the capacity of a given vector.
	 *
	 * \parblock
	 * \par Performance semantics
	 * This primitive inherits the performance semantics of #grb::resize of the
	 * underlying process-local backend, which is the reference backend by default.
	 * It adds to those:
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ work,
	 *   -# \f$ \Theta( P ) \f$ intra-process data movement,
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ inter-process
	 *      data movement,
	 *   -# two inter-process synchronisation steps.
	 * Here, \f$ P \f$ is the number of user processes.
	 * \endparblock
	 *
	 * \note The two synchronisation steps are required for error detection and
	 *       global capacity synchronisation, respectively; note that even though
	 *       the current process may report no errors, others might.
	 *
	 * \todo Employ non-blocking collectives and arbitrary-order write conflict
	 *       resolution to enable both synchronisations within a single step.
	 *
	 * \internal
	 * For sparse vectors, there is no way of knowing beforehand which element
	 * is distributed where. Therefore, \a new_nz can only be interpreted as a
	 * local value, although the user gives a global number. We first detect a
	 * mismatch, then correct the value against the local maximum length, and
	 * then delegate to the underlying backend.
	 */
	template< typename InputType, typename Coords >
	RC resize(
		Vector< InputType, BSP1D, Coords > &x,
		const size_t new_nz
	) noexcept {
#ifdef _DEBUG
		std::cerr << "In grb::resize (vector, BSP1D)\n"
			<< "\t vector size is " << size(x) << "\n"
			<< "\t requested capacity is " << new_nz << "\n";
#endif

		// check trivial op
		const size_t n = size( x );
		if( n == 0 ) {
			return clear( x );
		}

		// check if we have a mismatch
		if( new_nz > n ) {
			return ILLEGAL;
		}

		// if \a new_nz is larger than local capacity, correct to local max
		const size_t local_size = grb::size( internal::getLocal( x ) );
		const size_t local_new_nz = new_nz > local_size ? local_size : new_nz;

#ifdef _DEBUG
		std::cerr << "\t will request local capacity " << local_new_nz << "\n";
#endif

		// try activate new capacity
		grb::RC rc = resize( internal::getLocal( x ), local_new_nz );

		// collect global error state
		if( collectives< BSP1D >::allreduce(
				rc, grb::operators::any_or< grb::RC >()
			) != SUCCESS
		) {
			return PANIC;
		}

		// on failure, old capacity remains in effect, so return
		if( rc != SUCCESS ) {
#ifdef _DEBUG
			std::cerr << "\t at least one user process reports error: "
				<< toString( rc ) << "\n";
#endif
			return rc;
		}

		// we have success, so get actual new global capacity
		rc = internal::updateCap( x );
		if( rc != SUCCESS ) { return PANIC; }
		x._nnz = 0;
		x._cleared = true;
		x._global_is_dirty = true;

		// delegate
		return rc;
	}

	/**
	 * Resizes the capacity of a given matrix.
	 *
	 * \parblock
	 * \par Performance semantics
	 * This primitive inherits the performance semantics of #grb::resize of the
	 * underlying process-local backend, which is the reference backend by default.
	 * It adds to those:
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ work,
	 *   -# \f$ \Theta( P ) \f$ intra-process data movement,
	 *   -# \f$ \Omega( \log P ) \f$ and \f$ \mathcal{O}( P ) \f$ inter-process
	 *      data movement,
	 *   -# two inter-process synchronisation steps.
	 * Here, \f$ P \f$ is the number of user processes.
	 * \endparblock
	 *
	 * \note The two synchronisation steps are required for error detection and
	 *       global capacity synchronisation, respectively; note that even though
	 *       the current process may report no errors, others might.
	 *
	 * \todo Employ non-blocking collectives and arbitrary-order write conflict
	 *       resolution to enable both synchronisations within a single step.
	 *
	 * \internal this function reserves the given amount of space <em>at this
	 * user process</em>. Rationale: it cannot be predicted how many nonzeroes
	 * end up at each separate user process, thus global information cannot be
	 * exploited to make rational process-local decisions (in general).
	 */
	template< typename InputType >
	RC resize( Matrix< InputType, BSP1D > &A, const size_t new_nz ) noexcept {
#ifdef _DEBUG
		std::cerr << "In grb::resize (matrix, BSP1D)\n"
			<< "\t matrix is " << nrows( A ) << " by " << ncols( A ) << "\n"
			<< "\t current capacity is " << capacity( A ) << "\n"
			<< "\t requested new capacity is " << new_nz << "\n";
#endif

		RC ret = clear( A );
		if( ret != SUCCESS ) { return ret; }

		// check trivial case and new_nz
		{
			const size_t m = nrows( A );
			const size_t n = ncols( A );
			if( m == 0 || n == 0 ) {
				return SUCCESS;
			}
			if( new_nz / m > n ||
				new_nz / n > m ||
				(new_nz / m == n && (new_nz % m > 0)) ||
				(new_nz / n == m && (new_nz % n > 0))
			) {
#ifdef _DEBUG
				std::cerr << "\t requested capacity is too large\n";
#endif
				return ILLEGAL;
			}
		}

		// delegate to local resize
		size_t old_capacity = capacity( internal::getLocal( A ) );
		const size_t m = nrows( internal::getLocal( A ) );
		const size_t n = ncols( internal::getLocal( A ) );
		// pre-catch trivial local case in order to avoid divide-by-zero
		if( m > 0 && n > 0 ) {
			// make sure new_nz does not overflow locally
#ifdef _DEBUG
			std::cerr << "\t delegating to process-local grb::resize\n";
#endif
			if( new_nz / m > n || new_nz / m > n ) {
				ret = resize( internal::getLocal( A ), m * n );
			} else {
				ret = resize( internal::getLocal( A ), new_nz );
			}
		}

		// check global error state while remembering if locally OK
		bool local_ok = ret == SUCCESS;
		if( collectives< BSP1D >::allreduce(
				ret,
				operators::any_or< RC >()
			) != grb::SUCCESS
		) {
#ifdef _DEBUG
			std::cerr << "\t some user processes reported error\n";
#endif
		}

		// if any one process reports an error, then try to get back old capacity and
		// exit
		if( ret != SUCCESS ) {
			if( local_ok ) {
				if( resize( internal::getLocal( A ), old_capacity ) != SUCCESS ) {
					// this situation is a breach of contract that we (apparently) cannot
					// recover from
#ifdef _DEBUG
					std::cerr << "\t could not recover old capacity\n";
#endif
					return PANIC;
				}
			}
			return ret;
		}

		// everyone is OK, so sync up new global capacity
		size_t new_global_cap = capacity( A._local );
		ret = collectives< BSP1D >::allreduce(
			new_global_cap,
			operators::add< size_t >()
		);
		if( ret != SUCCESS ) {
#ifdef _DEBUG
			std::cerr << "\t could not synchronise new global capacity\n";
#endif
			return PANIC;
		}
		A._cap = new_global_cap;
#ifdef _DEBUG
		std::cerr << "\t new global capacity is " << new_global_cap << "\n";
#endif

		// done
		return ret;
	}

	/** \internal Requires no inter-process communication. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Coords,
		typename T
	>
	RC set(
		Vector< DataType, BSP1D, Coords > &x,
		const T val,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< T >::value, void
		>::type * const = nullptr
	) noexcept {
		const size_t n = size( x );
		const size_t old_nnz = nnz( x );
		if( capacity( x ) < n ) {
			if( phase == RESIZE ) {
				return resize( x, n );
			} else {
				const RC clear_rc = clear( x );
				if( clear_rc != SUCCESS ) {
					return PANIC;
				} else {
					return FAILED;
				}
			}
		}

		assert( capacity( x ) == n );
		if( phase == RESIZE ) {
			return SUCCESS;
		}

		assert( phase == EXECUTE );
		RC ret = SUCCESS;
		if( descr & descriptors::use_index ) {
			const internal::BSP1D_Data &data = internal::grb_BSP1D.cload();
			const auto p = data.P;
			const auto s = data.s;
			const auto n = grb::size( x );
			if( old_nnz < size( x ) ) {
				internal::getCoordinates( internal::getLocal( x ) ).assignAll();
			}
			ret = eWiseLambda( [ &x, &n, &s, &p ]( const size_t i ) {
				x[ i ] = internal::Distribution< BSP1D >::local_index_to_global(
						i, n, s, p
					);
				}, x );
		} else {
			ret = set< descr >( internal::getLocal( x ), val );
		}
		if( ret == SUCCESS ) {
			internal::setDense( x );
		}
		return ret;
	}

	/**
	 * \internal Delegates to underlying backend iff index-to-process translation
	 * indicates ownership.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename DataType, typename Coords,
		typename T
	>
	RC setElement(
		Vector< DataType, BSP1D, Coords > &x,
		const T val,
		const size_t i,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< DataType >::value &&
			!grb::is_object< T >::value, void
		>::type * const = nullptr
	) {
		const size_t n = size( x );
		// sanity check
		if( i >= n ) {
			return MISMATCH;
		}

		// prepare return code and get access to BSP1D data
		RC ret = SUCCESS;
		const internal::BSP1D_Data &data = internal::grb_BSP1D.cload();

		// check if local
		// if( (i / x._b) % data.P != data.s ) {
		if( internal::Distribution< BSP1D >::global_index_to_process_id(
				i, n, data.P
			) == data.s
		) {
			// local, so translate index and perform requested operation
			const size_t local_index =
				internal::Distribution< BSP1D >::global_index_to_local( i, n, data.P );
#ifdef _DEBUG
			std::cout << data.s << ", grb::setElement translates global index "
				<< i << " to " << local_index << "\n";
#endif
			ret = setElement< descr >( internal::getLocal( x ), val, local_index,
				phase );
		}

		// Gather remote error state
		if( collectives< BSP1D >::allreduce( ret, operators::any_or< RC >() )
			!= SUCCESS
		) {
			return PANIC;
		}

		if( phase == RESIZE ) {
			if( ret == SUCCESS ) {
				// on successful local resize, sync new global capacity
				ret = internal::updateCap( x );
			} else if( ret == FAILED ) {
				// on any failed local resize, clear vector
				const RC clear_rc = clear( x );
				if( clear_rc != SUCCESS ) { ret = PANIC; }
			} else {
				assert( ret == PANIC );
			}
		} else {
			assert( phase == EXECUTE );
			if( ret == SUCCESS ) {
				// on successful execute, sync new nnz count
				ret = internal::updateNnz( x );
			}
		}

		// done
		return ret;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, BSP1D, Coords > &x,
		const Vector< InputType, BSP1D, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		// dynamic checks
		if( size( y ) != size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) ) {
				return ILLEGAL;
			}
		}

		// capacity check
		if( capacity( x ) < nnz( y ) ) {
			if( phase == EXECUTE ) {
				const RC clear_rc = clear( x );
				if( clear_rc != SUCCESS ) {
					return PANIC;
				} else {
					return FAILED;
				}
			}
		}

		// all OK, try to do assignment
		RC ret = set< descr >( internal::getLocal( x ),
			internal::getLocal( y ), phase );

		// in resize mode, we hit two collectives and otherwise none
		if( phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce( ret, operators::any_or< RC >() )
				!= SUCCESS
			) {
				return PANIC;
			}
			const RC update_rc = internal::updateCap( x );
			if( ret == SUCCESS ) {
				ret = update_rc;
			} else {
				if( update_rc != SUCCESS ) {
					return PANIC;
				}
			}
		} else {
			assert( phase == EXECUTE );
			// if successful, update nonzero count
			if( ret == SUCCESS ) {
				// reset nonzero count flags
				x._nnz = y._nnz;
				x._nnz_is_dirty = y._nnz_is_dirty;
				x._became_dense = y._became_dense;
				x._global_is_dirty = y._global_is_dirty;
			}
		}

		// done
		return ret;
	}

	/** \internal Requires sync on nonzero structure. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename MaskType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, BSP1D, Coords > &x,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Vector< InputType, BSP1D, Coords > &y,
		const Phase &phase = EXECUTE
	) {
		// check dispatch to simpler variant
		if( size( mask ) == 0 ) {
			return set< descr >( x, y, phase );
		}

		// dynamic checks
		if( grb::size( y ) != grb::size( x ) ) {
			return MISMATCH;
		}
		if( grb::size( mask ) != grb::size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) || nnz( mask ) < size( mask ) ) {
				return ILLEGAL;
			}
		}

		// cannot do capacity pre-check in EXECUTE mode due to the mask and descr;
		// or, rather, the check is possible but only for some combinations, so we
		// rather keep it simple and provide just the generic implementation here

		// all OK, try to do assignment
		RC ret = set< descr >(
			internal::getLocal( x ), internal::getLocal( mask ),
			internal::getLocal( y ),
			phase
		);

		if( collectives< BSP1D >::allreduce( ret, operators::any_or< RC >() )
			!= SUCCESS
		) {
			return PANIC;
		}

		if( phase == RESIZE ) {
			if( ret == SUCCESS ) {
				ret = updateCap( x );
			}
		} else {
			assert( phase == EXECUTE );
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( x );
			} else if( ret == FAILED ) {
				const RC clear_rc = clear( x );
				if( clear_rc != SUCCESS ) {
					ret = PANIC;
				}
			} else {
				assert( ret == PANIC );
			}
		}

		// done
		return ret;
	}

	/** \internal Requires sync on nonzero structure. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename OutputType, typename MaskType, typename InputType,
		typename Coords
	>
	RC set(
		Vector< OutputType, BSP1D, Coords > &x,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const InputType &y,
		const Phase &phase = EXECUTE
	) {
		// check dispatch to simpler variant
		if( size( mask ) == 0 ) {
			return set< descr >( x, y, phase );
		}

		// sanity check
		if( grb::size( mask ) != grb::size( x ) ) {
			return MISMATCH;
		}

		// on capacity pre-check, see above

		// all OK, try to do assignment
		RC ret = set< descr >( internal::getLocal( x ),
			internal::getLocal( mask ), y, phase );

		if( collectives< BSP1D >::allreduce( ret, operators::any_or< RC >() )
			!= SUCCESS
		) {
			return PANIC;
		}

		if( phase == RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateCap( x );
			}
		} else {
			assert( phase == EXECUTE );
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( x );
			} else if( ret == FAILED ) {
				const RC clear_rc = clear( x );
				if( clear_rc != SUCCESS ) {
					ret = PANIC;
				}
			} else {
				assert( ret == PANIC );
			}
		}

		// done
		return ret;
	}

	/**
	 * \internal
	 * Implementation details:
	 *
	 * All user processes read in all input data but record only the data which
	 * are to be stored locally.
	 *
	 * No communication will be incurred. The cost of this function, however, is
	 *   \f$ \Theta( n ) \f$,
	 * where \a n is the global vector size.
	 *
	 * \warning If the number of user processes is larger than one, a parallel
	 *          \a IOMode is not supported.
	 *
	 * \note If the number of user processes is equal to one, a parallel \a IOMode
	 *       is equivalent to a sequential one.
	 *
	 * \warning Thus, this performance of this function does \em not scale.
	 *
	 * @see grb::buildVector for the user-level specification.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator, typename Coords,
		class Dup = operators::right_assign< InputType >
	>
	RC buildVector(
		Vector< InputType, BSP1D, Coords > &x,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode, const Dup &dup = Dup()
	) {
		// static checks
		NO_CAST_ASSERT( (!(descr & descriptors::no_casting) ||
				std::is_same< InputType, typename std::iterator_traits< fwd_iterator >::value_type >::value
			), "grb::buildVector (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set"
		);

		// prepare
		RC ret = SUCCESS;
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();
		// differentiate trivial case from general case
		if( data.P == 1 ) {
			ret = buildVector< descr >( x._local, start, end, SEQUENTIAL, dup );
		} else {
			// parallel mode input is disallowed in the dense constructor.
			if( mode == PARALLEL ) {
				return ILLEGAL;
			} else {
				// sanity check
				assert( mode == SEQUENTIAL );
				// cache only elements going to this processor
				std::vector< InputType > cache;
				size_t i = 0;
				for( ; ret == SUCCESS && start != end; ++start, ++i ) {
					// sanity check
					if( i >= x._n ) {
						ret = MISMATCH;
					} else {
						// if this element is distributed to me
						if( internal::Distribution< BSP1D >::global_index_to_process_id(
								i, x._n, data.P
							) == data.s
						) {
							// cache it locally
							cache.push_back( *start );
						}
					}
				}

				// defer to local constructor
				if( ret == SUCCESS ) {
					ret = buildVector< descr >(
						x._local,
						cache.begin(), cache.end(),
						SEQUENTIAL, dup
					);
				}
			}
		}

		// check for illegal at sibling processes
		if( data.P > 1 && (descr & descriptors::no_duplicates) ) {
#ifdef _DEBUG
			std::cout << "\t global exit-check\n";
#endif
			if( collectives< BSP1D >::allreduce(
					ret, grb::operators::any_or< grb::RC >()
				) != SUCCESS
			) {
				return PANIC;
			}
		}

		// update nnz count
		if( ret == SUCCESS ) {
			x._nnz_is_dirty = true;
			ret = x.updateNnz();
		}

		// done
		return ret;
	}

	/**
	 * \internal
	 * Implementation details:
	 *
	 * In sequential mode, the input from the iterators is filtered and cached in
	 * memory. Afterwards, the buildVector of the reference implementation is
	 * called.
	 *
	 * In parallel mode, the input iterators corresponding to indices that are to
	 * be stored locally, are directly read into local memory. Remote elements are
	 * sent to the process who owns the nonzero via bulk-synchronous message
	 * passing. After the iterators have been exhausted, the incoming message
	 * buffers are drained into the storage memory.
	 * \endinternal
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator1, typename fwd_iterator2,
		typename Coords, class Dup = operators::right_assign< InputType >
	>
	RC buildVector( Vector< InputType, BSP1D, Coords > &x,
		fwd_iterator1 ind_start,
		const fwd_iterator1 ind_end,
		fwd_iterator2 val_start,
		const fwd_iterator2 val_end,
		const IOMode mode,
		const Dup &dup = Dup()
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType, decltype( *std::declval< fwd_iterator2 >() ) >::value ||
				std::is_integral< decltype( *std::declval< fwd_iterator1 >() ) >::value
			), "grb::buildVector (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// get access to user process data on s and P
		const internal::BSP1D_Data & data = internal::grb_BSP1D.cload();

		// sequential case first. This one is easier as it simply discards input iterator elements
		// whenever they are not local
		if( mode == SEQUENTIAL ) {
#ifdef _DEBUG
			std::cout << "buildVector< BSP1D > called, index + value "
						 "iterators, SEQUENTIAL mode\n";
#endif

			// sequential mode is not performant anyway, so let us just rely on the reference
			// implementation of the buildVector routine for Vector< InputType, reference >.
			std::vector< InputType > value_cache;
			std::vector< typename std::iterator_traits< fwd_iterator1 >::value_type > index_cache;
			const size_t n = grb::size( x );

			// loop over all input
			for( ; ind_start != ind_end && val_start != val_end; ++ind_start, ++val_start ) {

				// sanity check on input
				if( *ind_start >= n ) {
#ifdef _DEBUG
					std::cout << "\t mismatch detected, returning\n";
#endif
					return MISMATCH;
				}

				// check if this element is distributed to me
				if( internal::Distribution< BSP1D >::global_index_to_process_id(
						*ind_start, n, data.P
					) == data.s
				) {
					// yes, so cache
					const size_t localIndex =
						internal::Distribution< BSP1D >::global_index_to_local(
							*ind_start, n, data.P
						);
					index_cache.push_back( localIndex );
					value_cache.push_back( static_cast< InputType >( *val_start ) );
#ifdef _DEBUG
					std::cout << "\t local nonzero will be added to " << localIndex << ", "
						<< "value " << ( *val_start ) << "\n";
#endif
				} else {
#ifdef _DEBUG
					std::cout << "\t remote nonzero at " << ( *ind_start ) << " will be skipped.\n";
#endif
				}
			}

			// do delegate
			auto ind_it = index_cache.cbegin();
			auto val_it = value_cache.cbegin();
			RC rc = buildVector< descr >(
				internal::getLocal( x ),
				ind_it, index_cache.cend(),
				val_it, value_cache.cend(),
				SEQUENTIAL,
				dup
			);

			if( data.P > 1 && (descr & descriptors::no_duplicates) ) {
#ifdef _DEBUG
				std::cout << "\t global exit check (2)\n";
#endif
				if( collectives< BSP1D >::allreduce(
						rc,
						grb::operators::any_or< grb::RC >()
					) != SUCCESS
				) {
					return PANIC;
				}
			}

			if( rc == SUCCESS ) {
				x._nnz_is_dirty = true;
				return x.updateNnz();
			} else {
				return rc;
			}
		}

		// now handle parallel IOMode
		assert( mode == PARALLEL );

		return PANIC;
	}

	/**
	 * \internal No implementation details.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename InputType, typename fwd_iterator
	>
	RC buildMatrixUnique(
		Matrix< InputType, BSP1D > &A,
		fwd_iterator start, const fwd_iterator end,
		const IOMode mode
	) {
		typedef typename fwd_iterator::row_coordinate_type IType;
		typedef typename fwd_iterator::column_coordinate_type JType;
		typedef typename fwd_iterator::nonzero_value_type VType;

		// static checks
		NO_CAST_ASSERT( !( descr & descriptors::no_casting ) || (
				std::is_same< InputType, VType >::value &&
				std::is_integral< IType >::value &&
				std::is_integral< JType >::value
			), "grb::buildMatrixUnique (BSP1D implementation)",
			"Input iterator does not match output vector type while no_casting "
			"descriptor was set" );

		// get access to user process data on s and P
		internal::BSP1D_Data & data = internal::grb_BSP1D.load();

#ifdef _DEBUG
		std::cout << "buildMatrixUnique is called from process " << data.s << " "
			<< "out of " << data.P << " processes total.\n";
#endif

		// delegate for sequential case
		if( data.P == 1 ) {
			return buildMatrixUnique< descr >(
				internal::getLocal(A),
				start, end,
				SEQUENTIAL
			);
		}

		// function semantics require the matrix be cleared first
		RC ret = clear( A );

		// local cache, used to delegate to reference buildMatrixUnique
		std::vector< typename fwd_iterator::value_type > cache;

		// caches non-local nonzeroes (in case of Parallel IO)
		std::vector< std::vector< typename fwd_iterator::value_type > > outgoing;
		if( mode == PARALLEL ) {
			outgoing.resize( data.P );
		}
		// NOTE: this copies a lot of the above methodology

#ifdef _DEBUG
		const size_t my_offset =
			internal::Distribution< BSP1D >::local_offset( A._n, data.s, data.P );
		std::cout << "Local column-wise offset at PID " << data.s << " is "
			<< my_offset << "\n";
#endif
		// loop over all input
		for( size_t k = 0; ret == SUCCESS && start != end; ++k, ++start ) {

			// sanity check on input
			if( start.i() >= A._m ) {
				return MISMATCH;
			}
			if( start.j() >= A._n ) {
				return MISMATCH;
			}

			// compute process-local indices (even if remote, for code readability)
			const size_t global_row_index = start.i();
			const size_t row_pid =
				internal::Distribution< BSP1D >::global_index_to_process_id(
					global_row_index, A._m, data.P
				);
			const size_t row_local_index =
				internal::Distribution< BSP1D >::global_index_to_local(
					global_row_index, A._m, data.P
				);
			const size_t global_col_index = start.j();
			const size_t column_pid =
				internal::Distribution< BSP1D >::global_index_to_process_id(
					global_col_index, A._n, data.P
				);
			const size_t column_local_index =
				internal::Distribution< BSP1D >::global_index_to_local(
					global_col_index, A._n, data.P
				);
			const size_t column_offset =
				internal::Distribution< BSP1D >::local_offset(
					A._n, column_pid, data.P
				);

			// check if local
			if( row_pid == data.s ) {
				// push into cache
				cache.push_back( *start );
				// translate nonzero
				utils::updateNonzeroCoordinates(
					cache[ cache.size() - 1 ],
					row_local_index,
					column_offset + column_local_index
				);
#ifdef _DEBUG
				std::cout << "Translating nonzero at ( " << start.i() << ", " << start.j()
					<< " ) to one at ( " << row_local_index << ", "
					<< ( column_offset + column_local_index ) << " ) at PID "
					<< row_pid << "\n";
#endif
			} else if( mode == PARALLEL ) {
#ifdef _DEBUG
				std::cout << "Sending nonzero at ( " << start.i() << ", " << start.j()
					<< " ) to PID " << row_pid << " at ( " << row_local_index
					<< ", " << ( column_offset + column_local_index ) << " )\n";
#endif
				// send original nonzero to remote owner
				outgoing[ row_pid ].push_back( *start );
				// translate nonzero here instead of at
				// destination for brevity / code readibility
				utils::updateNonzeroCoordinates(
					outgoing[ row_pid ][ outgoing[ row_pid ].size() - 1 ],
					row_local_index, column_offset + column_local_index
				);
			} else {
#ifdef _DEBUG
				std::cout << "PID " << data.s << " ignores nonzero at ( "
					<< start.i() << ", " << start.j() << " )\n";
#endif
			}
		}

		// report on memory usage
		(void)config::MEMORY::report( "grb::buildMatrixUnique",
			"has local cache of size",
			cache.size() * sizeof( typename fwd_iterator::value_type )
		);

		if( mode == PARALLEL ) {
			// declare memory slots
			lpf_memslot_t cache_slot = LPF_INVALID_MEMSLOT;
			std::vector< lpf_memslot_t > out_slot( data.P, LPF_INVALID_MEMSLOT );

			// make sure we have enough space available to all-to-all #outgoing messages,
			// and enough space to do a prefix sum on those
			if( ret == SUCCESS ) {
				ret = data.checkBufferSize( 3 * data.P * sizeof( size_t ) );
			}

			// get handle directly into the BSP buffer, interpreted as size_t *
			size_t * const buffer_sizet = data.template getBuffer< size_t >();

			// make sure we support allgather/all-to-all patterns
			if( ret == SUCCESS ) {
				ret = data.ensureMaxMessages( 2 * data.P - 2 );
			}

			// make sure we have enough memslots available
			// cache_slot plus P-1 out_slots
			if( ret == SUCCESS ) {
				ret = data.ensureMemslotAvailable( data.P );
			}

			// send remote contribution counts
			size_t outgoing_bytes = 0;
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					// copy process-local data directly into destination area
					buffer_sizet[ data.P + k ] = 0;
					// sanity check
					assert( outgoing[ k ].size() == 0 );
					// done
					continue;
				}
				// cache size into buffer
				buffer_sizet[ k ] = outgoing[ k ].size();
				outgoing_bytes += outgoing[ k ].size() *
					sizeof( typename fwd_iterator::value_type );
#ifdef _DEBUG
				std::cout << "Process " << data.s << ", which has " << cache.size()
					<< " local nonzeroes, sends " << buffer_sizet[ k ]
					<< " nonzeroes to process " << k << "\n";
				std::cout << data.s << ": lpf_put( ctx, " << data.slot << ", "
					<< ( k * sizeof( size_t ) ) << ", " << k << ", " << data.slot << ", "
					<< ( data.P * sizeof( size_t ) + data.s * sizeof( size_t ) ) << ", "
					<< sizeof( size_t ) << ", LPF_MSG_DEFAULT );\n";
#endif
				// request RDMA
				const lpf_err_t brc = lpf_put( data.context,
					data.slot, k * sizeof( size_t ),
					k, data.slot, data.P * sizeof( size_t ) + data.s * sizeof( size_t ),
					sizeof( size_t ), LPF_MSG_DEFAULT
				);
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			(void)config::MEMORY::report( "grb::buildMatrixUnique (PARALLEL mode)",
				"has an outgoing cache of size", outgoing_bytes
			);
			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// do local prefix
			buffer_sizet[ 0 ] = cache.size();
			for( size_t k = 1; ret == SUCCESS && k < data.P; ++k ) {
				// no need to skip k == data.s as we set buffer_sizet[ data.P + data.s ] to 0
				buffer_sizet[ k ] = buffer_sizet[ k - 1 ] + buffer_sizet[ data.P + k - 1 ];
			}
			// self-prefix is not used, update to reflect total number of local elements
			// if data.s == data.P - 1 then the current number is already correct
			if( data.s + 1 < data.P ) {
				// otherwise overwrite with correct number
				buffer_sizet[ data.s ] =
					buffer_sizet[ data.P - 1 ] + buffer_sizet[ 2 * data.P - 1 ];
			}
			// communicate prefix
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				// but we do need to skip here or else we violate our max messages contract
				if( k == data.s ) {
					continue;
				}
#ifdef _DEBUG
				std::cout << "Process " << data.s << ", which has " << cache.size()
					<< " local nonzeroes, sends offset " << buffer_sizet[ k ]
					<< " to process " << k << "\n";
				std::cout << data.s << ": lpf_put( ctx, " << data.slot << ", "
					<< ( k * sizeof( size_t ) ) << ", " << k << ", " << data.slot << ", "
					<< ( 2 * data.P * sizeof( size_t ) + data.s * sizeof( size_t ) )
					<< ", " << sizeof( size_t ) << ", LPF_MSG_DEFAULT );\n";
#endif
				// send remote offsets
				const lpf_err_t brc = lpf_put( data.context,
					data.slot, k * sizeof( size_t ),
					k, data.slot, 2 * data.P * sizeof( size_t ) + data.s * sizeof( size_t ),
					sizeof( size_t ), LPF_MSG_DEFAULT
				);
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// register nonzero memory areas for all-to-all: global
			if( ret == SUCCESS ) {
				// enure local cache is large enough
				(void)config::MEMORY::report( "grb::buildMatrixUnique (PARALLEL mode)",
					"will increase local cache to size",
					buffer_sizet[ data.s ] * sizeof( typename fwd_iterator::value_type ) );
				cache.resize( buffer_sizet[ data.s ] ); // see self-prefix comment above
				// register memory slots for all-to-all
				const lpf_err_t brc = cache.size() > 0 ?
					lpf_register_global(
						data.context,
						&( cache[ 0 ] ), cache.size() *
							sizeof( typename fwd_iterator::value_type ),
						&cache_slot
					) :
                                                lpf_register_global(
							data.context,
							nullptr, 0,
							&cache_slot
						);
#ifdef _DEBUG
				std::cout << data.s << ": address " << &( cache[ 0 ] ) << " (size "
					<< cache.size() * sizeof( typename fwd_iterator::value_type )
					<< ") binds to slot " << cache_slot << "\n";
#endif
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// register nonzero memory areas for all-to-all: local
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					continue;
				}
				assert( out_slot.size() == data.P );
				const lpf_err_t brc = outgoing[ k ].size() > 0 ?
					lpf_register_local( data.context,
						&(outgoing[ k ][ 0 ]),
						outgoing[ k ].size() *
							sizeof( typename fwd_iterator::value_type ),
						&(out_slot[ k ])
					) :
					lpf_register_local( data.context,
						nullptr, 0,
						&(out_slot[ k ])
					);
#ifdef _DEBUG
				std::cout << data.s << ": address " << &( outgoing[ k ][ 0 ] ) << " (size "
					<< outgoing[ k ].size() * sizeof( typename fwd_iterator::value_type )
					<< ") binds to slot " << out_slot[ k ] << "\n";
#endif
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}

			// schedule all-to-all
			for( size_t k = 0; ret == SUCCESS && k < data.P; ++k ) {
				if( k == data.s ) {
					continue;
				}
#ifdef _DEBUG
				for( size_t s = 0; ret == SUCCESS && s < data.P; ++s ) {
					if( s == data.s ) {
						std::cout << data.s << ": lpf_put( ctx, "
							<< out_slot[ k ] << ", 0, " << k << ", " << cache_slot << ", "
							<< buffer_sizet[ 2 * data.P + k ] *
								sizeof( typename fwd_iterator::value_type )
							<< ", " << outgoing[ k ].size() *
								sizeof( typename fwd_iterator::value_type )
							<< ", LPF_MSG_DEFAULT );\n";
					}
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}
#endif
				if( outgoing[ k ].size() > 0 ) {
					const lpf_err_t brc = lpf_put( data.context,
							out_slot[ k ], 0,
							k, cache_slot, buffer_sizet[ 2 * data.P + k ] *
								sizeof( typename fwd_iterator::value_type ),
							outgoing[ k ].size() *
								sizeof( typename fwd_iterator::value_type ),
							LPF_MSG_DEFAULT
					);
					if( brc != LPF_SUCCESS ) {
						ret = PANIC;
					}
				}
			}
			// wait for RDMA to finish
			if( ret == SUCCESS ) {
				const lpf_err_t brc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				if( brc != LPF_SUCCESS ) {
					ret = PANIC;
				}
			}
			// clean up memslots, even on error (but still cause error when cleanup fails)
			for( size_t k = 0; k < data.P; ++k ) {
				if( out_slot[ k ] != LPF_INVALID_MEMSLOT ) {
					const lpf_err_t brc = lpf_deregister( data.context, out_slot[ k ] );
					if( brc != LPF_SUCCESS && ret == SUCCESS ) {
						ret = PANIC;
					}
				}
			}
			if( cache_slot != LPF_INVALID_MEMSLOT ) {
				const lpf_err_t brc = lpf_deregister( data.context, cache_slot );
				if( brc != LPF_SUCCESS && ret == SUCCESS ) {
					ret = PANIC;
				}
			}
			// clean up outgoing slots, which goes from 2x to 1x memory store for the
			// nonzeroes here contained
			{
				std::vector< std::vector< typename fwd_iterator::value_type > > emptyVector;
				std::swap( emptyVector, outgoing );
			}
		}

#ifdef _DEBUG
		std::cout << "Dimensions at PID " << data.s << ": "
			<< "( " << A._m << ", " << A._n << " ). "
			<< "Locally cached: " << cache.size() << "\n";
#endif

		if( ret == SUCCESS ) {
			// sanity check
			assert( nnz( A._local ) == 0 );
			// delegate and done!
			ret = buildMatrixUnique< descr >( A._local,
				utils::makeNonzeroIterator< IType, JType, VType >( cache.begin() ),
				utils::makeNonzeroIterator< IType, JType, VType >( cache.end() ),
				SEQUENTIAL
			);
			// sanity checks
			assert( ret != MISMATCH );
			assert( nnz( A._local ) == cache.size() );
		}

#ifdef _DEBUG
		std::cout << "Number of nonzeroes at the local matrix at PID " << data.s
			<< " is " << nnz( A._local ) << "\n";
#endif

		return ret;
	}

	template<>
	RC wait< BSP1D >();

	/** \internal Dispatch to base wait implementation */
	template<
		typename InputType, typename Coords,
		typename... Args
	>
	RC wait(
		const Vector< InputType, BSP1D, Coords > &x,
		const Args &... args
	) {
		(void) x;
		return wait( args... );
	}

	/** \internal Dispatch to base wait implementation */
	template< typename InputType, typename... Args >
	RC wait(
		const Matrix< InputType, BSP1D > &A,
		const Args &... args
	) {
		(void) A;
		return wait( args... );
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

#endif // end ``_H_GRB_BSP1D_IO''

