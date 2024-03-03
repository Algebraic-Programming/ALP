
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
 * @date 20th of January, 2017
 */

#ifndef _H_GRB_BSP1D_BLAS1
#define _H_GRB_BSP1D_BLAS1

#include <graphblas/blas0.hpp>
#include <graphblas/blas1.hpp>
#include <graphblas/bsp/collectives.hpp>
#include <graphblas/ops.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/type_traits.hpp>

#include "distribution.hpp"
#include "vector.hpp"

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
		"* Possible fix 2 | Provide a value that matches the expected type.\n"     \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );


namespace grb {

	/**
	 * \defgroup BLAS1_REF The Level-1 ALP/GraphBLAS routines -- BSP1D backend
	 *
	 * @{
	 */

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType, typename MaskType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, BSP1D, Coords > &x,
		const Vector< MaskType, BSP1D, Coords > &mask,
		IOType &beta,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::foldr",
			"called with a mask element type that is not Boolean" );

		// dynamic checks
		if( size( mask ) > 0 && size( mask ) != size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
			if( size( mask ) > 0 && nnz( mask ) < size( mask ) ) {
				return ILLEGAL;
			}
		}

		// cache local result
		IOType local = monoid.template getIdentity< IOType >();

		// do local foldr
		RC rc = foldr< descr >( internal::getLocal( x ), internal::getLocal( mask ),
			local, monoid );

		// do allreduce using \a op
		if( rc == SUCCESS ) {
			rc = collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );
		}

		// accumulate end result
		if( rc == SUCCESS ) {
			rc = foldr( local, beta, monoid.getOperator() );
		}

		// done
		return rc;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename InputType, typename IOType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, BSP1D, Coords > &x,
		IOType &beta,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		if( descr & descriptors::dense ) {
			if( nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
		}

		// cache local result
		IOType local = monoid.template getIdentity< IOType >();

		// do local foldr
		RC rc = foldr< descr >( internal::getLocal( x ), local, monoid );

		// do allreduce using \a op
		if( rc == SUCCESS ) {
			rc = collectives< BSP1D >::allreduce< descr >( local, monoid.getOperator() );
		}

		// accumulate end result
		if( rc == SUCCESS ) {
			rc = foldr( local, beta, monoid.getOperator() );
		}

		// done
		return rc;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename InputType, typename MaskType,
		typename Coords
	>
	RC foldl(
		IOType &alpha,
		const Vector< InputType, BSP1D, Coords > &y,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Monoid &monoid,
		const typename std::enable_if< !grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "foldl: IOType <- [InputType] with a monoid called. Array has "
			<< "size " << size( y ) << " with " << nnz( y ) << " nonzeroes. It has a "
			<< "mask of size " << size( mask ) << " with " << nnz( mask )
			<< " nonzeroes." << std::endl;
#endif
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::foldl",
			"called with a non-bool mask vector type while no_casting descriptor "
			"was set" );

		// dynamic sanity checks
		if( size( mask ) > 0 && size( mask ) != size( y ) ) {
			return MISMATCH;
		}
		if( size( y ) == 0 ) {
			return ILLEGAL;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) ) {
				return ILLEGAL;
			}
			if( size( mask ) > 0 && nnz( mask ) < size( mask ) ) {
				return ILLEGAL;
			}
		}

		// do local foldr
		IOType local = monoid.template getIdentity< IOType >();
		RC rc = foldl< descr >( local, internal::getLocal( y ),
			internal::getLocal( mask ), monoid );

#ifdef _DEBUG
		std::cout << "After process-local delegation, local value has become "
			<< local << ". Entering allreduce..." << std::endl;
#endif

		// do allreduce
		if( rc == SUCCESS ) {
			rc = collectives< BSP1D >::allreduce< descr >( local, monoid );
		}

		// accumulate end result
		if( rc == SUCCESS ) {
			rc = foldl( alpha, local, monoid.getOperator() );
		}

		// done
		return SUCCESS;
	}

	/**
	 * Folds a vector into a scalar.
	 *
	 * Unmasked variant.
	 *
	 * For performance semantics, see the masked variant of this primitive.
	 *
	 * \internal Dispatches to the masked variant, using an empty mask.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		IOType &x,
		const Vector< InputType, BSP1D, Coords > &y,
		const Monoid &monoid = Monoid(),
		const typename std::enable_if<
			!grb::is_object< IOType >::value &&
			!grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		Vector< bool, BSP1D, Coords > empty_mask( 0 );
		return foldl< descr >( x, y, empty_mask, monoid );
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename IOType, typename Coords, typename InputType
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Operator::D1 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D2 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the second domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		const size_t n = size( y );
		if( (descr & descriptors::dense) ) {
			if( nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// nonzero structure remains unchanged, so just dispatch
		RC ret = foldr< descr >( alpha, internal::getLocal( y ), op, phase );
		assert( ret == SUCCESS );
		if( ret != SUCCESS ) {
			ret = PANIC;
		}
		return ret;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename Coords, typename InputType
	>
	RC foldr(
		const InputType &alpha,
		Vector< IOType, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) ) {
				return ILLEGAL;
			}
		}

		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// simply delegate to reference implementation will yield correct result
		RC ret = foldr< descr >( alpha, internal::getLocal( y ), monoid, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				internal::setDense( y );
			} else if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
				ret == FAILED
			) {
				const RC subrc = internal::updateNnz( y );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, BSP1D, Coords > &x,
		Vector< IOType, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D2 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the second domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Operator::D1 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D3 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		constexpr bool dense_descr_is_given = (descr & descriptors::dense);
		const bool at_least_one_vector_is_sparse =
			nnz( x ) < size( x ) || nnz( y ) < size( y );
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}
		if( dense_descr_is_given ) {
			if( at_least_one_vector_is_sparse ) {
				return ILLEGAL;
			}
		} else if( at_least_one_vector_is_sparse ) {
			// this is a short-cut for having to do a lot of work only to find out
			// something is wrong. It does *not* preclude, however, needing to do
			// the final check near the end of this function
			if( nnz( x ) != nnz( y ) ) {
				return ILLEGAL;
			}
		}

		// in RESIZE phase, this operation does nothing as there cannot be fill-in
		if( phase == RESIZE ) { return SUCCESS; }

		// simply delegating will yield the correct result
		assert( phase == EXECUTE );
		RC ret = foldr< descr >( internal::getLocal( x ), internal::getLocal( y ),
			op, phase );

		// we do not need to sync nnz, as this method cannot generate fill-in

		// if x or y are sparse, however, this method could fail if there are nonzero
		// positions in either vector that are not populated in the other. A check is
		// necessary to catch such a violation -- but, gain, only in the sparse case.
		if( !dense_descr_is_given && at_least_one_vector_is_sparse ) {
			if( collectives< BSP1D >::allreduce( ret, grb::operators::any_or< RC >() )
				!= SUCCESS
			) {
				return PANIC;
			}
		}

		// done
		return ret;
	}

	/** No implementation ntoes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldr(
		const Vector< InputType, BSP1D, Coords > &x,
		Vector< IOType, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D2 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the second domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D1 >::value ), "grb::foldr",
			"called with an input vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldr",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		constexpr bool dense_descr_given = (descr & descriptors::dense);
		const size_t n = size( x );
		const bool dense_vectors_in = (nnz( x ) == n || nnz(y) == n);
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( dense_descr_given && !dense_vectors_in ) {
			return ILLEGAL;
		}

		// handle trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate non-trivial resize as well as any execute

		// option 1: dispatch to dense operator-variant (less overhead)
		if( dense_descr_given || dense_vectors_in ) {
			return foldr< descr >( x, y, monoid.getOperator(), phase );
		}

		// option 2: delegate to underlying backend
		RC ret = foldr< descr >( internal::getLocal( x ),
			internal::getLocal( y ), monoid, phase );

		// the only error that could occur is out-of-memory during a resize phase
		if( phase == RESIZE &&
			!config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities()
		) {
			if( collectives< BSP1D >::allreduce( ret, grb::operators::any_or< RC >() )
				!= SUCCESS
			) {
				return PANIC;
			}
			// indeed all other possible errors (MISMATCH, ILLEGAL) should already have
			// been caught on this backend's level. If they occur on the final backend,
			// there is a logic error on the BSP1D backend side somewhere.
			assert( ret == SUCCESS || ret == OUTOFMEM );
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				// in this case, the number of nonzeroes in the output vector may have
				// changed (recall that the dense case is not handled here)
				assert( !dense_vectors_in );
				ret = internal::updateNnz( y );
			} else if( ret == FAILED ) {
				// in this case, the full computation has not completed but the contents of
				// y do contain a subset of results. Therefore, the number of nonzeroes may
				// have changed, but we need to take care to still propagate FAILED
				assert( phase == TRY );
				const RC subrc = internal::updateNnz( y );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename IOType, typename Coords, typename InputType
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const InputType &beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Operator::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		if( descr & descriptors::dense ) {
			if( nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
		}

		// nonzero structure remains unchanged, so just dispatch
		RC ret = foldl< descr >( internal::getLocal( x ), beta, op, phase );
		assert( ret == SUCCESS );
		if( ret != SUCCESS ) {
			ret = PANIC;
		}
		return ret;
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename Coords, typename InputType
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const InputType &beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		if( descr & descriptors::dense ) {
			if( nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
		}

		// check for trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = foldl< descr >( internal::getLocal( x ), beta, monoid, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				internal::setDense( x );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( x );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename IOType, typename MaskType,
		typename Coords, typename InputType
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		Vector< MaskType, BSP1D, Coords > &mask,
		const InputType &beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Operator::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::foldl",
			"called with a mask value type that is not Boolean" );

		// check trivial dispatch
		if( size( mask ) == 0 ) {
			return foldl< descr >( x, beta, op, phase );
		}

		// dynamic checks
		const size_t n = size( x );
		if( size( mask ) != n ) {
			return MISMATCH;
		}
		if( (descr & descriptors::dense) ) {
			if( nnz( x ) < n ) {
				return ILLEGAL;
			}
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// nonzero structure remains unchanged, so just dispatch
		RC ret = foldl< descr >( internal::getLocal( x ), internal::getLocal( mask ),
			beta, op, phase );
		assert( ret == SUCCESS );
		if( ret != SUCCESS ) {
			ret = PANIC;
		}
		return ret;
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename MaskType,
		typename Coords, typename InputType
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		Vector< MaskType, BSP1D, Coords > &mask,
		const InputType &beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< InputType >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::foldl",
			"called with a mask value type that is not Boolean" );

		// check trivial dispatch
		if( size( mask ) == 0 ) {
			return foldl< descr >( x, beta, monoid, phase );
		}

		// dynamic checks
		const size_t n = size( x );
		if( size( mask ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n ) {
				return ILLEGAL;
			}
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// check for trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = foldl< descr >( internal::getLocal( x ), internal::getLocal( mask ),
			beta, monoid, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			assert( phase == EXECUTE || phase == TRY );
			if( ret == SUCCESS ) {
				if( nnz( mask ) == n &&
					(descr & descriptors::structural) &&
					!(descr & descriptors::invert_mask)
				) {
					internal::setDense( x );
				} else if( nnz( mask ) == 0 && (descr & descriptors::invert_mask) ) {
					internal::setDense( x );
				} else {
					const RC subrc = internal::updateNnz( x );
					if( subrc != SUCCESS ) { ret = PANIC; }
				}
			} else if( ret == FAILED ) {
				assert( phase == TRY );
				const RC subrc = internal::updateNnz( x );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/**
	 * \internal Number of nonzeroes in \a x cannot change, hence no
	 * synchronisation required.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename IOType, typename InputType, typename Coords
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const Vector< InputType, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Operator::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Operator::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		constexpr bool dense_descr_is_given = (descr & descriptors::dense);
		const bool at_least_one_vector_is_sparse =
			nnz( x ) < size( x ) || nnz( y ) < size( y );
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( dense_descr_is_given ) {
			if( at_least_one_vector_is_sparse ) {
				return ILLEGAL;
			}
		} else if( at_least_one_vector_is_sparse ) {
			// this is a short-cut for having to do a lot of work only to find out
			// something is wrong. It does *not* preclude, however, needing to do
			// the final check near the end of this function
			if( nnz( x ) != nnz( y ) ) {
				return ILLEGAL;
			}
		}

		// resize is a no-op since nnz of x cannot change
		if( phase == RESIZE ) { return SUCCESS; }

		// simply delegating will yield the correct result
		assert( phase == EXECUTE );
		RC ret = foldl< descr >( internal::getLocal( x ),
			internal::getLocal( y ), op, phase );

		// we do not need to sync nnz, as this method cannot generate fill-in

		// if x or y are sparse, however, this method could fail if there are nonzero
		// positions in either vector that are not populated in the other. A check is
		// necessary to catch such a violation -- but, gain, only in the sparse case.
		if( !dense_descr_is_given && at_least_one_vector_is_sparse ) {
			if( collectives< BSP1D >::allreduce( ret, grb::operators::any_or< RC >() )
				!= SUCCESS
			) {
				return PANIC;
			}
		}

		// done
		return ret;
	}

	/** \internal Requires synchronisation of output vector nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename IOType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const Vector< InputType, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D1 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the first domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType, typename Monoid::D2 >::value ), "grb::foldl",
			"called with an input vector value type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< IOType, typename Monoid::D3 >::value ), "grb::foldl",
			"called with an I/O value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		const size_t n = size( x );
		if( n != size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n || nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = SUCCESS;
		if( (descr | descriptors::dense) || ((nnz( x ) == n) && (nnz( y ) == n)) ) {
			// dense case will handle the remainder
			return foldl( x, y, monoid.getOperator(), phase );
		}

		// otherwise simply delegating will yield the correct result
		ret = foldl< descr >( internal::getLocal( x ), internal::getLocal( y ),
			monoid, phase );

		// the only error that could occur non-collectively is OOM during resize
		if( phase == RESIZE &&
			!config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities()
		) {
			if( collectives< BSP1D >::allreduce( ret, grb::operators::any_or< RC >() )
				!= SUCCESS
			) {
				return PANIC;
			}
			// indeed all other possible errors (MISMATCH, ILLEGAL) should already have
			// been caught on this backend's level. If they occur on the final backend,
			// there is a logic error on the BSP1D backend side somewhere.
			assert( ret == SUCCESS || ret == OUTOFMEM );
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				// x may have a new global number of nonzeroes that needs to be synced
				// (recall that the dense case is not handled here)
				ret = internal::updateNnz( x );
			} else if( ret == FAILED ) {
				// x may contain useful results that are a subset of the requested
				// computation. Therefore the nnz may have changed, but we should
				// take care to continue propagate FAILED
				const RC subrc = internal::updateNnz( x );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal No implementation notes */
	template<
		Descriptor descr = descriptors::no_operation, class OP,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType, BSP1D, Coords > &y,
		const OP &op = OP(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_operator< OP >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename OP::D1, IOType >::value ),
			"grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename OP::D2, InputType >::value ),
			"grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename OP::D3, IOType >::value ),
			"grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::foldl",
			"called with a mask that does not have boolean entries " );

		// catch empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, y, op, phase );
		}

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		// handle trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = foldl< descr >(
			internal::getLocal( x ), internal::getLocal( m ),
			internal::getLocal( y ),
			op, phase
		);

		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( x );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( x );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal No implementation notes */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename IOType, typename MaskType, typename InputType,
		typename Coords
	>
	RC foldl(
		Vector< IOType, BSP1D, Coords > &x,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType, BSP1D, Coords > &y,
		const Monoid &monoid = Monoid(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< grb::is_monoid< Monoid >::value &&
			!grb::is_object< IOType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType >::value, void
		>::type * = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D1, IOType >::value ),
			"grb::foldl",
			"called with a vector x of a type that does not match the first domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D2, InputType >::value ),
			"grb::foldl",
			"called on a vector y of a type that does not match the second domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Monoid::D3, IOType >::value ),
			"grb::foldl",
			"called on a vector x of a type that does not match the third domain "
			"of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::foldl",
			"called with a mask that does not have boolean entries" );

		// catch empty mask
		if( size( m ) == 0 ) {
			return foldl< descr >( x, y, monoid, phase );
		}

		// dynamic sanity checks
		const size_t n = size( x );
		if( n != size( y ) || n != size( m ) ) {
			return MISMATCH;
		}

		// handle trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = foldl< descr >(
			internal::getLocal( x ), internal::getLocal( m ),
			internal::getLocal( y ),
			monoid, phase
		);

		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( x );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( x );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal No communication necessary, output is guaranteed dense. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
			"[T1]<-T2<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input scalar type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input scalar type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		const size_t n = size( z );
		if( (descr & descriptors::dense) && nnz( z ) != n ) {
			return ILLEGAL;
		}
		if( capacity( z ) < n && phase == EXECUTE ) {
			return FAILED;
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate to set
		OutputType temp;
		RC ret = apply< descr >( temp, alpha, beta, op );
		ret = ret ? ret : set< descr >( z, temp, phase );

		// done
		return ret;
	}

	/** \internal Delegates to masked set. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
			"[T1]<-T2<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input scalar type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input scalar type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask value type that is not bool" );

		// dynamic checks
		const size_t n = size( z );
		if( (descr & descriptors::dense) && nnz( mask ) != n ) {
			return ILLEGAL;
		}
		if( (descr & descriptors::dense) && nnz( z ) != n ) {
			return ILLEGAL;
		}
		if( size( mask ) != n ) {
			return MISMATCH;
		}
		if( capacity( z ) < n && phase == EXECUTE ) {
			return FAILED;
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate to set
		OutputType temp;
		RC ret = apply< descr >( temp, alpha, beta, op );
		ret = ret ? ret : set< descr >( z, mask, temp, phase );

		// done
		return ret;
	}

	/** \internal No communication necessary, output is guaranteed dense. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
			"[T1]<-T2<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input scalar type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input scalar type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );

		// dynamic checks
		const size_t n = size( z );
		if( (descr & descriptors::dense) && nnz( z ) != n ) {
			return ILLEGAL;
		}
		if( capacity( z ) < n && phase == EXECUTE ) {
			return FAILED;
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate to set
		OutputType temp;
		RC ret = apply< descr >( temp, alpha, beta, monoid.getOperator() );
		ret = ret ? ret : set< descr >( z, temp, phase );

		// done
		return ret;
	}

	/** \internal Delegates to masked set. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const InputType1 alpha,
		const InputType2 beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
			"[T1]<-T2<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input scalar type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input scalar type that does not match the second "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ), "grb::eWiseApply",
			"called with an output mask value type that is not bool" );

		// dynamic checks
		const size_t n = size( z );
		if( (descr & descriptors::dense) && nnz( mask ) != n ) {
			return ILLEGAL;
		}
		if( (descr & descriptors::dense) && nnz( z ) != n ) {
			return ILLEGAL;
		}
		if( size( mask ) != n ) {
			return MISMATCH;
		}
		if( capacity( z ) < n && phase == EXECUTE ) {
			return FAILED;
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate to set
		OutputType temp;
		RC ret = apply< descr >( temp, alpha, beta, monoid.getOperator() );
		ret = ret ? ret : set< descr >( z, mask, temp, phase );

		// done
		return ret;
	}

	/** \internal No communication necessary, output is guaranteed dense. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
			"[T1]<-[T2]<-T3\n";
#endif

		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
			return ILLEGAL;
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ),
			internal::getLocal( x ), beta, op, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase == TRY ) {
			if( ret == SUCCESS || ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) {
					if( ret == SUCCESS ) { ret = subrc; }
					else { ret = PANIC; }
				}
			}
		} else if( phase == EXECUTE ) {
			if( ret == SUCCESS ) {
				internal::setDense( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal No communication necessary, output is guaranteed dense. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), T1]<-T2<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		const size_t n = size( z );
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( nnz( y ) < n ) {
			// note: this is illegal regardless of whether the dense descriptor was given
			return ILLEGAL;
		}

		// check for trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ), alpha,
			internal::getLocal( y ), op, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase == TRY ) {
			if( ret == SUCCESS || ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) {
					if( ret == SUCCESS ) { ret = subrc; }
					else { ret = PANIC; }
				}
			}
		} else if( phase == EXECUTE ) {
			if( ret == SUCCESS ) {
				internal::setDense( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/**
	 * \internal Requires communication in case of sparse vectors.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );

		// dynamic checks
		const size_t n = size( z );
		if( size( x ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails because size(x) "
				<< "!= size(z) -- " << size( x ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( size( y ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails because size(y) "
				<< "!= size(z) -- " << size( y ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n ) {
#ifdef _DEBUG
				std::cerr << "Warning: call to z = x + y (eWiseApply) fails because x is "
					<< "sparse -- nnz(x) = " << nnz( x ) << " -- while the dense "
					<< "descriptor was given\n";
#endif
				return ILLEGAL;
			}
			if( nnz( y ) < n ) {
#ifdef _DEBUG
				std::cerr << "Warning: call to z = x + y (eWiseApply) fails because y is "
					<< "sparse -- nnz(y) = " << nnz( y ) << " -- while the dense "
					<< "descriptor was given\n";
#endif
				return ILLEGAL;
			}
		}

		// catch trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ),
			internal::getLocal( x ), internal::getLocal( y ), op, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// catch execute
		if( phase != RESIZE ) {
			assert( phase == EXECUTE );
			if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			} else if( ret == SUCCESS ) {
				if( !(descr & descriptors::dense) ) {
					ret = internal::updateNnz( z );
				} else {
					internal::setDense( z );
				}
			}
		}

		// done
		return ret;
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not bool" );

		// check dispatch to simpler variant
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, alpha, y, op, phase );
		}

		// dynamic checks
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( nnz( y ) < n ) {
			// note that this is illegal regardless of whether dense descriptor is used
			return ILLEGAL;
		}
		if( descr & descriptors::dense ) {
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ),
			internal::getLocal( mask ), alpha, internal::getLocal( y ), op, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to sync nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename MaskType, typename InputType1,
		typename InputType2, typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not bool" );

		// check dispatch to simpler variant
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, beta, op, phase );
		}

		// dynamic checks
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( nnz( x ) < n ) {
			// note this is illegal regardless of whether a dense descriptor is passed
			return ILLEGAL;
		}
		if( descr & descriptors::dense ) {
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ),
			internal::getLocal( mask ), internal::getLocal( x ), beta, op, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to update global nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Operator,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Operator &op,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_operator< Operator >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (operator-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Operator::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Operator::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the second "
			"domain of the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Operator::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given operator" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not bool" );


		// check if we can dispatch to simpler variant
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, y, op );
		}

		// dynamic checks
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n ) {
				return ILLEGAL;
			}
			if( nnz( y ) < n ) {
				return ILLEGAL;
			}
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// check for trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			(descr & descriptors::dense) &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >(
			internal::getLocal( z ), internal::getLocal( mask ),
			internal::getLocal( x ), internal::getLocal( y ),
			op, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			}
			if( ret == FAILED ) {
				const RC update_rc = internal::updateNnz( z );
				if( update_rc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Does not require communication. */
	template<
		Descriptor descr = descriptors::no_operation, class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with an output vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input scalar type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );

		// check if can delegate to dense variant
		const size_t n = size( z );
		if( (descr & descriptors::dense) || (
			nnz( x ) == n && nnz( z ) == n
		) ) {
			return eWiseApply< descr | descriptors::dense >(
				z, x, beta, monoid.getOperator(), phase
			);
		}

		// dynamic checks
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n || nnz( z ) < n ) {
				return ILLEGAL;
			}
		}

		// handle for trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ),
			internal::getLocal( x ), beta, monoid, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute phase
		if( phase != RESIZE ) {
			assert( phase == EXECUTE );
			if( ret == SUCCESS ) {
				internal::setDense( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}
		return ret;
	}

	/** \internal Does not require communication. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
			"[T1]<-T2<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input scalar type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );

		// check if can delegate to dense variant
		const size_t n = size( z );
		if( (descr & descriptors::dense) || (
			nnz( y ) == n && nnz( z ) == n
		) ) {
			return eWiseApply< descr | descriptors::dense >(
				z, alpha, y, monoid.getOperator(), phase
			);
		}

		// dynamic checks
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < n || nnz( z ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >( internal::getLocal( z ), alpha,
			internal::getLocal( y ), monoid, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute
		if( phase != RESIZE ) {
			assert( phase == EXECUTE );
			if( ret == SUCCESS ) {
				internal::setDense( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to sync global nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D unmasked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );

		// check if we can delegate to dense variant
		const size_t n = size( z );
		if( (descr & descriptors::dense) || (
			nnz( x ) == n && nnz( y ) == n && nnz( z ) == n
		) ) {
			return eWiseApply< descr | descriptors::dense >(
				z, x, y, monoid.getOperator(), phase
			);
		}

		// run-time checks
		if( size( x ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
				"because size(x) != size(z) -- " << size( x ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}
		if( size( y ) != n ) {
#ifdef _DEBUG
			std::cerr << "Warning: call to z = x + y (eWiseApply) fails "
				"because size(y) != size(z) -- " << size( y ) << " != " << n << "\n";
#endif
			return MISMATCH;
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >(
			internal::getLocal( z ), internal::getLocal( x ), internal::getLocal( y ),
			monoid, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to sync global nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value,
		void >::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-T2<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not a bool" );

		// check dispatch to unmasked variant
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, alpha, y, monoid, phase );
		}

		// dynamic checks
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < n || nnz( mask ) < n || nnz( z ) < n ) {
				return ILLEGAL;
			}
		}

		// check trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >(
			internal::getLocal( z ), internal::getLocal( mask ),
			alpha, internal::getLocal( y ),
			monoid, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute and try phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to sync global nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-T3\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand input vector value type that does not match the "
			"first domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not a bool" );

		// check delegate to unmaked variant
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, beta, monoid, phase );
		}

		// dynamic checks
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
			if( nnz( x ) < n ) {
				return ILLEGAL;
			}
			if( nnz ( z ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >(
			internal::getLocal( z ), internal::getLocal( mask ),
			internal::getLocal( x ), beta,
			monoid, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute and try phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires communication to sync global nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Monoid,
		typename OutputType, typename MaskType,
		typename InputType1, typename InputType2,
		typename Coords
	>
	RC eWiseApply(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &mask,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Monoid &monoid,
		const Phase &phase = EXECUTE,
		const typename std::enable_if<
			!grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_monoid< Monoid >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cerr << "In BSP1D masked eWiseApply (monoid-based), "
					 "[T1]<-[T2]<-[T3]\n";
#endif
		// static checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType1, typename Monoid::D1 >::value ),
			"grb::eWiseApply",
			"called with a left-hand vector value type that does not match the first "
			"domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< InputType2, typename Monoid::D2 >::value ),
			"grb::eWiseApply",
			"called with a right-hand input vector value type that does not match the "
			"second domain of the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< OutputType, typename Monoid::D3 >::value ),
			"grb::eWiseApply",
			"called with an output value type that does not match the third domain of "
			"the given monoid" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< MaskType, bool >::value ), "grb::eWiseApply",
			"called with a mask value type that is not a bool" );

		// delegate to unmasked if possible
		const size_t n = size( mask );
		if( n == 0 ) {
			return eWiseApply< descr >( z, x, y, monoid, phase );
		}

		// dynamic checks
		if( size( x ) != n ) {
			return MISMATCH;
		}
		if( size( y ) != n ) {
			return MISMATCH;
		}
		if( size( z ) != n ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( x ) < n || nnz( y ) < n || nnz( z ) < n ) {
				return ILLEGAL;
			}
			if( nnz( mask ) < n ) {
				return ILLEGAL;
			}
		}

		// check for trivial resize phase
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseApply< descr >(
			internal::getLocal( z ), internal::getLocal( mask ),
			internal::getLocal( x ), internal::getLocal( y ),
			monoid, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/**
	 * \internal Requires communication to sync global nonzero count.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &a,
		const Vector< InputType2, BSP1D, Coords > &x,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, n );
		}
		assert( phase == EXECUTE );

		const bool sparse = grb::nnz( a ) != n ||
			grb::nnz( x ) != n ||
			grb::nnz( y ) != n;
		if( !sparse ) {
			internal::setDense( z );
			return grb::eWiseMulAdd< descr >(
				internal::getLocal( z ),
				internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ),
				ring
			);
		}
		const RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ),
			internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ),
			ring
		);
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This primitive has been deprecated since version 0.5. Please update
	 *          any use of this operation to an equivalent one using a sequence of
	 *          folds using the additive monoid if \a z is used in-place, or in the
	 *          case of out-of-place use of \a z by a call to grb::eWiseApply using
	 *          the additive monoid.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 &alpha,
		const Vector< InputType2, BSP1D, Coords > &x,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseAdd< descr >( internal::getLocal( z ), alpha,
			internal::getLocal( x ), ring, phase );
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 &alpha,
		const Vector< InputType2, BSP1D, Coords > &x,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha,
			internal::getLocal( x ), internal::getLocal( y ), ring, phase );
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ),
			internal::getLocal( a ), chi, internal::getLocal( y ), ring, phase );
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename Coords
	>
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &a,
		const Vector< InputType2, BSP1D, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, n );
		}

		assert( phase == EXECUTE );
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >(
			internal::getLocal( z ),
			internal::getLocal( a ), internal::getLocal( x ), gamma,
			ring
		);
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, n );
		}

		assert( phase == EXECUTE );
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >(
			internal::getLocal( z ),
			internal::getLocal( a ), beta, gamma,
			ring
		);
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, n );
		}

		assert( phase == EXECUTE );
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >(
			internal::getLocal( z ),
			alpha, internal::getLocal( x ), gamma,
			ring
		);
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, BSP1D, Coords > & y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, n );
		}

		assert( phase == EXECUTE );
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), alpha, beta, internal::getLocal( y ), ring
		);
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename Coords
	>
	RC eWiseMulAdd( Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value,
		void >::type * const = nullptr
	) {
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}
		if( phase == RESIZE ) {
			return resize( z, size( z ) );
		}
		assert( phase == EXECUTE );
		internal::setDense( z );
		return grb::eWiseMulAdd< descr >( internal::getLocal( z ), alpha, beta,
			gamma, ring );
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n || nnz( x ) < n || nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >(
			internal::getLocal( z ),
			internal::getLocal( x ), internal::getLocal( y ),
			ring, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = grb::size( z );
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n || nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >( internal::getLocal( z ), alpha,
			internal::getLocal( y ), ring, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute and try phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >( internal::getLocal( z ),
			internal::getLocal( x ), beta, ring, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = FAILED; }
			}
		}

		// done
		return ret;
	}

	/** \internal no implementation details */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );

		// dynamic checks
		const size_t n = grb::size( z );
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >( internal::getLocal( z ),
			alpha, beta, ring, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				internal::setDense( z );
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, typename MaskType,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D1, InputType1 >::value ), "grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D2, InputType2 >::value ), "grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< typename Ring::D3, OutputType >::value ), "grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
			std::is_same< bool, MaskType >::value ), "grb::eWiseMulAdd",
			"called with a mask vector with a non-bool element type" );

		// dynamic checks
		const size_t n = grb::size( z );
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n || nnz( m ) < n || nnz( x ) < n || nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( x ), internal::getLocal( y ),
			ring, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, typename MaskType,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, alpha, y, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( n != size( m ) || n != size( y ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n || nnz( y ) < n ) {
				return ILLEGAL;
			}
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			alpha, internal::getLocal( y ),
			ring, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle execute and try phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = PANIC; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, typename MaskType,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &x,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, x, beta, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( n != size( m ) || n != size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
			if( nnz( x ) < n ) { return ILLEGAL; }
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( x ), beta,
			ring, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = FAILED; }
			}
		}

		// done
		return ret;
	}

	/** \internal Requires syncing of output nonzero count. */
	template<
		Descriptor descr = descriptors::no_operation,
		class Ring, typename MaskType,
		typename InputType1, typename InputType2, typename OutputType,
		typename Coords
	>
	RC eWiseMul(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		// static sanity checks
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D1, InputType1 >::value ),
			"grb::eWiseMul",
			"called with a left-hand side input vector with element type that does not "
			"match the first domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D2, InputType2 >::value ),
			"grb::eWiseMul",
			"called with a right-hand side input vector with element type that does "
			"not match the second domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< typename Ring::D3, OutputType >::value ),
			"grb::eWiseMul",
			"called with an output vector with element type that does not match the "
			"third domain of the given semiring" );
		NO_CAST_ASSERT( ( !(descr & descriptors::no_casting) ||
				std::is_same< bool, MaskType >::value ),
			"grb::eWiseMulAdd",
			"called with a mask vector _m with a non-bool element type" );

		// check empty mask
		if( size( m ) == 0 ) {
			return eWiseMul< descr >( z, alpha, beta, ring, phase );
		}

		// dynamic checks
		const size_t n = size( z );
		if( n != size( m ) ) { return MISMATCH; }
		if( descr & descriptors::dense ) {
			if( nnz( z ) < n ) { return ILLEGAL; }
			if( nnz( m ) < n ) { return ILLEGAL; }
		}

		// handle trivial resize
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() &&
			phase == RESIZE
		) {
			return SUCCESS;
		}

		// delegate
		RC ret = eWiseMul< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			alpha, beta,
			ring, phase
		);
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		// handle try and execute phases
		if( phase != RESIZE ) {
			if( ret == SUCCESS ) {
				ret = internal::updateNnz( z );
			} else if( ret == FAILED ) {
				const RC subrc = internal::updateNnz( z );
				if( subrc != SUCCESS ) { ret = FAILED; }
			}
		}

		// done
		return ret;
	}

	/**
	 * \internal Requires communication to sync global nonzero count.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3, typename OutputType,
		typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &a,
		const Vector< InputType2, BSP1D, Coords > &x,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, a, x, y, ring, phase );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( a ), internal::getLocal( x ), internal::getLocal( y ),
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Does not require communication.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2,
		typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 &alpha,
		const Vector< InputType2, BSP1D, Coords > &x,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			alpha, internal::getLocal( x ), internal::getLocal( y ),
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType,
		typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &a,
		const InputType2 chi,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, a, chi, y, ring, phase );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( a ), chi, internal::getLocal( y ),
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &a,
		const Vector< InputType2, BSP1D, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, a, x, gamma, ring, phase );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		const RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( a ), internal::getLocal( x ), gamma,
			ring, phase
		);

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const Vector< InputType1, BSP1D, Coords > &a,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, a, beta, gamma, ring, phase );
		}
		if( n != grb::size( a ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			internal::getLocal( a ), beta, gamma,
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 alpha,
		const Vector< InputType2, BSP1D, Coords > &x,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, alpha, x, gamma, ring, phase );
		}
		if( n != grb::size( x ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			alpha, internal::getLocal( x ), gamma,
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const Vector< InputType3, BSP1D, Coords > &y,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, alpha, beta, y, ring, phase );
		}
		if( n != grb::size( y ) ) {
			return MISMATCH;
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >(
			internal::getLocal( z ), internal::getLocal( m ),
			alpha, beta, internal::getLocal( y ),
			ring, phase
		);
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal Requires synchronisation of global number of nonzeroes.
	 *
	 * \warning This function has been deprecated since version 0.5. If required,
	 *          consider instead a sequence of grb::foldl using the additive
	 *          monoid, followed by a call to grb::eWiseMul.
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename InputType1, typename InputType2, typename InputType3,
		typename OutputType, typename MaskType, typename Coords
	>
	RC eWiseMulAdd(
		Vector< OutputType, BSP1D, Coords > &z,
		const Vector< MaskType, BSP1D, Coords > &m,
		const InputType1 alpha,
		const InputType2 beta,
		const InputType3 gamma,
		const Ring &ring = Ring(),
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< OutputType >::value &&
			!grb::is_object< MaskType >::value &&
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< InputType3 >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
		const size_t n = grb::size( z );
		if( grb::size( m ) == 0 || (
			grb::nnz( m ) == n &&
			(descr & descriptors::structural) &&
			!(descr & descriptors::invert_mask)
		) ) {
			return eWiseMulAdd< descr >( z, alpha, beta, gamma, ring, phase );
		}
		if( n != grb::size( m ) ) {
			return MISMATCH;
		}
		if( config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() && phase == RESIZE ) {
			return SUCCESS;
		}

		RC ret = grb::eWiseMulAdd< descr >( internal::getLocal( z ),
			internal::getLocal( m ), alpha, beta, gamma, ring, phase );
		if( !config::IMPLEMENTATION< BSP1D >::fixedVectorCapacities() ) {
			if( collectives< BSP1D >::allreduce(
				ret, grb::operators::any_or< RC >()
			) != SUCCESS ) {
				return PANIC;
			}
		}

		if( ret == SUCCESS && phase != RESIZE ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/**
	 * \internal
	 *
	 * BSP1D implementation of the \f$ \alpha = xy \f$ operation;
	 * the dot-product.
	 *
	 * @tparam descr      The descriptor used. If left unspecified,
	 *                    grb::descriptors::no_operation is used.
	 * @tparam Ring       The semiring to be used.
	 * @tparam OutputType The output type.
	 * @tparam InputType1 The input element type of the left-hand input vector.
	 * @tparam InputType2 The input element type of the right-hand input vector.
	 *
	 * @param[out]  z  The output element \f$ \alpha \f$.
	 * @param[in]   x  The left-hand input vector.
	 * @param[in]   y  The right-hand input vector.
	 * @param[in] ring The semiring to perform the dot-product under. If left
	 *                 undefined, the default constructor of \a Ring will be used.
	 *
	 * @return grb::MISMATCH When the dimensions of \a x and \a y do not match. All
	 *                       input data containers are left untouched if this exit
	 *                       code is returned; it will be as though this call was
	 *                       never made.
	 * @return grb::SUCCESS  On successful completion of this call.
	 *
	 * \parblock
	 * \par Performance semantics
	 *      -# This call takes \f$ \Theta(n/p) \f$ work at each user process, where
	 *         \f$ n \f$ equals the size of the vectors \a x and \a y, and
	 *         \f$ p \f$ is the number of user processes. The constant factor
	 *         depends on the cost of evaluating the addition and multiplication
	 *         operators. A good implementation uses vectorised instructions
	 *         whenever the input domains, output domain, and the operators used
	 *         allow for this.
	 *
	 *      -# This call takes \f$ \mathcal{O}(1) \f$ memory beyond the memory used
	 *         by the application at the point of a call to this function.
	 *
	 *      -# This call incurs at most
	 *         \f$ n( \mathit{sizeof}(\mathit{D1}) + \mathit{sizeof}(\mathit{D2}) ) + \mathcal{O}(p) \f$
	 *         bytes of data movement.
	 *
	 *      -# This call incurs at most \f$ \Theta(\log p) \f$ synchronisations
	 *         between two or more user processes.
	 *
	 *      -# A call to this function does result in any system calls.
	 * \endparblock
	 *
	 * \parblock
	 * \par Valid descriptors
	 *   -# grb::descriptors::no_operation
	 *   -# grb::descriptors::no_casting
	 * \endparblock
	 *
	 * The vector distributions are block-cyclic and thus conforms to the work
	 * performance guarantee.
	 *
	 * This function performs a local dot product and then calls
	 * grb::collectives::allreduce(), and thus conforms to the bandwidth and
	 * synchornisation semantics defined above.
	 */
	template<
		Descriptor descr = grb::descriptors::no_operation,
		class AddMonoid, class AnyOp,
		typename OutputType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot(
		OutputType &z,
		const Vector< InputType1, BSP1D, Coords > &x,
		const Vector< InputType2, BSP1D, Coords > &y,
		const AddMonoid &addMonoid,
		const AnyOp &anyOp,
		const typename std::enable_if< !grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< OutputType >::value &&
			grb::is_monoid< AddMonoid >::value &&
			grb::is_operator< AnyOp >::value, void
		>::type * const = nullptr
	) {
		// sanity check
		if( size( y ) != size( x ) ) {
			return MISMATCH;
		}
		if( descr & descriptors::dense ) {
			if( nnz( y ) < size( y ) || nnz( x ) < size( x ) ) {
				return ILLEGAL;
			}
		}

		// get field for out-of-place dot
		OutputType oop = addMonoid.template getIdentity< OutputType >();

		// all OK, try to do assignment
		RC ret = grb::dot< descr >( oop,
			internal::getLocal( x ), internal::getLocal( y ),
			addMonoid, anyOp
		);
		ret = ret ? ret : collectives< BSP1D >::allreduce( oop, addMonoid );

		// fold out-of-place dot product into existing value and exit
		ret = ret ? ret : foldl( z, oop, addMonoid.getOperator() );
		return ret;
	}

	/**
	 * \internal
	 * Provides a generic implementation of the dot computation on semirings by
	 * translating it into a dot computation on an additive commutative monoid
	 * with any multiplicative operator.
	 * \endinternal
	 */
	template<
		Descriptor descr = descriptors::no_operation, class Ring,
		typename IOType, typename InputType1, typename InputType2,
		typename Coords
	>
	RC dot(
		IOType &x,
		const Vector< InputType1, BSP1D, Coords > &left,
		const Vector< InputType2, BSP1D, Coords > &right,
		const Ring &ring = Ring(),
		const typename std::enable_if<
			!grb::is_object< InputType1 >::value &&
			!grb::is_object< InputType2 >::value &&
			!grb::is_object< IOType >::value &&
			grb::is_semiring< Ring >::value, void
		>::type * const = nullptr
	) {
#ifdef _DEBUG
		std::cout << "In grb::dot (BSP1D, semiring version)\n"
			<< "\t dispatches to monoid-operator version\n";
#endif
		return grb::dot< descr >( x,
			left, right,
			ring.getAdditiveMonoid(),
			ring.getMultiplicativeOperator()
		);
	}

	/** \internal No implementation notes. */
	template< typename Func, typename DataType, typename Coords >
	RC eWiseMap( const Func f, const Vector< DataType, BSP1D, Coords > &x ) {
		return eWiseMap( f, internal::getLocal( x ) );
	}

	/**
	 * \internal
	 * We can simply delegates to the reference implementation because all vectors
	 * are distributed equally in this reference implementation. Length checking is
	 * also distributed which is correct, since all calls are collective there may
	 * never be a mismatch in globally known vector sizes.
	 */
	template<
		Descriptor descr = descriptors::no_operation,
		typename Func, typename DataType, typename Coords
	>
	RC eWiseLambda( const Func f, const Vector< DataType, BSP1D, Coords > &x ) {
		const internal::BSP1D_Data &data = internal::grb_BSP1D.cload();
		// rely on local lambda, passing in the active global distribution, global
		// length, and number of user processes
		return internal::eWiseLambda<
			descr,
			typename internal::Distribution< BSP1D >
		>( f, internal::getLocal( x ), x._n, data.s, data.P );
		// note the sparsity structure will not change by the above call
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename Func,
		typename DataType1, typename DataType2, typename Coords,
		typename... Args
	>
	RC eWiseLambda(
		const Func f,
		const Vector< DataType1, BSP1D, Coords > &x,
		const Vector< DataType2, BSP1D, Coords > &y,
		Args const &... args
	) {
		// check dimension mismatch
		if( size( x ) != size( y ) ) {
			return MISMATCH;
		}
		// in this implementation, the distributions are equal so no need for any
		// synchronisation
		return eWiseLambda< descr >( f, x, args... );
		// note the sparsity structure will not change by the above call
	}

	/** \internal No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC zip(
		Vector< std::pair< T, U >, BSP1D, Coords > &z,
		const Vector< T, BSP1D, Coords > &x,
		const Vector< U, BSP1D, Coords > &y,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< T >::value &&
			!grb::is_object< U >::value, void
		>::type * const = nullptr
	) {
		const size_t n = size( z );
		if( size( x ) != n || n != size( y ) ) {
			return MISMATCH;
		}
		if( nnz( x ) != nnz( y ) ) {
			return ILLEGAL;
		}
		if( phase == RESIZE ) {
			return resize( z, nnz( x ) );
		}

		assert( phase == EXECUTE );
		const RC ret = zip( internal::getLocal( z ),
			internal::getLocal( x ), internal::getLocal( y ) );
		if( ret == SUCCESS ) {
			return internal::updateNnz( z );
		} else {
			return ret;
		}
	}

	/** No implementation notes. */
	template<
		Descriptor descr = descriptors::no_operation,
		typename T, typename U, typename Coords
	>
	RC unzip(
		Vector< T, BSP1D, Coords > &x,
		Vector< U, BSP1D, Coords > &y,
		const Vector< std::pair< T, U >, BSP1D, Coords > &in,
		const Phase &phase = EXECUTE,
		const typename std::enable_if< !grb::is_object< T >::value &&
			!grb::is_object< U >::value, void
		>::type * const = nullptr
	) {
		const size_t n = size( in );
		if( size( x ) != n && n != size( y ) ) {
			return MISMATCH;
		}
		RC ret = SUCCESS;

		if( phase == RESIZE ) {
			const size_t target = nnz( in );
			ret = resize( x, target );
			if( ret == SUCCESS ) {
				ret = resize( y, target );
			}
			if( ret != SUCCESS ) {
				ret = clear( x );
				ret = ret ? ret : clear( y );
				if( ret != SUCCESS ) {
					ret = PANIC;
				}
			}
			return ret;
		}

		assert( phase == EXECUTE );
		ret = unzip(
			internal::getLocal( x ), internal::getLocal( y ),
			internal::getLocal( in )
		);
		if( ret == SUCCESS ) {
			ret = internal::updateNnz( x );
		}
		if( ret == SUCCESS ) {
			ret = internal::updateNnz( y );
		}
		return ret;
	}

	/** @} */

} // namespace grb

#undef NO_CAST_ASSERT

#endif

