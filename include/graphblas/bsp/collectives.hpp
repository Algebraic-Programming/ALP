
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
 * @author A. N. Yzelman & J. M. Nash
 * @date 20th of February, 2017
 */

#ifndef _H_GRB_BSP_COLL
#define _H_GRB_BSP_COLL

#include <type_traits>

#include <assert.h>
#include <string.h>

#include <graphblas/ops.hpp>
#include <graphblas/blas0.hpp>
#include <graphblas/backends.hpp>

#include <graphblas/base/collectives.hpp>

#include <graphblas/bsp/error.hpp>

#include <graphblas/bsp1d/init.hpp>

#define NO_CAST_ASSERT_BLAS0( x, y, z )                                            \
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
		"* Possible fix 2 | Provide a value of the same type as the first domain " \
		"of the given operator.\n"                                                 \
		"* Possible fix 3 | Ensure the operator given to this call to " y " has "  \
		"all "                                                                     \
		"of "                                                                      \
		"its "                                                                     \
		"domain"                                                                   \
		"s "                                                                       \
		"equal "                                                                   \
		"to "                                                                      \
		"each "                                                                    \
		"other."                                                                   \
		"\n"                                                                       \
		"************************************************************************" \
		"************************************************************************" \
		"**********************\n" );

#ifdef _DEBUG
 #define _DEBUG_BSP_COLLECTIVES
#endif


namespace grb {

	namespace {

		/**
		 * This is a reducer function of the signature specified by lpf_reducer_t.
		 *
		 * @tparam OP    The operator used for reduction
		 * @tparam foldl Whether the originating call is foldl (<tt>true</tt>), or
		 *               is foldr instead (<tt>false</tt>)
		 */
		template< typename Operator, bool foldl >
		void generic_reducer( size_t n, const void * array_p, void * value_p ) {
			typedef typename Operator::D1 lhs_type;
			typedef typename Operator::D2 rhs_type;
			typedef typename Operator::D3 out_type;
			static_assert( is_associative< Operator >::value,
				"A generic reducer requires an associative operator. This is an internal "
				"error. Please submit a bug report." );
			static_assert(
					foldl || std::is_same< rhs_type, out_type >::value,
					"A generic reducer from an array into a scalar requires the monoid input "
					"domain corresponding to the scalar (i.e., the RHS domain for foldr) and "
					"its output type to be the same. This is an internal error. Please submit "
					"a bug report."
				);
			static_assert(
					(!foldl) || std::is_same< lhs_type, out_type >::value,
					"A generic reducer from an array into a scalar requires the monoid input "
					"domain corresponding to the scalar (i.e., the LHS domain for foldl) and "
					"its output type to be the same. This is an internal error. Please submit "
					"a bug report."
				);
			assert( array_p != value_p );

			typedef typename std::conditional< foldl, rhs_type, lhs_type >::type
				array_type;
			const array_type * const __restrict__ array =
				static_cast< const array_type * >( array_p );
			out_type * const __restrict__ value =
				static_cast< out_type * >( value_p );

			// SIMD loop
			size_t i = 0;
			array_type array_buffer[ Operator::blocksize ];
			while( i + Operator::blocksize < n ) {
				// load
				for( size_t k = 0; k < Operator::blocksize; ++k ) {
					array_buffer[ k ] = array[ i + k ];
				}
				// compute
				for( size_t k = 0; k < Operator::blocksize; ++k ) {
					if( foldl ) {
						Operator::foldl( *value, array_buffer[ k ] );
					} else {
						Operator::foldr( array_buffer[ k ], *value );
					}
				}
				// increment
				i += Operator::blocksize;
			}

			// scalar coda
			for( ; i < n; ++i ) {
				if( foldl ) {
					Operator::foldl( *value, array[ i ] );
				} else {
					Operator::foldr( array[ i ], *value );
				}
			}

			// done
		}

	} // end anon namespace

	/**
	 * Collective communications using ALP operators for reduce-style operations.
	 *
	 * This is the BSP1D implementation.
	 *
	 * TODO internal issue #198
	 */
	template<>
	class collectives< BSP1D > {

		private:

			/** Disallow instantiation of this class. */
			collectives() {}


		protected:

			/**
			 * @tparam use_id Whether called by monoid variants, which makes available
			 *                an identity of type Operator::D3
			 * @tparam all    Whether to perform all-reduction (<tt>true</tt>) or to
			 *                perform reduction (<tt>false</tt>).
			 *
			 * If \a all is <tt>true</tt> then \a root will not be used.
			 *
			 * \todo If bumping to C++17 or higher, then can use constexpr
			 *       string_view in the last arg
			 */
			template<
				Descriptor descr,
				typename Operator,
				typename IOType,
				bool use_id,
				bool all
			>
			static RC reduce_allreduce_generic(
				IOType &inout,
				const lpf_pid_t root,
				const Operator &op,
				const typename Operator::D3 * const id,
				internal::BSP1D_Data &data,
				const char * const source
			) {
				// static sanity checks
				static_assert( is_associative< Operator >::value,
					"Internal logic error: reduce_generic requires associative operators."
					"Please submit a bug report." );
				static_assert(
					(std::is_same< typename Operator::D1, typename Operator::D3 >::value) ||
					(std::is_same< typename Operator::D2, typename Operator::D3 >::value),
					"grb::collectives::internal::reduce_generic, "
					"in reduction, the given operator must have at least one of its input "
					"types equal to its output type. This is an internal error. Please "
					"submit a bug report."
				);
				static_assert(
					(std::is_same< typename Operator::D1, typename Operator::D2 >::value) ||
					use_id,
					"grb::collectives::internal::reduce_generic, "
					"if not all domains of the operator match, and identity must be given. "
					"This is an internal error. Please submit a bug report."
				); // i.e., if triggered, it is likely to mean somehow reduction with
				   // operators was called, while a monoid was necessary
#ifdef _DEBUG_BSP_COLLECTIVES
				{
					for( lpf_pid_t k = 0; k < data.P; ++k ) {
						if( k == data.s ) {
							std::cout << "\t " << k << ": called reduce_allreduce_generic\n";
						}
						lpf_sync( data.context, LPF_SYNC_DEFAULT );
					}
				}
#endif

				// dynamic sanity checks
				assert( all || root < data.P );
				assert( !use_id || id != nullptr );

				// catch trivial case early
				if( data.P == 1 ) {
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t trivial no-op case: P == 1\n";
#endif
					return SUCCESS;
				}

				// create reduction buffer based on the operator IO type
				typedef typename Operator::D3 OPIOT;
				// ensure the global buffer has enough capacity
				RC rc = data.ensureBufferSize( sizeof( OPIOT ) );
				// ensure we can execute the requested collective call
				rc = rc ? rc : data.ensureCollectivesCapacity( 1, sizeof( OPIOT ), 0 );
				if( all ) {
					rc = rc ? rc : data.ensureMaxMessages( 2 * data.P - 2 );
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t asked for hmax = " << (2*data.P-2) << "\n";
#endif
				} else {
					rc = rc ? rc : data.ensureMaxMessages( data.P - 1 );
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t asked for hmax = " << (data.P - 1) << "\n";
#endif
				}
				// exit on failed preconditions
				if( rc != SUCCESS ) {
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t could not reserve enough capacity: "
						<< grb::toString( rc ) << "\n";
#endif
					return rc;
				}

				// retrieve buffer area
				OPIOT * const __restrict__ buffer = data.template getBuffer< OPIOT >();

				// figure out which direction to reduce to
				constexpr bool left_looking =
					std::is_same< typename Operator::D1, typename Operator::D3 >::value;

				// copy payload into buffer
				// rationale: this saves one global registration, which otherwise is likely
				//            to dominate most uses for this collective call
				if( use_id ) {
					if( left_looking ) {
						(void) apply( *buffer, *id, inout, op );
					} else {
						(void) apply( *buffer, inout, *id, op );
					}
				} else {
					// no operator application necessary, they are the same type so we can
					// just copy
					*buffer = inout;
				}

				// get the lpf_reducer_t
				lpf_reducer_t reducer = &(generic_reducer< Operator, left_looking >);

				// schedule collective
				lpf_err_t lpf_rc = LPF_SUCCESS;
				if( all ) {
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t collectives< BSP >::reduce_allreduce_generic, calls "
						<< "lpf_allreduce with size " << sizeof(OPIOT) << std::endl;
#endif
					(void) root;
					lpf_rc = lpf_allreduce(
						data.coll,
						buffer, data.slot,
						sizeof(OPIOT),
						reducer
					);
				} else {
#ifdef _DEBUG_BSP_COLLECTIVES
					std::cout << "\t\t collectives< BSP >::reduce_allreduce_generic calls "
						<< "lpf_reduce with size " << sizeof(OPIOT) << std::endl;
#endif
					lpf_rc = lpf_reduce(
						data.coll,
						buffer, data.slot,
						sizeof(OPIOT),
						reducer,
						root
					);
				}

				// execute collective
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// catch LPF errors
				rc = internal::checkLPFerror( lpf_rc, source );

				// copy back
				if( all ) {
					if( rc == SUCCESS ) {
						inout = *buffer;
					}
				} else {
					if( rc == SUCCESS && data.s == static_cast< size_t >( root ) ) {
						inout = *buffer;
					}
				}

				// done
				return SUCCESS;
			}


		public:


			/**
			 * Schedules an allreduce operation of a single object of type IOType per
			 * process. The allreduce shall be complete by the end of the call. This is a
			 * collective graphBLAS operation.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Operator \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
			 * \endparblock
			 *
			 * This function may place an alloc of \f$ P\mathit{sizeof}(IOType) \f$ bytes
			 * if the internal buffer was not sufficiently large.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Operator, typename IOType
			>
			static RC allreduce(
				IOType &inout, const Operator &op = Operator(),
				const typename std::enable_if<
						grb::is_operator< Operator >::value,
					void >::type * const = nullptr
			) {
#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "Entered grb::collectives< BSP >::allreduce with inout = "
					<< inout << " (byte size " << sizeof(inout) << ") and op = " << &op
					<< std::endl;
#endif

				// static sanity checks
				static_assert( !grb::is_object< IOType >::value,
					"grb::collectives::allreduce cannot have another ALP object as its scalar "
					"type!" );
				static_assert( is_associative< Operator >::value,
					"grb::collectives::allreduce requires an associative operator" );
				NO_CAST_ASSERT_BLAS0( ( !(descr & descriptors::no_casting) ||
						std::is_same< IOType, typename Operator::D1 >::value ||
						std::is_same< IOType, typename Operator::D2 >::value ||
						std::is_same< IOType, typename Operator::D3 >::value
					),
					"grb::collectives::allreduce",
					"Incompatible given value type and monoid domains while the no_casting "
					"descriptor was set"
				);
				NO_CAST_ASSERT_BLAS0(
					(std::is_same< typename Operator::D1, typename Operator::D2 >::value) &&
					(std::is_same< typename Operator::D2, typename Operator::D3 >::value),
					"grb::collectives::allreduce",
					"In all-reduction, the given operator must have all of its domains equal "
					"to one another. If different domains are required, a monoid must be "
					"provided instead"
				);

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// dispatch
				const RC ret = reduce_allreduce_generic<
					descr, Operator, IOType, false, true
				>(
					inout, 0, op, nullptr, data,
					"grb::collectives< BSP >::allreduce (operator)"
				);
#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "\t\t returning inout = " << inout << "\n";
#endif
				return ret;
			}

			/**
			 * Schedules an allreduce operation of a single object of type IOType per
			 * process. The allreduce shall be complete by the end of the call. This is a
			 * collective graphBLAS operation.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Monoid \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Monoid + l \f$;
			 * \endparblock
			 *
			 * This function may place an alloc of \f$ P\mathit{sizeof}(IOType) \f$ bytes
			 * if the internal buffer was not sufficiently large.
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Monoid, typename IOType
			>
			static RC allreduce(
				IOType &inout, const Monoid &monoid = Monoid(),
				const typename std::enable_if<
					is_monoid< Monoid >::value,
				void >::type * const = nullptr
			) {
#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "Entered grb::collectives< BSP >::allreduce with inout = "
					<< inout << " (byte size " << sizeof(inout) << ") and monoid = " << &monoid
					<< std::endl;
#endif
				// static sanity checks
				static_assert( !grb::is_object< IOType >::value,
					"grb::collectives::allreduce cannot have another ALP object as its scalar "
					"type!" );
				NO_CAST_ASSERT_BLAS0( ( !(descr & descriptors::no_casting) ||
						std::is_same< IOType, typename Monoid::D1 >::value ||
						std::is_same< IOType, typename Monoid::D2 >::value ||
						std::is_same< IOType, typename Monoid::D3 >::value
					),
					"grb::collectives::allreduce",
					"Incompatible given value type and monoid domains while the no_casting "
					"descriptor was set"
				);

				// check whether the monoid has all its domains equal
				constexpr bool same_domains =
					std::is_same< typename Monoid::D1, typename Monoid::D2 >::value &&
					std::is_same< typename Monoid::D2, typename Monoid::D3 >::value;

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// get identity
				const typename Monoid::D3 id = monoid.template
					getIdentity< typename Monoid::D3 >();

				// dispatch
				const RC ret = reduce_allreduce_generic<
					descr, typename Monoid::Operator, IOType, !same_domains, true
				>(
					inout, 0, monoid.getOperator(), &id, data,
					"grb::collectives< BSP >::allreduce (monoid)"
				);

#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "\t\t returning inout = " << inout << "\n";
#endif
				return ret;
			}

			/**
			 * Schedules a reduce operation of a single object of type IOType per process.
			 * The reduce shall be complete by the end of the call. This is a collective
			 * graphBLAS operation. The BSP costs are as for the PlatformBSP #reduce.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Operator \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
			 * \endparblock
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Operator, typename IOType
			>
			static RC reduce(
				IOType &inout, const lpf_pid_t root = 0,
				const Operator &op = Operator(),
				const typename std::enable_if<
					grb::is_operator< Operator >::value,
				void >::type * const = nullptr
			) {
#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "Entered grb::collectives< BSP >::reduce with inout = "
					<< inout << " and op = " << &op << std::endl;
#endif
				// static sanity checks
				static_assert( !grb::is_object< IOType >::value,
					"grb::collectives::allreduce cannot have another ALP object as its scalar "
					"type!" );
				static_assert( is_associative< Operator >::value,
					"grb::collectives::reduce requires an associative operator" );
				NO_CAST_ASSERT_BLAS0( ( !(descr & descriptors::no_casting) ||
						std::is_same< IOType, typename Operator::D1 >::value ||
						std::is_same< IOType, typename Operator::D2 >::value ||
						std::is_same< IOType, typename Operator::D3 >::value
					), "grb::collectives::reduce",
					"Incompatible given value type and monoid domains while the no_casting"
					"descriptor was set"
				);
				NO_CAST_ASSERT_BLAS0(
					(std::is_same< typename Operator::D1, typename Operator::D2 >::value) &&
					(std::is_same< typename Operator::D2, typename Operator::D3 >::value),
					"grb::collectives::reduce",
					"In reduction, the given operator must have all of its domains equal to "
					"one another. If different domains are required, a monoid must be "
					"provided instead"
				);

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// dynamic checks
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// dispatch
				return reduce_allreduce_generic<
					descr, Operator, IOType, false, false
				>(
					inout, root, op, nullptr, data,
					"grb::collectives< BSP >::reduce (operator)"
				);
			}

			/**
			 * Schedules a reduce operation of a single object of type IOType per process.
			 * The reduce shall be complete by the end of the call. This is a collective
			 * GraphBLAS operation. The BSP costs are as for the LPF lpf_reduce.
			 *
			 * \parblock
			 * \par Performance semantics:
			 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ N*Monoid \f$ ;
			 * -# transferred bytes: \f$ N \f$ ;
			 * -# BSP cost: \f$ Ng + N*Monoid + l \f$;
			 * \endparblock
			 */
			template<
				Descriptor descr = descriptors::no_operation,
				typename Monoid, typename IOType
			>
			static RC reduce(
				IOType &inout, const lpf_pid_t root = 0,
				const Monoid monoid = Monoid(),
				const typename std::enable_if<
					is_monoid< Monoid >::value,
				void >::type * const = nullptr
			) {
#ifdef _DEBUG_BSP_COLLECTIVES
				std::cout << "Entered grb::collectives< BSP >::reduce with inout = "
					<< inout << " (byte size " << sizeof(IOType) << ") and monoid = "
					<< &monoid << std::endl;
#endif
				// static sanity checks
				static_assert( !grb::is_object< IOType >::value,
					"grb::collectives::allreduce cannot have another ALP object as its scalar "
					"type!" );
				static_assert( is_monoid< Monoid >::value,
					"grb::collectives::reduce requires a monoid" );
				NO_CAST_ASSERT_BLAS0( ( !(descr & descriptors::no_casting) ||
						std::is_same< IOType, typename Monoid::D1 >::value ||
						std::is_same< IOType, typename Monoid::D2 >::value ||
						std::is_same< IOType, typename Monoid::D3 >::value
					), "grb::collectives::reduce",
					"Incompatible given value type and monoid domains while the no_casting"
					"descriptor was set"
				);

				// check whether the monoid has all its domains equal
				constexpr bool same_domains =
					std::is_same< typename Monoid::D1, typename Monoid::D2 >::value &&
					std::is_same< typename Monoid::D2, typename Monoid::D3 >::value;

				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// dynamic checks
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// get identity
				typename Monoid::D3 id = monoid.template
					getIdentity< typename Monoid::D3 >();

				// dispatch
				return reduce_allreduce_generic<
					descr, typename Monoid::Operator, IOType, !same_domains, false
				>(
					inout, root, monoid.getOperator(), &id, data,
					"grb::collectives< BSP >::reduce (monoid)"
				);
			}

			/**
			 * Schedules a broadcast operation of a single object of type IOType per process.
			 * The broadcast shall be complete by the end of the call. This is a collective
			 * graphBLAS operation. The BSP costs are as for the PlatformBSP #broadcast.
			 *
			 * @tparam IOType The type of the to-be broadcast value.
			 *
			 * @param[in,out] inout On input: the value at the root process to be broadcast.
			 *                      On output at process \a root: the same value.
			 *                      On output at non-root processes: the value at root.
			 *
			 * \parblock
			 * \par Performance semantics: common
			 * Whether system calls will happen depends on the LPF engine compiled with,
			 * as does whether buffer space is proportional to the payload size is
			 * required. In principle, when using a fabric like Inifiband and when using
			 * the LPF ibverbs engine, the intended IB zero-copy behaviour is attained.
			 *
			 * All below variants in any backend shall not result in dynamic memory
			 * allocations.
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: serial
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ NP \f$ ;
			 * -# BSP cost: \f$ NPg + l \f$;
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: two phase
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ 2N \f$ ;
			 * -# BSP cost: \f$ 2(Ng + l) \f$;
			 * \endparblock
			 *
			 * \parblock
			 * \par Performance semantics: two level tree
			 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
			 * -# local work: \f$ 0 \f$ ;
			 * -# transferred bytes: \f$ 2\sqrt{P}N \f$ ;
			 * -# BSP cost: \f$ 2(\sqrt{P}Ng + l) \f$;
			 * \endparblock
			 */
			template< typename IOType >
			static RC broadcast( IOType &inout, const lpf_pid_t root = 0 ) {
				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// dynamic checks
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// catch trivial request first
				if( data.P == 1 ) {
					return SUCCESS;
				}

				// ensure the global buffer has enough capacity
				RC rc = data.ensureBufferSize( sizeof( IOType ) );
				// ensure we have enough memory slots for local registration
				rc = rc ? rc : data.ensureMemslotAvailable();
				// ensure we can execute the requested collective call
				rc = rc ? rc : data.ensureCollectivesCapacity( 1, 0, sizeof( IOType ) );
				// ensure we have the required h-relation capacity
				// note the below cannot overflow since we guarantee data.P > 1
				rc = rc ? rc
					: data.ensureMaxMessages( std::max( data.P + 1, 2 * data.P - 3 ) );
				if( rc != SUCCESS ) { return rc; }

				// root retrieve buffer area and copies payload into buffer
				// rationale: this saves one global registration, which otherwise is likely
				//            to dominate most uses for this collective call
				if( data.s == static_cast< size_t >( root ) ) {
					IOType * const __restrict__ buffer = data.template getBuffer< IOType >();
					*buffer = inout;
				}

				// register destination area, schedule broadcast, and wait for it to finish
				lpf_memslot_t dest_slot = LPF_INVALID_MEMSLOT;
				lpf_err_t lpf_rc = lpf_register_local(
					data.context, &inout, sizeof( IOType ), &dest_slot );
				if( lpf_rc == SUCCESS ) {
					lpf_rc = lpf_broadcast( data.coll, data.slot, dest_slot, sizeof( IOType ),
						root );
				}
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// LPF error handling
				rc = internal::checkLPFerror( lpf_rc,
					"grb::collectives< BSP >::broadcast (scalar)" );

				// cleanup
				if( dest_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
					(void) lpf_deregister( data.context, dest_slot );
				}

				// done
				return rc;
			}

			/**
			 * Schedules a broadcast of a raw array of a given type.
			 *
			 * @tparam IOType The array element type.
			 *
			 * @param[in,out] inout A pointer to the array to broadcast (for the root
			 *                      user process), or a pointer where to store the array
			 *                      to be broadcast (for all other user processes).
			 * @param[in] size      The size, in number of array elements, of the array
			 *                      to be broadcast. Must match across all user processes
			 *                      in the collective call.
			 * @param[in] root      Which user process ID is the root.
			 *
			 * \parblock
			 * \par Performance semantics
			 *
			 * Please refer to the LPF collectives higher-level library for the
			 * performance semantics of this call. (This function does not implements
			 * its own custom logic for this primitive.)
			 *
			 * This cost should be appended with the cost of registering \a inout as a
			 * memory space for global RDMA communication.
			 * \endparblock
			 *
			 * @returns grb::SUCCESS On successful broadcast of the requested array.
			 * @returns grb::PANIC   If the communication layer has failed.
			 */
			template< Descriptor descr = descriptors::no_operation, typename IOType >
			static RC broadcast(
				IOType * inout, const size_t size, const size_t root = 0
			) {
				// we need access to LPF context
				internal::BSP1D_Data &data = internal::grb_BSP1D.load();

				// check contract
				if( root >= data.P ) {
					return ILLEGAL;
				}

				// catch trivial cases
				if( data.P == 1 || size == 0 ) {
					return SUCCESS;
				}

				// an array of arbitrary size is probably best not copied, we hence incur
				// the extra latency of registering inout

				// ensure we have enough memory slots for global registration
				RC rc = data.ensureMemslotAvailable();
				// ensure we can execute the requested collective call
				rc = rc ? rc
					: data.ensureCollectivesCapacity( 1, 0, size * sizeof( IOType ) );
				// ensure we have the required h-relation capacity
				// note the below cannot overflow since we guarantee data.P > 1
				rc = rc ? rc
					: data.ensureMaxMessages( std::max( data.P + 1, 2 * data.P - 3 ) );
				// propagate any errors
				if( rc != SUCCESS ) { return rc; }

				// get byte size
				const size_t bsize = size * sizeof( IOType );

				// register array
				lpf_memslot_t user_slot = LPF_INVALID_MEMSLOT;
				lpf_err_t lpf_rc = lpf_register_global(
					data.context, &inout, bsize, &user_slot );

				// activate registration
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// schedule broadcast
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_rc = lpf_broadcast( data.coll, user_slot, user_slot, bsize, root );
				}

				// execute broadcast
				if( lpf_rc == LPF_SUCCESS ) {
					lpf_sync( data.context, LPF_SYNC_DEFAULT );
				}

				// LPF error handling
				rc = internal::checkLPFerror( lpf_rc,
					"grb::collectives< BSP >::broadcast (array)" );

				// cleanup
				if( user_slot != LPF_INVALID_MEMSLOT && lpf_rc != LPF_ERR_FATAL ) {
					(void) lpf_deregister( data.context, user_slot );
				}

				// done
				return rc;
			}

	};

} // namespace grb

#undef NO_CAST_ASSERT_BLAS0

#endif // end ``_H_GRB_BSP_COLL''

