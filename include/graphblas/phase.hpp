
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
 * Defines the various phases an ALP/GraphBLAS primitive may be executed with.
 *
 * @author A. N. Yzelman
 */

#ifndef _H_GRB_PHASE
#define _H_GRB_PHASE


namespace grb {

	/**
	 * Primitives with sparse ALP/GraphBLAS output containers may run into the
	 * issue where an appropriate #grb::capacity may not always be clear. This is
	 * classically the case for level-3 sparse BLAS primitives, which commonly is
	 * solved by splitting up the computation into a symbolic and numeric phase.
	 * During the symbolic phase, the computation is simulated in order to derive
	 * the required capacity of the output container, which is then immediately
	 * resized. Then during the numeric phase, the actual computation is carried
	 * out, knowing that the output container is large enough to hold the requested
	 * output.
	 *
	 * A separation in a symbolic and numeric phase is not the only possible split;
	 * for example, required output capacities may be estimated during a first
	 * stage, while a second stage will then dynamically allocate additional memory
	 * if the estimation proved too optimistic.
	 *
	 * We recognise that:
	 *    1. not only level-3 primitives may require a two-stage approach-- for
	 *       example, a backend could be designed to support extremely large-sized
	 *       vectors that contains relatively few nonzeroes, in which case also
	 *       level-1 and level-2 primitives may benefit of symbolic and numeric
	 *       phases.
	 *    2. especially for level-1 and level-2 primitives, it may also be that
	 *       single-phase approaches are feasible. Hence ALP/GraphBLAS defines that
	 *       the execute phase, #grb::EXECUTE, is the default when calling
	 *       an ALP/GraphBLAS primitive without an explicit phase argument.
	 *    3. sometimes speculative execution is warranted; these apply to
	 *       situations where
	 *          -# capacities are almost surely sufficient, \em and
	 *          -# partial results, if the full output could not be computed due to
	 *             capacity issues, are in fact acceptable.
	 *
	 * To cater to a wide range of approaches and use cases, we support the
	 * following three phases:
	 *    1. #grb::RESIZE, which resizes capacities based on the requested
	 *       operation;
	 *    2. #grb::EXECUTE, which attempts to execute the computation
	 *       assuming the capacity is sufficient;
	 *    3. #grb::TRY, which attempts to execute the computation, and
	 *       does not mind if the capacity turns out to be insufficient.
	 *
	 * Backends must give precise performance semantics to primitives executing in
	 * each of the three possible phases. Backends can only fail with
	 * #grb::OUTOFMEM or #grb::PANIC when an operation is called using the resize
	 * phase and is immediately followed by an equivalent call using the execute
	 * phase-- otherwise, it must succeed and complete the requested computation.
	 *
	 * Summarising the above, a call to any ALP/GraphBLAS primitive f with
	 * (potentially sparse) output container A can be made in three ways:
	 *   1. f( A, ..., EXECUTE ), which shall always be successful if it somehow
	 *      is guaranteed that \a A has enough capacity prior to the call. If
	 *      \a A did not have enough capacity, the call to \a f shall fail and
	 *      the contents of \a A, after function exit, shall be cleared. Failure
	 *      is indicated by the #grb::RC::ILLEGAL error code (since it indicates
	 *      a container with invalid capacity was used for output).
	 *   2. a successful call to f( A, ..., RESIZE ) shall guarantee that a
	 *      following call to f( A, ..., EXECUTE ) is successful;
	 *   3. a call to f( A, ..., TRY ), which may or may not succeed. If the call
	 *      does not succeed, then \a A, after function exit:
	 *        -# contains exactly #grb::capacity (of \a A) nonzeroes;
	 *        -# has nonzeroes at the coordinates where \a A on entry had
	 *           nonzeroes;
	 *        -# has nonzeroes with values equal to those that would have been
	 *           computed at its coordinates were the call successul; and
	 *        -# does not have computed all nonzeroes that would have been present
	 *           if the call were successful (or otherwise it should have returned
	 *           #grb::SUCCESS).
	 *
	 * \note Calls can typically also return #grb::PANIC, which, if returned,
	 *       makes undefined the contents of all ALP/GraphBLAS containers as well
	 *       as makes undefined the state of ALP/GraphBLAS as a whole.
	 *
	 * The following code snippets, assuming all unchecked return codes are
	 * #grb::SUCCESS, thus are semantically equivalent:
	 *
	 * \code
	 * // default capacity of A is sufficient for \a f to succeed
	 * f( A, ..., EXECUTE );
	 * \endcode
	 *
	 * \code
	 * if( resize( A, sufficient_capacity_for_output_of_f ) == SUCCESS ) {
	 *    f( A, ..., EXECUTE );
	 * }
	 * \endcode
	 *
	 * \code
	 * if( f( A, ..., RESIZE ) == SUCCESS ) {
	 *     f( A, ..., EXECUTE );
	 * }
	 * \endcode
	 *
	 * \code
	 * resize( B, nnz( A ) );
	 * set( B, A );
	 * if( f( A, ..., EXECUTE ) == ILLEGAL ) {
	 *     f( B, ..., RESIZE );
	 *     std::swap( A, B );
	 * }
	 * \endcode
	 *
	 * \code
	 * resize( B, nnz( A ) );
	 * set( B, A );
	 * while( f( A, ..., EXECUTE ) == ILLEGAL ) {
	 *     resize( A, capacity( A ) + 1 );
	 *     set( A, B );
	 * }
	 * \endcode
	 *
	 * \note If the matrix \a A is empty on entry, then the latter two code
	 *       snippets do not require the use \a B as a temporary buffer.
	 *
	 * \note Since #grb::EXECUTE is the default phase, any occurrance of
	 *       <code>f( A, ..., EXECUTE )</code> may be replaced with
	 *       <code>f( A, ... )</code>.
	 *
	 * The above code snippets do not include try phases since whenever output
	 * containers do not have enough capacity, primitives executed using
	 * #grb::TRY will \em not generate equivalent results.
	 *
	 */
	enum Phase {

		/**
		 * Speculatively assumes that the output container(s) of the requested
		 * operation lack the necessary capacity to hold all outputs of the
		 * computation. Instead of executing the requested operation, this phase
		 * attempts to both estimate and resize the output container(s).
		 *
		 * A successful call using this phase guarantees that a subsequent and
		 * equivalent call using the #grb::EXECUTE phase shall be successful.
		 *
		 * Here, an <em>equivalent call</em> means that the operation must be called
		 * with exactly the same arguments, except for the #grb::Phase argument.
		 *
		 * Here, <em>subsequent</em> means that all involved containers are not
		 * arguments to any other ALP/GraphBLAS primitives prior to the final call
		 * that requests the execute phase.
		 *
		 * Different from #grb::resize, calling operations using the resize phase does
		 * \em not modify the contents of output containers, and may only enlargen
		 * capacities-- not shrink them.
		 *
		 * \note This specification does \em not disallow implementations or backends
		 *       that perform part of the computation during the resize phase. Any
		 *       such behaviour is totally optional for implementations and backends.
		 *       However, any progress made in such manner must remain hidden from the
		 *       user since output container contents must not be modified by
		 *       primitives executing a resize phase.
		 *
		 * A backend must define clear performance semantics for each primitive and
		 * for each phase that primitive can be called with. In particular, backends
		 * must specify whether system calls such as dynamic memory allocations or
		 * frees may occur, and whether primtives operating in a resize phase may
		 * return #grb::OUTOFMEM.
		 */
		RESIZE,

		/**
		 * Speculatively assumes that the output container of the requested operation
		 * has enough capacity to complete the computation, and attempts to do so.
		 *
		 * If the capacity was indeed found to be sufficient, then the computation
		 * \em must complete as specified-- unless #grb::PANIC is returned.
		 *
		 * If, nevertheless, capacity was not sufficient then the result of the
		 * computation is incomplete and the primitive shall return #grb::FAILED.
		 * Regarding each output container \a A, the following are guaranteed:
		 *    -# the capacity of \a A remains unchanged;
		 *    -# contains #grb::capacity (of \a A) nonzeroes;
		 *    -# has nonzeroes at the coordinates where \a A on entry had nonzeroes;
		 *    -# has nonzeroes with values equal to those that would have been
		 *       computed at its coordinates were the call successul; and
		 *    -# does not contain all nonzeroes that would have been present in \a A
		 *       were the call successful (or otherwise #grb::SUCCESS would have been
		 *       returned instead).
		 *
		 * \warning If execution failed, then even though the semantics guarantee
		 *          valid partial output, there generally is no way to recover the
		 *          full output without re-initiating the full computation. In other
		 *          words, this mechanism does not allow for the partial computation
		 *          to complete the remainder computation using less effort than the
		 *          full computation would have required. This is the main difference
		 *          with the #grb::EXECUTE phase.
		 *
		 * \note This phase is particularly useful if partial output is still usable
		 *       and recomputation to generate the full output is not required.
		 *
		 * A backend must define clear performance semantics for each primitive and
		 * for each phase that primitive can be called with.
		 *
		 * \warning The <tt>try</tt> phase is current experimental and \em not broadly
		 *          supported in the reference implementation.
		 */
		TRY,

		/**
		 * Speculatively assumes that the output container of the requested operation
		 * has enough capacity to complete the computation, and attempts to do so.
		 *
		 * If the capacity was indeed found to be sufficient, then the computation
		 * \em must complete as specified. In this case, capacities are additionally
		 * \em not allowed to be modified by the call to the primitive using the
		 * execute phase.
		 *
		 * If, instead, the output container capacity was found to be insufficient,
		 * then the requested operation may return #grb::FAILED, in which case the
		 * contents of output containers shall be cleared.
		 *
		 * \note That on failure a primitive called using the execute phase may
		 *       destroy any pre-existing contents of output containers is a critical
		 *       difference with the #grb::TRY phase.
		 *
		 * \warning When calling ALP/GraphBLAS primitives without specifying a phase
		 *          explicitly, this execute phase will be assumed by default.
		 *
		 * A backend must define clear performance semantics for each primitive and
		 * for each phase that primitive can be called with. In particular, backends
		 * must specify whether system calls such as dynamic memory allocations or
		 * frees may occur, and whether primtives operating in a resize phase may
		 * return #grb::OUTOFMEM.
		 *
		 * \note Typically, implementations and backends are advised to specify no
		 *       system calls and in particular dynamic memory management calls are
		 *       allowed as part of an execute phase.
		 */
		EXECUTE

	};

} // namespace grb

#endif // end ``_H_GRB_PHASE''

