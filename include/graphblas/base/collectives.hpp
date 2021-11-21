
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

#ifndef _H_GRB_COLL_BASE
#define _H_GRB_COLL_BASE

#include <graphblas/backends.hpp>
#include <graphblas/descriptors.hpp>
#include <graphblas/rc.hpp>

namespace grb {

	/**
	 * A static class defining various collective operations on scalars. This
	 * class is templated in terms of the backends that are implemented-- each
	 * implementation provides its own mechanisms to handle collective
	 * communications. These are required for users employing grb::eWiseLambda,
	 * or for users who perform explicit SPMD programming.
	 */
	template< enum Backend implementation >
	class collectives {

	private:
		/** Disallow creating an instance. */
		collectives() {}

	public:
		/**
		 * Schedules an allreduce operation of a single object of type IOType per
		 * process. The allreduce shall be complete by the end of the call. This is a
		 * collective graphBLAS operation. After the collective call finishes, each
		 * user process will locally have available the allreduced value.
		 *
		 * Since this is a collective call, there are \a P values \a inout spread over
		 * all user processes. Let these values be denoted by \f$ x_s \f$, with
		 * \f$ s \in \{ 0, 1, \ldots, P-1 \}, \f$ such that \f$ x_s \f$ equals the
		 * argument \a inout on input at the user process with ID \a s. Let
		 * \f$ \pi:\ \{ 0, 1, \ldots, P-1 \} \to \{ 0, 1, \ldots, P-1 \} \f$ be a
		 * bijection, some unknown permutation of the process ID. This permutation is
		 * must be fixed for any given combination of GraphBLAS implementation and value
		 * \a P. Let the binary operator \a op be denoted by \f$ \odot \f$.
		 *
		 * This function computes \f$ \odot_{i=0}^{P-1} x_{\pi(i)} \f$ and writes the
		 * exact same result to \a inout at each of the \a P user processes.
		 *
		 * In summary, this means 1) this operation is coherent across all processes and
		 * produces bit-wise equivalent output on all user processes, and 2) the result
		 * is reproducible across different runs using the same input and \a P. Yet it
		 * does \em not mean that the order of addition is fixed.
		 *
		 * Since each user process supplies but one value, there is no difference
		 * between a reduce-to-the-left versus a reduce-to-the-right (see grb::reducel
		 * and grb::reducer).
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for reduction.
		 * @tparam IOType   The type of the to-be reduced value.
		 *
		 * @param[in,out] inout On input:  the value at the calling process to be
		 *     	                reduced. On output: the reduced value.
		 * @param[in]      op   The associative operator to reduce by.
		 *
		 * \note If \op is commutative, the implementation free to employ a different
		 *       allreduce algorithm, as long as it is documented well enough so that
		 *       its cost can be quantified.
		 *
		 * @returns grb::SUCCESS When the operation succeeds as planned.
		 * @returns grb::PANIC   When the communication layer unexpectedly fails. When
		 *                       this error code is returned, the library enters an
		 *                       undefined state.
		 *
		 * \parblock
		 * \par Valid descriptors:
		 * -# grb::descriptors::no_operation
		 * -# grb::descriptors::no_casting
		 *  Any other descriptors will be ignored.
		 *  \endparblock
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ N*Operator \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 */
		template< Descriptor descr = descriptors::no_operation, typename Operator, typename IOType >
		static RC allreduce( IOType & inout, const Operator op = Operator() ) {
			(void)inout;
			(void)op;
			return PANIC;
		}

		/**
		 * Schedules a reduce operation of a single object of type IOType per process.
		 * The reduce shall be complete by the end of the call. This is a collective
		 * graphBLAS operation. The BSP costs are as for the PlatformBSP #reduce.
		 *
		 * Since this is a collective call, there are \a P values \a inout spread over
		 * all user processes. Let these values be denoted by \f$ x_s \f$, with
		 * \f$ s \in \{ 0, 1, \ldots, P-1 \}, \f$ such that \f$ x_s \f$ equals the
		 * argument \a inout on input at the user process with ID \a s. Let
		 * \f$ \pi:\ \{ 0, 1, \ldots, P-1 \} \to \{ 0, 1, \ldots, P-1 \} \f$ be a
		 * bijection, some unknown permutation of the process ID. This permutation is
		 * must be fixed for any given combination of GraphBLAS implementation and value
		 * \a P. Let the binary operator \a op be denoted by \f$ \odot \f$.
		 *
		 * This function computes \f$ \odot_{i=0}^{P-1} x_{\pi(i)} \f$ and writes the
		 * result to \a inout at the user process with ID \a root.
		 *
		 * In summary, this the result is reproducible across different runs using the
		 * same input and \a P. Yet it does \em not mean that the order of addition is
		 * fixed.
		 *
		 * Since each user process supplies but one value, there is no difference
		 * between a reduce-to-the-left versus a reduce-to-the-right (see grb::reducel
		 * and grb::reducer).
		 *
		 * @tparam descr    The GraphBLAS descriptor.
		 *                  Default is grb::descriptors::no_operation.
		 * @tparam Operator Which operator to use for reduction.
		 * @tparam IOType   The type of the to-be reduced value.
		 *
		 * @param[in,out] inout On input: the value at the calling process to be
		 *                      reduced. On output at process \a root: the reduced value.
		 *                      On output as non-root processes: same value as on input.
		 * @param[in]       op  The associative operator to reduce by.
		 * @param[in]      root Which process should hold the reduced value. This
		 *                      number must be larger or equal to zero, and must be
		 *                      strictly smaller than the number of user processes
		 *                      \a P.
		 *
		 * @return SUCCESS When the function completes successfully.
		 * @return ILLEGAL When root is larger or equal than \a P. When this code is
		 *                 returned, the state of the GraphBLAS shall be as though
		 *                 this call was never made.
		 * @return PANIC   When an unmitigable error within the GraphBLAS occurs.
		 *                 Upon returning this error, the GraphBLAS enters an
		 *                 undefined state.
		 *
		 * \note If \op is commutative, the implementation free to employ a different
		 *       allreduce algorithm, as long as it is documented well enough so that
		 *       its cost can be quantified.
		 *
		 * \parblock
		 * \par Performance semantics:
		 * -# Problem size N: \f$ P * \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ N*Operator \f$ ;
		 * -# transferred bytes: \f$ N \f$ ;
		 * -# BSP cost: \f$ Ng + N*Operator + l \f$;
		 * \endparblock
		 */
		template< Descriptor descr = descriptors::no_operation, typename Operator, typename IOType >
		static RC reduce( IOType & inout, const size_t root = 0, const Operator op = Operator() ) {
			(void)inout;
			(void)op;
			(void)root;
			return PANIC;
		}

		/**
		 * Schedules a broadcast operation of a single object of type IOType per
		 * process. The broadcast shall be complete by the end of the call. This is
		 * a collective graphBLAS operation. The BSP costs are as for the PlatformBSP
		 * #broadcast.
		 *
		 * @tparam IOType   The type of the to-be broadcast value.
		 *
		 * @param[in,out] inout On input at process \a root: the value to be
		 *                      broadcast.
		 *                      On input at non-root processes: initial values are
		 *                      ignored.
		 *                      On output at process \a root: the input value remains
		 *                      unchanged.
		 *                      On output at non-root processes: the same value held
		 *                      at process ID \a root.
		 * @param[in]      root The user process which is to send out the given input
		 *                      value \a inout so that it becomes available at all
		 *                      \a P user processes. This value must be larger or
		 *                      equal to zero and must be smaller than the total
		 *                      number of user processes \a P.
		 *
		 * @return SUCCESS On the successful completion of this function.
		 * @return ILLEGAL When \a root is larger or equal to \a P. If this code is
		 *                 returned, it shall be as though the call to this function
		 *                 had never occurred.
		 * return PANIC    When the function fails and the library enters an
		 *                 undefined state.
		 *
		 * \parblock
		 * \par Performance semantics: serial
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ NP \f$ ;
		 * -# BSP cost: \f$ NPg + l \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two phase
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2N \f$ ;
		 * -# BSP cost: \f$ 2(Ng + l) \f$;
		 * \endparblock
		 *
		 * \par Performance semantics: two level tree
		 * -# Problem size N: \f$ \mathit{sizeof}(\mathit{IOType}) \f$
		 * -# local work: \f$ 0 \f$ ;
		 * -# transferred bytes: \f$ 2\sqrt{P}N \f$ ;
		 * -# BSP cost: \f$ 2(\sqrt{P}Ng + l) \f$;
		 * \endparblock
		 */
		template< typename IOType >
		static RC broadcast( IOType &inout, const size_t root = 0 ) {
			(void)inout;
			(void)root;
			return PANIC;
		}

		/**
		 * Broadcast on an array of \a IOType.
		 *
		 * The above documentation applies with \a size times <tt>sizeof(IOType)</tt>
		 * substituted in.
		 */
		template< Descriptor descr = descriptors::no_operation, typename IOType >
		static RC broadcast( IOType * inout, const size_t size, const size_t root = 0 ) {
			(void)inout;
			(void)size;
			(void)root;
			return PANIC;
		}

	}; // end class ``collectives''

} // end namespace grb

#endif // end _H_GRB_COLL_BASE
