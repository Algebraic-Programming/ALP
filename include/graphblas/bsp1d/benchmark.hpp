
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
 * @author A. N. Yzelman; Alberto Scolari
 * @date 17th of April, 2017; 28 of August 2023
 */

#ifndef _H_GRB_BSP1D_BENCH
#define _H_GRB_BSP1D_BENCH

#include <string>
#include <type_traits>

#include <lpf/core.h>

#include <graphblas/rc.hpp>

#include <graphblas/base/benchmark.hpp>

#include <graphblas/utils/timerResults.hpp>

#include "exec.hpp"


namespace grb {

	namespace internal {

		/**
		 * Data structure with input and benchmarking information.
		 *
		 * @tparam InputType  The input type.
		 * @tparam OutputType The output type.
		 * @tparam _mode      The #grb::EXEC_MODE of the benchmarker.
		 *
		 * In automatic mode, this struct must be broadcast from process 0 to the
		 * other processes, as it contains the valid number of inner and outer
		 * iterations. In other modes, all processes must choose the same number
		 * of inner/outer iterations, otherwise deadlocks may occur.
		 *
		 * @tparam _requested_broadcast Whether or not the user has requested input be
		 *                              broadcast.
		 *
		 * @tparam untyped_call         Whether the user has made a benchmark request
		 *                              using an untyped ALP program.
		 */
		template<
			typename InputType, typename OutputType,
			EXEC_MODE _mode,
			bool _requested_broadcast,
			bool untyped_call
		>
		struct BenchmarkDispatcher :
			ExecDispatcher<
				InputType, OutputType,
				_mode, _requested_broadcast,
				untyped_call
			>,
			protected BenchmarkerBase
		{
			/** Whether the dispatcher requires broadcasting. */
			static constexpr bool needs_initial_broadcast = _mode == AUTOMATIC;

			/** Inner number of experiments. */
			size_t inner;

			/** Outer number of experiments. */
			size_t outer;

			/**
			 * Builds dispatcher from basic information.
			 *
			 * @param[in] _in      Pointer to the input data.
			 * @param[in] _in_size Byte size of the input data.
			 * @param[in] _inner   The nummer of inner iterations.
			 * @param[in] _outer   The number of outer iterations.
			 */
			BenchmarkDispatcher(
				const InputType *_in, const size_t _in_size,
				size_t _inner, size_t _outer
			) :
				ExecDispatcher< InputType, OutputType, _mode, _requested_broadcast,
					untyped_call >( _in, _in_size ),
				inner( _inner ), outer( _outer )
			{}

			/**
			 * Reconstruct object from LPF args, where it is embedded in its input field.
			 *
			 * @param[in] s    The process ID.
			 * @param[in] args The LPF I/O arguments.
			 */
			BenchmarkDispatcher( const lpf_pid_t s, const lpf_args_t args ) :
				ExecDispatcher<
					InputType, OutputType,
					_mode, _requested_broadcast,
					untyped_call
				>( nullptr, 0 )
			{
				if( s > 0 && _mode == AUTOMATIC ) {
					inner = 0;
					outer = 0;
					return;
				}
				typedef BenchmarkDispatcher<
					InputType, OutputType,
					_mode, _requested_broadcast,
					untyped_call
				> self_t;
				const self_t *orig = reinterpret_cast< const self_t * >( args.input );
				this->in = orig->in;
				this->in_size = orig->in_size;
				inner = orig->inner;
				outer = orig->outer;
			}

			/**
			 * Benchmark the ALP function \a fun with the given input/output parameters.
			 *
			 * @param[in]  fun     The ALP function to run.
			 * @param[in]  s       The process ID.
			 * @param[in]  P       The total nuber of processes.
			 * @param[in]  in      Pointer to the input data.
			 * @param[in]  in_size Byte size of the input data.
			 * @param[out] out     Pointer where to output.
			 */
			grb::RC operator()(
				const lpf_func_t fun,
				const lpf_pid_t s, const lpf_pid_t P,
				const InputType * const in, const size_t in_size,
				OutputType * const out
			) const {
				auto runner = [ fun, in_size, in, out, s, P ] () {
					ExecDispatcher<
						InputType, OutputType,
						_mode, _requested_broadcast,
						untyped_call
					>::lpf_grb_call( fun, s, P, in, in_size, out );
				};
				return benchmark< BSP1D >( runner, out->times, inner, outer, s );
			}

		};

	} // namespace internal

	/**
	 * Collection of processes that can launch an ALP function and benchmark it.
	 */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, BSP1D > : protected Launcher< mode, BSP1D > {

		private:

			/** Pack input/output data and run the given ALP function. */
			template< typename T, typename U, bool untyped_call >
			RC pack_and_run(
				const lpf_func_t alp_program,
				const T * const data_in, const size_t in_size,
				U * const data_out,
				const size_t inner, const size_t outer,
				const bool broadcast
			) const {
				if( broadcast ) {
					typedef internal::BenchmarkDispatcher<
						T, U, mode, true,
						untyped_call
					> Disp;
					Disp disp_info( data_in, in_size, inner, outer );
					return this->template run_lpf< T, U, Disp >(
						alp_program,
						reinterpret_cast< void * >( &disp_info ),
						sizeof( Disp ), data_out
					);
				} else {
					typedef internal::BenchmarkDispatcher<
						T, U, mode, false,
						untyped_call
					> Disp;
					Disp disp_info = { data_in, in_size, inner, outer };
					return this->template run_lpf< T, U, Disp >(
						alp_program,
						reinterpret_cast< void * >( &disp_info ),
						sizeof( Disp ), data_out
					);
				}
			}


		public:

			/** import constructor(s) from base class, implicitly based on mode */
			using Launcher< mode, BSP1D >::Launcher;

			/**
			 * Run an untyped ALP function in parallel.
			 *
			 * @tparam U The output type.
			 *
			 * @param[in]  alp_program ALP function to execute in parallel.
			 * @param[in]  data_in     Pointer to input data.
			 * @param[in]  in_size     Size (in bytes) of the input data.
			 * @param[out] data_out    Output data.
			 * @param[in]  inner       Number of inner iterations.
			 * @param[in]  outer       Number of outer iterations.
			 * @param[in]  broadcast   Whether to broadcast inputs from user process zero
			 *                         to all other user processes.
			 *
			 * @returns grb::SUCCESS On a successfully completed benchmark call.
			 * @returns grb::ILLEGAL If \a data_in is <tt>nullptr</tt> but \a in_size is
			 *                       larger than zero.
			 * @returns grb::PANIC   On an unrecoverable critical failure (see base
			 *                       specification).
			 */
			template< typename U >
			RC exec(
				const AlpUntypedFunc< U > alp_program,
				const void * const data_in, const size_t in_size,
				U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {
				// check input arguments
				if( in_size > 0 && data_in == nullptr ) {
					return grb::ILLEGAL;
				}
				return pack_and_run< void, U, true >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					data_in, in_size, &data_out,
					inner, outer,
					broadcast
				);
			}

			/**
			 * Run a typed ALP function in parallel.
			 *
			 * @tparam T Input type.
			 * @tparam U Output type.
			 *
			 * @param[in]  alp_program The ALP function to execute in parallel.
			 * @param[in]  data_in     Pointer to the input data.
			 * @param[out] data_out    The output data.
			 * @param[in]  inner       Number of inner iterations.
			 * @param[in]  outer       Number of outer iterations.
			 * @param[in]  broadcast   Whether to broadcast inputs from user process zero
			 *                         to all other user processes.
			 *
			 * @returns grb::SUCCESS On a successfully completed benchmark call.
			 * @returns grb::ILLEGAL If \a broadcast was false and the benchmarker is in
			 *                       #AUTOMATIC mode, while \a T is not default-
			 *                       constructible.
			 * @returns grb::PANIC   On unrecoverable errors (see the base specification
			 *                       for details).
			 */
			template< typename T, typename U >
			RC exec(
				const AlpTypedFunc< T, U > alp_program,
				const T &data_in, U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {
				return pack_and_run< T, U, false >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					&data_in, sizeof( T ), &data_out,
					inner, outer,
					broadcast
				);
			}

			/** Reuse BSP1D launcher implementation of finalize. */
			using Launcher< mode, BSP1D >::finalize;

	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_BENCH''

