
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

#include <graphblas/base/benchmark.hpp>
#include <graphblas/rc.hpp>
#include <graphblas/utils/TimerResults.hpp>

#include "exec.hpp"


namespace grb {

	namespace internal {

		/** Data structure with input and benchmarking information. */
		template< typename InputType, typename OutputType, bool untyped_call >
		struct BenchmarkDispatcher :
			ExecDispatcher< InputType, OutputType, untyped_call >,
			protected BenchmarkerBase
		{

			size_t inner;
			size_t outer;

			BenchmarkDispatcher() : inner( 0 ), outer( 0 ) {}

			BenchmarkDispatcher(
				const InputType *in, size_t s, bool bc,
				size_t _inner, size_t _outer
			) : ExecDispatcher< InputType, OutputType, untyped_call >( in, s, bc),
				inner( _inner ), outer( _outer )
			{}

			/**
			 * Benchmark the ALP function \p fun with the given input/output parameters.
			 */
			grb::RC operator()(
				const lpf_func_t fun,
				size_t in_size, const InputType *in, OutputType *out,
				lpf_pid_t s, lpf_pid_t P
			) const {
				struct void_runner {
					const lpf_func_t _fun;
					size_t _in_size;
					const InputType *_in;
					OutputType *_out;
					lpf_pid_t _s;
					lpf_pid_t _P;

					inline void operator()() {
						ExecDispatcher< InputType, OutputType, untyped_call >::lpf_grb_call
							( _fun, _in_size, _in, _out, _s, _P );
					}

				} runner = { fun, in_size, in, out, s, P };
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
				lpf_func_t alp_program, const T *data_in, size_t in_size, U *data_out,
				const size_t inner, const size_t outer, bool broadcast
			) const {
				typedef internal::BenchmarkDispatcher< T, U, untyped_call > Disp;
				Disp disp_info = { data_in, in_size, broadcast, inner, outer };
				return this->template run_lpf< T, U, Disp >( alp_program, disp_info,
					data_out );
			}


		public:

			// import constructor(s) from base class, implicitly based on mode
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
			 * @param[in]  broadcast   Whether to broadcast inputs from user process 0
			 *                         to all other user processes.
			 *
			 * @returns grb::SUCCESS On a successfully completed benchmark call, and a
			 *                       descriptive error code otherwise.
			 */
			template< typename U > RC exec(
				AlpUntypedFunc< void, U > alp_program,
				const void * data_in, const size_t in_size,
				U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {
				// check input arguments
				if( in_size > 0 && data_in == nullptr ) {
					return ILLEGAL;
				}
				return pack_and_run< void, U, true >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					data_in, in_size, &data_out, broadcast,
					inner, outer
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
			 * @returns grb::SUCCESS On a successfully completed benchmark call, and a
			 *                       descriptive error code otherwise.
			 */
			template< typename T, typename U > RC exec(
				AlpTypedFunc< T, U > alp_program, const T &data_in, U &data_out,
				const size_t inner, const size_t outer, const bool broadcast = false
			) const {
				return pack_and_run< T, U, false >(
					reinterpret_cast< lpf_func_t >( alp_program ),
					&data_in, sizeof( T ), &data_out, broadcast, inner, outer );
			}

			using Launcher< mode, BSP1D >::finalize;

		};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_BENCH''

