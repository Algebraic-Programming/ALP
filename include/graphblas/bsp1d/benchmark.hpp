
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

		/** Data structure with input and benchmarkign information. */
		template< typename InputType, typename OutputType, bool untyped_call > struct BenchmarkDispatcher :
			ExecDispatcher< InputType, OutputType, untyped_call >, protected BenchmarkerBase {
			size_t inner;
			size_t outer;

			BenchmarkDispatcher() = default;

			BenchmarkDispatcher( const InputType *in, size_t s, bool bc, size_t _inner, size_t _outer ) :
				ExecDispatcher< InputType, OutputType, untyped_call >( in, s, bc), inner( _inner ), outer( _outer ) {}

			/** Benchmark the GRB function \p fun with the various input/output parameters. */
			void operator()( const lpf_func_t fun, size_t in_size, const InputType *in, OutputType *out, lpf_pid_t s, lpf_pid_t P ) const {
				struct void_runner {
					const lpf_func_t _fun;
					size_t _in_size;
					const InputType *_in;
					OutputType *_out;
					lpf_pid_t _s;
					lpf_pid_t _P;

					inline void operator()() {
						ExecDispatcher< InputType, OutputType, untyped_call >::lpf_grb_call( _fun, _in_size, _in, _out, _s, _P );
					}

				} runner{ fun, in_size, in, out, s, P };
				benchmark< BSP1D >( runner, out->times, inner, outer, s ); // TODO: handle return value
			}
		};

	} // namespace internal

	/** Data structure to launch a GRB function and benchmark it. */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, BSP1D > : protected Launcher< mode, BSP1D >
	{

	private:

		/** Pack input/output data and run the given GRB function. */
		template< typename T, typename U, bool untyped_call >
		RC pack_and_run(
			lpf_func_t grb_program, // user GraphBLAS program
			const T *data_in,
			size_t in_size,
			U *data_out,
			const size_t inner,
			const size_t outer,
			bool broadcast
		) const {
			using Disp = internal::BenchmarkDispatcher< T, U, untyped_call >;
			Disp disp_info{ data_in, in_size, broadcast, inner, outer };
			return this-> template run_lpf< T, U, Disp >( grb_program, disp_info, data_out );
		}

	public:
		// import constructor(s) from base class, implicitly based on mode
		using Launcher< mode, BSP1D >::Launcher;

		/**
		 * Run an untyped GRB function distributed.
		 * 
		 * @tparam U output type
		 * @param grb_program GRB function to run
		 * @param data_in pointer to input data
		 * @param in_size size of input data
		 * @param data_out output data
		 * @param inner number of inner iterations
		 * @param outer number of outer iterations
		 * @param broadcast whether to bradcaso inputs from node 0 to other nodes
		 * @return RC error code
		 */
		template< typename U > RC exec(
			AlpUntypedFunc< void, U > grb_program,
			const void * data_in,
			const size_t in_size,
			U &data_out,
			const size_t inner,
			const size_t outer,
			const bool broadcast = false
		) const {
			// check input arguments
			if( in_size > 0 && data_in == nullptr ) {
				return ILLEGAL;
			}
			return pack_and_run< void, U, true >( reinterpret_cast< lpf_func_t >( grb_program ),
				data_in, in_size, &data_out, broadcast, inner, outer );
		}

		/**
		 * Run a typed GRB function distributed.
		 * 
		 * @tparam T input type
		 * @tparam U output type
		 * @param grb_program GRB function to run
		 * @param data_in input data
		 * @param data_out output data
		 * @param inner number of inner iterations
		 * @param outer number of outer iterations
		 * @param broadcast whether to bradcaso inputs from node 0 to other nodes
		 * @return RC error code
		 */
		template< typename T, typename U > RC exec(
			AlpTypedFunc< T, U > grb_program,
			const T &data_in,
			U &data_out,
			const size_t inner,
			const size_t outer,
			const bool broadcast = false
		) const {
			return pack_and_run< T, U, false >( reinterpret_cast< lpf_func_t >( grb_program ),
				&data_in, sizeof( T ), &data_out, broadcast, inner, outer );
		}

		using Launcher< mode, BSP1D >::finalize;
	};

} // namespace grb

#endif // end ``_H_GRB_BSP1D_BENCH''

