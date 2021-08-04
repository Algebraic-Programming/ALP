
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
 * @date 17th of April, 2017
 */

#if ! defined _H_GRB_BANSHEE_BENCH
#define _H_GRB_BANSHEE_BENCH

// this file contains partial template specialisations and should be included from the header containing the base definition
#ifndef _H_GRB_BENCH
#error "This file should be included from the header containing the base class definition of grb::Benchmarker"
#endif

#include "graphblas/banshee/exec.hpp"
#include "graphblas/benchmark.hpp"
#include "graphblas/exec.hpp"
#include "graphblas/rc.hpp"
#include <graphblas/base/benchmark.hpp>

namespace grb {

	template< enum EXEC_MODE mode >
	class Benchmarker< mode, banshee > : protected Launcher< mode, banshee >, protected internal::BenchmarkerBase {

	public:
		Benchmarker( size_t process_id = 0,     // user process ID
			size_t nprocs = 1,                  // total number of user processes
			std::string hostname = "localhost", // one of the user process hostnames
			std::string port = "0"              // a free port at hostname
			) :
			Launcher< mode, banshee >( process_id, nprocs, hostname, port ) {}

		template< typename U >
		RC
		exec( void ( *grb_program )( const void *, const size_t, U & ), const void * data_in, const size_t in_size, U & data_out, const size_t inner, const size_t outer, const bool broadcast = false )
			const {
			(void)broadcast; // value doesn't matter for a single user process
			// initialise GraphBLAS
			RC ret = init();
			// call graphBLAS algo
			if( ret == SUCCESS ) {
				benchmark< U >( grb_program, data_in, in_size, data_out, inner, outer, 0 );
			}
			// finalise the GraphBLAS
			const RC frc = finalize();
			if( ret == SUCCESS ) {
				ret = frc;
			}
			// and done
			return ret;
		}

		/** No implementation notes. */
		template< typename T, typename U >
		RC exec( void ( *grb_program )( const T &, U & ), // user GraphBLAS program
			const T & data_in,
			U & data_out, // input & output data
			const size_t inner,
			const size_t outer,
			const bool broadcast = false ) {
			(void)broadcast; // value doesn't matter for a single user process
			// initialise GraphBLAS
			RC ret = init();
			// call graphBLAS algo
			if( ret == SUCCESS ) {
				// call graphBLAS algo
				benchmark< T, U >( grb_program, data_in, data_out, inner, outer, 0 );
			}
			// finalise the GraphBLAS
			const RC frc = finalize();
			if( ret == SUCCESS ) {
				ret = frc;
			}
			// and done
			return ret;
		}

		/** No implementation notes. */
		static RC finalize() {
			return Launcher< mode, banshee >::finalize();
		}
	};

} // namespace grb

#endif // end ``_H_GRB_BANSHEE_BENCH''
