
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

#if ! defined _H_GRB_REFERENCE_BENCH || defined _H_GRB_REFERENCE_OMP_BENCH
#define _H_GRB_REFERENCE_BENCH

#include <graphblas/base/benchmark.hpp>
#include <graphblas/rc.hpp>

#include "exec.hpp"

namespace grb {

	/**
	 * \internal
	 * Implementation inherits from #grb::internal::BenchmarkerBase and
	 * #grb::Launcher (reference).
	 * \endinternal
	 */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, reference > :
		protected Launcher< mode, reference >, protected internal::BenchmarkerBase
	{

		public:

			/** \internal Delegates to #grb::Launcher (reference) constructor. */
			Benchmarker(
				const size_t process_id = 0,        // user process ID
				const size_t nprocs = 1,            // total number of user processes
				std::string hostname = "localhost", // one of the user process hostnames
				std::string port = "0"              // a free port at hostname
			) : Launcher< mode, reference >( process_id, nprocs, hostname, port ) {}

			/** \internal No implementation notes. */
			template< typename U >
			RC exec(
				void ( *grb_program )( const void *, const size_t, U & ),
				const void * data_in, const size_t in_size,
				U &data_out,
				const size_t inner, const size_t outer,
				const bool broadcast = false
			) const {
				(void) broadcast; // value doesn't matter for a single user process
				// catch illegal argument
				if( in_size > 0 && data_in == nullptr ) {
					return ILLEGAL;
				}
				// initialise GraphBLAS
				RC ret = grb::init();

				// call graphBLAS algo
				if( ret == SUCCESS ) {
					benchmark< U >( grb_program, data_in, in_size, data_out, inner, outer, 0 );
				}
				// finalise the GraphBLAS
				const RC frc = grb::finalize();
				if( ret == SUCCESS ) {
					ret = frc;
				}
				// and done
				return ret;
			}

			/** \internal No implementation notes. */
			template< typename T, typename U >
			RC exec(
				void ( *grb_program )( const T &, U & ), // user GraphBLAS program
				const T &data_in, U &data_out, // input & output data
				const size_t inner,
				const size_t outer,
				const bool broadcast = false
			) {
				(void) broadcast; // value doesn't matter for a single user process
				// initialise GraphBLAS
				RC ret = grb::init();
				// call graphBLAS algo
				if( ret == SUCCESS ) {
					// call graphBLAS algo
					benchmark< T, U, reference >( grb_program, data_in, data_out, inner, outer, 0 );
				}
				// finalise the GraphBLAS
				const RC frc = grb::finalize();
				if( ret == SUCCESS ) {
					ret = frc;
				}
				// and done
				return ret;
			}

	};

} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_BENCH
  #define _H_GRB_REFERENCE_OMP_BENCH
  #define reference reference_omp
  #include "benchmark.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_BENCH
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_BENCH''

