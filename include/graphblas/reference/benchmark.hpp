
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

	template< enum EXEC_MODE mode >
	class Benchmarker< mode, reference > : protected Launcher< mode, reference >, protected internal::BenchmarkerBase {

	public:

		/**
		 * @param[in] process_id User process ID
		 * @param[in] nprocs     Total number of user processes
		 * @param[in] hostname   One of the user process hostnames
		 * @param[in] port       A free port at the host indicated by \a hostname
		 *
		 * The arguments \a nprocs, \a hostname, and \a port must be equal across all
		 * \a nprocs calls to this function.
		 *
		 * \internal Relies on #grb::Launcher.
		 */
		Benchmarker( size_t process_id = 0,
			size_t nprocs = 1,
			std::string hostname = "localhost",
			std::string port = "0"
		) : Launcher< mode, reference >( process_id, nprocs, hostname, port ) {}

		/** \internal No implementation notes */
		template< typename U >
		RC exec(
			void ( *grb_program )( const void *, const size_t, U & ),
			const void * data_in, const size_t in_size,
			U &data_out,
			const size_t inner, const size_t outer,
			const bool broadcast = false
		) const {
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
		RC exec(
			void ( *grb_program )( const T &, U & ),
			const T &data_in,
			U &data_out,
			const size_t inner,
			const size_t outer,
			const bool broadcast = false
		) {
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
			return Launcher< mode, reference >::finalize();
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

