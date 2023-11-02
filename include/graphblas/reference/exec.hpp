
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

#if ! defined _H_GRB_REFERENCE_EXEC || defined _H_GRB_REFERENCE_OMP_EXEC
#define _H_GRB_REFERENCE_EXEC

#include <graphblas/backends.hpp>
#include <graphblas/base/exec.hpp>

#include "init.hpp"


namespace grb {

	/**
	 * No implementation notes.
	 */
	template< EXEC_MODE mode >
	class Launcher< mode, reference > {

		protected:

			template< typename Runner >
			RC init_and_run(
				Runner &runner,
				const bool broadcast
			) const {
				// value doesn't matter for a single user process
				(void) broadcast;
				// intialise
				RC ret = grb::init();
				// call algo
				if( ret == SUCCESS ) {
					runner();
					ret = grb::finalize();
				}
				// and done
				return ret;
			}


		public:

			/**
			 * This implementation only accepts a single user process.
			 * It ignores \a hostname and \a port.
			 *
			 * @param[in] process_id The ID of the calling process.
			 * @param[in] nprocs     The number of calling processes.
			 * @param[in] hostname   One of the user process host names.
			 * @param[in] port       A free port at the given host name.
			 */
			Launcher(
				const size_t process_id = 0,
				const size_t nprocs = 1,
				const std::string hostname = "localhost",
				const std::string port = "0"
			) {
				// ignore hostname and port
				(void) hostname;
				(void) port;
				// sanity checks
				if( nprocs != 1 ) {
					throw std::invalid_argument( "Total number of user processes must be "
						"exactly one when using the reference implementation."
					);
				}
				if( process_id != 0 ) {
					throw std::invalid_argument( "Process ID must always be zero in the "
						"reference implementation."
					);
				}
			}

			/** No implementation notes. */
			~Launcher() {}

			/** No implementation notes. */
			template< typename U >
			RC exec(
				AlpUntypedFunc< U > alp_program,
				const void * const data_in, const size_t in_size,
				U &data_out,
				const bool broadcast = false
			) const {
				// check input arguments
				if( in_size > 0 && data_in == nullptr ) {
					return ILLEGAL;
				}
				auto fun = [ data_in, in_size, &data_out, alp_program ] {
					(*alp_program)( data_in, in_size, data_out );
				};
				return init_and_run( fun, broadcast );
			}

			/** No implementation notes. */
			template< typename T, typename U >
			RC exec(
				AlpTypedFunc< T, U > alp_program,
				const T &data_in, U &data_out,
				const bool broadcast = false
			) {
				auto fun = [ &data_in, &data_out, alp_program ] {
					(*alp_program)( data_in, data_out );
				};
				return init_and_run( fun, broadcast );
			}

			/** No implementation notes. */
			static grb::RC finalize() { return grb::SUCCESS; }

	};

} // namespace grb

// parse this unit again for OpenMP support
#ifdef _GRB_WITH_OMP
 #ifndef _H_GRB_REFERENCE_OMP_EXEC
  #define _H_GRB_REFERENCE_OMP_EXEC
  #define reference reference_omp
  #include "graphblas/reference/exec.hpp"
  #undef reference
  #undef _H_GRB_REFERENCE_OMP_EXEC
 #endif
#endif

#endif // end ``_H_GRB_REFERENCE_EXEC''

