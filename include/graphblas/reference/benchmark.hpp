
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
		public Launcher< mode, reference >, protected internal::BenchmarkerBase
	{

		public:

			/** \internal Delegates to #grb::Launcher (reference) constructor. */
			using Launcher< mode, reference >::Launcher;

			/** \internal No implementation notes. */
			template< typename U >
			RC exec(
				AlpUntypedFunc< void, U > grb_program,
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
				auto fun = [ data_in, in_size, &data_out, grb_program, inner, outer ] {
					benchmark< U, reference >( grb_program, data_in, in_size, data_out, inner,
						outer, 0 );
				};
				return this->init_and_run( fun, broadcast );
			}

			/** \internal No implementation notes. */
			template< typename T, typename U >
			RC exec(
				AlpTypedFunc< T, U > grb_program,
				const T &data_in, U &data_out,
				const size_t inner,
				const size_t outer,
				const bool broadcast = false
			) {
				auto fun = [ &data_in, &data_out, grb_program, inner, outer ] {
					benchmark< T, U, reference >( grb_program, data_in, data_out, inner,
						outer, 0 );
				};
				return this->init_and_run( fun, broadcast );
			}

			using Launcher< mode, reference >::finalize;
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

