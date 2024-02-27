
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
 * Nonblocking implementation of the benchmarker.
 *
 * @author Aristeidis Mastoras
 * @date 16th of May, 2022
 */

#ifndef _H_GRB_NONBLOCKING_BENCH
#define _H_GRB_NONBLOCKING_BENCH

#include <graphblas/rc.hpp>
#include <graphblas/reference/benchmark.hpp>


namespace grb {

	/**
	 * The Benchmarker class is based on that of the reference backend
	 *
	 * \internal The public API simply wraps the reference Benchmarker.
	 */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, nonblocking >: public Benchmarker< mode, reference > {

		public:

			/** \internal Delegates to #grb::Benchmarker (reference) constructor. */
			using Benchmarker< mode, reference >::Benchmarker;

			/** \internal Delegates to #grb::Benchmarker (reference) finalize. */
			using Benchmarker< mode, reference >::finalize;

	};

} // namespace grb

#endif // end ``_H_GRB_NONBLOCKING_BENCH''

