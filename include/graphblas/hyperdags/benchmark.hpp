
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
 * Provides the Benchmarker for the HyperDAGs backend
 *
 * @author A. Karanasiou
 * @date 11th of May, 2022
 */

#ifndef _H_GRB_HYPERDAGS_BENCH
#define _H_GRB_HYPERDAGS_BENCH

#include <graphblas/rc.hpp>
#include <graphblas/base/benchmark.hpp>


namespace grb {

	/** \internal Simply wraps around the underlying Benchmarker implementation. */
	template< enum EXEC_MODE mode >
	class Benchmarker< mode, hyperdags > :
		public Benchmarker< mode, _GRB_WITH_HYPERDAGS_USING >
	{
		public:

			typedef Benchmarker< mode, _GRB_WITH_HYPERDAGS_USING > MyBenchmarkerType;

			/** \internal Delegates to #grb::Launcher (reference) constructor. */
			using MyBenchmarkerType::Benchmarker;

			using MyBenchmarkerType::finalize;
	};

} // namespace grb

#endif // end ``_H_GRB_HYPERDAGS_BENCH''

