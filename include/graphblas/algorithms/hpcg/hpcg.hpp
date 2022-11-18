
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

#ifndef _H_GRB_ALGORITHMS_HPCG_HPCG
#define _H_GRB_ALGORITHMS_HPCG_HPCG

#include <utility>

#include <graphblas/algorithms/multigrid/red_black_gauss_seidel.hpp>
#include <graphblas/algorithms/multigrid/coarsener.hpp>
#include <graphblas/algorithms/multigrid/multigrid_v_cycle.hpp>
#include <graphblas/algorithms/multigrid/multigrid_cg.hpp>

namespace grb {
	namespace algorithms {

		// simply "assemble" types
		template<
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring,
			class Minus
		> using HPCGRunnerType = mg_cg_runner< IOType, NonzeroType, InputType, ResidualType,
			multigrid_runner< IOType, NonzeroType, InputType,
				red_black_smoother_runner< IOType, NonzeroType, Ring >,
				single_point_coarsener< IOType, NonzeroType, Ring, Minus >,
				Ring, Minus >,
			Ring, Minus
		>;

		template<
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring,
			class Minus
		> HPCGRunnerType< IOType, ResidualType, NonzeroType, InputType, Ring, Minus >
			build_hpcg_runner( size_t smoother_steps ) {

			single_point_coarsener< IOType, NonzeroType, Ring, Minus > coarsener;
			red_black_smoother_runner< IOType, NonzeroType, Ring >
				smoother{ smoother_steps, smoother_steps, 1UL, {}, Ring() };

			multigrid_runner< IOType, NonzeroType, InputType,
				red_black_smoother_runner< IOType, NonzeroType, Ring >,
				single_point_coarsener< IOType, NonzeroType, Ring, Minus >,
				Ring, Minus
			> mg_runner( std::move( smoother ), std::move( coarsener ) );

			return HPCGRunnerType< IOType, ResidualType, NonzeroType, InputType, Ring, Minus >(
				std::move( mg_runner ) );
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_HPCG
