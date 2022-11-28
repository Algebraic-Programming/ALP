
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

/**
 * @dir include/graphblas/algorithms/hpcg
 * This folder contains the code specific to the HPCG benchmark implementation: generation of the physical system,
 * generation of the single point coarsener and coloring algorithm.
 */

/**
 * @file hpcg.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * Utility to build a full HPCG runner, bringing together all needed data structures.
 */

#ifndef _H_GRB_ALGORITHMS_HPCG_HPCG
#define _H_GRB_ALGORITHMS_HPCG_HPCG

#include <utility>

#include <graphblas/algorithms/multigrid/red_black_gauss_seidel.hpp>
#include <graphblas/algorithms/multigrid/single_matrix_coarsener.hpp>
#include <graphblas/algorithms/multigrid/multigrid_v_cycle.hpp>
#include <graphblas/algorithms/multigrid/multigrid_cg.hpp>

namespace grb {
	namespace algorithms {

		// simply "assemble" types
		template<
			Descriptor descr,
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring,
			class Minus
		> using HPCGRunnerType = MultiGridCGRunner< IOType, NonzeroType, InputType, ResidualType,
			MultiGridRunner<
				RedBlackGSSmootherRunner< IOType, NonzeroType, Ring, descr >,
				SingleMatrixCoarsener< IOType, NonzeroType, Ring, Minus, descr >,
				IOType, NonzeroType, Ring, Minus, descr
			>, Ring, Minus, descr
		>;

		/**
		 * Builds a full HPCG runner object by "assemblying" all needed information,
		 * with default type for smoother, coarsener and multi-grid runner.
		 *
		 * @param[in] smoother_steps how many times the smoother should run (both pre- and post-smoothing)
		 */
		template<
			Descriptor descr,
			typename IOType,
			typename ResidualType,
			typename NonzeroType,
			typename InputType,
			class Ring,
			class Minus
		> HPCGRunnerType< descr, IOType, ResidualType, NonzeroType, InputType, Ring, Minus >
			build_hpcg_runner( size_t smoother_steps ) {

			SingleMatrixCoarsener< IOType, NonzeroType, Ring, Minus, descr > coarsener;
			RedBlackGSSmootherRunner< IOType, NonzeroType, Ring, descr >
				smoother( { smoother_steps, smoother_steps, 1UL, {}, Ring() } );

			MultiGridRunner<
				RedBlackGSSmootherRunner< IOType, NonzeroType, Ring, descr >,
				SingleMatrixCoarsener< IOType, NonzeroType, Ring, Minus, descr >,
				IOType, NonzeroType, Ring, Minus, descr
			> mg_runner( std::move( smoother ), std::move( coarsener ) );

			return HPCGRunnerType< descr, IOType, ResidualType, NonzeroType, InputType, Ring, Minus >(
				std::move( mg_runner ) );
		}

	} // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGORITHMS_HPCG_HPCG
