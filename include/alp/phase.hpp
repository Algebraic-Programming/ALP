
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
 */

#ifndef _H_ALP_PHASE
#define _H_ALP_PHASE

namespace alp {

	/**
	 * Some primitives may require a symbolic phase prior to executing a numeric
	 * phase. The symbolic phase may require system calls in order to, for example,
	 * reallocate storage to account for fill-in.
	 *
	 * For vectors, the user usually is able to pass in a reasonable upper bound on
	 * the number of nonzeroes and as * such, level-1 and level-2 primitives need
	 * not rely on a symbolic phase. For matrices that act as output on level-3
	 * primitives, however, it is instead far more common to not know of a
	 * reasonable upper bound beforehand; in these cases the use of a symbolic
	 * phase usually cannot be avoided.
	 *
	 * The performance semantics of primitives, which often do not allow system
	 * calls, are guaranteed only for numeric phases.
	 */
	enum PHASE {

		/**
		 * Simulates the operation with the sole purpose of determining the number of
		 * nonzeroes that the output container should hold. If this should be higher
		 * than the current capacity, then the output container will be reallocated.
		 *
		 * This means the performance costs increase with sum of the container
		 * dimensions plus the number of output nonzeroes, both in terms of work and
		 * data movement, whenever the call must reallocate. In that case it will also
		 * make system calls.
		 */
		SYMBOLIC,

		/**
		 * With the numerical phase, the user guarantees (all) output container(s)
		 * have enough capacity-- including for any newly materialised nonzeroes.
		 * The user may either give this guarantee through her knowledge of the
		 * overall computation (e.g., in a Conjugate Gradient solver for linear
		 * systems the vectors of length n will hold at most n nonzeroes), or may
		 * ensure sufficient capacity by first calling the primitive using a
		 * SYMBOLIC phase.
		 */
		NUMERICAL
	};

} // namespace alp

#endif // end ``_H_ALP_PHASE''
