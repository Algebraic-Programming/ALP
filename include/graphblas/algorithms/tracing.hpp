
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
 * @file tracing.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Contains basic routines to trace the execution of an algorithm.
 * @date 2021-05-10
 */

#ifndef _H_GRB_ALGO_TRACING
#define _H_GRB_ALGO_TRACING

#include <graphblas.hpp>

namespace grb {
	namespace algorithms {
		namespace utils {

			/**
			 * @brief Prints the norm of the vector \p r and the header \p head.
			 *
			 * @tparam T type of vector values
			 * @tparam Ring ring for the multiplication and accumulation operator
			 *
			 * @param[in] r vector to compute the norm
			 * @param[in] head headerto print before the norm (if not \c nullptr)
			 * @param[in] ring ring with multiplication and accumulation
			 */
			template< typename T, class Ring = Semiring< grb::operators::add< T >, grb::operators::mul< T >, grb::identities::zero, grb::identities::one > >
			void print_norm( const grb::Vector< T > & r, const char * head, const Ring & ring = Ring() ) {
				T norm;
				RC ret = grb::dot( norm, r, r, ring ); // residual = r' * r;
				(void)ret;
				assert( ret == SUCCESS );
				std::cout << ">>> ";
				if( head != nullptr ) {
					std::cout << head << ": ";
				}
				std::cout << norm << std::endl;
			}

		} // namespace utils
	}     // namespace algorithms
} // namespace grb

#endif // _H_GRB_ALGO_TRACING
