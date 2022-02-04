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
 *
 * @file This file registers available index mapping functions (IMFs).
 *       IMFs are maps between integer intervals.
 */

#ifndef _H_GRB_IMF
#define _H_GRB_IMF

#include <memory>

namespace grb {

	namespace imf {

        class IMF {
            public:
                size_t n, N;

                IMF(size_t n, size_t N): n(n), N(N) {}

                virtual size_t map(size_t i) = 0;

        };

        /**
         * The identity IMF.  
         * I =[0, n)
         * Id = I -> I; i \mapsto i
         */
        class Id: public IMF {

            public:
                size_t map(size_t i) {
                    return i;
                }

                Id(size_t n): IMF(n, n) { }
        };

        /**
         * The strided IMF.  
         * I_n =[0, n), I_N =[0, N)
         * Strided_{b, s} = I_n -> I_N; i \mapsto b + s * i
         */
        class Strided: public IMF {

            public:
                size_t b, s;

                size_t map(size_t i) {

                    return b + s * i;
                }

                Strided(size_t n, size_t N, size_t b, size_t s): IMF(n, N), b(b), s(s) { }
        };

        /**
         * A composition of two IMFs f\circ g = f(g(\cdot))
         * I_gn =[0, n), I_fN =[0, m)
         * I_fn =[0, m), I_fN =[0, N)
         * Composed_{f, g} = I_gn -> I_fN; i \mapsto f( g( i ) )
         */
        class Composed: public IMF {

            public:
                std::shared_ptr<IMF> f, g;

                size_t map(size_t i) {
                    return f->map( g->map( i ) );
                }

                Composed(std::shared_ptr<IMF> f, std::shared_ptr<IMF> g): IMF( g->n, f->N ), f(f), g(g) { }

        };

	}; // namespace imf

} // namespace grb

#endif // _H_GRB_IMF
