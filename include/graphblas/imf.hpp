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
 * @file 
 * 
 * This file registers available index mapping functions (IMFs).
 * IMFs are maps between integer intervals and can be used to define
 * affine \em access transformations in the form of access matrices.
 * For example, an access matrix \f$G_f\in R^{N\times N}\f$ 
 * parametrized by the IMF \f$f\f$ such that
 * \f[G_f = \sum_{i=0}^{n-1} e_i^n\left(e_{f(i)}^N\right)^T\f]
 * could be used to access a group of $n\eN$ rows of matrix 
 * \f$A\in R^{N\times N}\f$
 * according to \f$f\f$ by multiplying \f$A\f$ by \f$G_f\f$ from the left:
 * \f[\tilde{A} = G_f\cdot A,\quad \tilde{A}\in R^{n\times N}\f]
 *      
 * \note The idea of parametrized matrices to express matrix accesses at 
 *       a higher level of mathematical abstractions is inspired by the 
 *       SPIRAL literature (Franchetti et al. SPIRAL: Extreme Performance Portability. 
 *       http://spiral.net/doc/papers/08510983_Spiral_IEEE_Final.pdf). 
 *       Similar affine formulations are also used in the polyhedral 
 *       compilation literature to express concepts such as access
 *       relations.
 *       In this draft we use integer maps. A symbolic version of them could be 
 *       defined using external libraries such as the Integer Set Library (isl 
 *       \link https://libisl.sourceforge.io/).
 *       
 */

#ifndef _H_GRB_IMF
#define _H_GRB_IMF

#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>


namespace grb {

	namespace imf {

        class IMF {
            public:
                size_t n, N;

                IMF(size_t n, size_t N): n(n), N(N) {}

                virtual size_t map(size_t i) = 0;

                virtual bool isSame( const IMF & other ) const {
                    return typeid( *this ) == typeid( other ) && n == other.n && N == other.N;
                }

        };

        /**
         * The identity IMF.  
         * \f$I_n = [0, n)\f$
         * \f$Id = I_n \rightarrow I_n; i \mapsto i\f$
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
         * \f$I_n =[0, n), I_N =[0, N)\f$
         * \f$Strided_{b, s} = I_n \rightarrow I_N; i \mapsto b + si\f$
         */

        class Strided: public IMF {

            public:
                size_t b, s;

                size_t map(size_t i) {

                    return b + s * i;
                }

                Strided(size_t n, size_t N, size_t b, size_t s): IMF(n, N), b(b), s(s) { }

                virtual bool isSame( const IMF & other ) const {
                    return IMF::isSame( other )
                        && b == dynamic_cast< const Strided & >( other ).b
                        && s == dynamic_cast< const Strided & >( other ).s;
                }
        };

        class Select: public IMF {

            public:
                std::vector< size_t > select;

                size_t map(size_t i) {
                    return select.at( i );
                }

                Select(size_t N, std::vector< size_t > & select): IMF( select.size(), N ), select( select ) {
                    if ( *std::max_element( select.cbegin(), select.cend() ) >= N) {
                        throw std::runtime_error("IMF Select beyond range.");
                    }
                }

                virtual bool isSame( const IMF & other ) const {
                    return IMF::isSame( other )
                        && select == dynamic_cast< const Select & >( other ).select;
                }
        };

        /**
         * A composition of two IMFs.
         * \f$I_{g,n} =[0, n), I_{g,N} =[0, N)\f$
         * \f$I_{f,n} =[0, n), I_{f,N} =[0, N)\f$
         * \f$Composed_{f, g} = I_{g,n} \rightarrow I_{f,N}; i \mapsto f( g( i ) )\f$
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
