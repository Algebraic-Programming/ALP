
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
 * @date 8th of August, 2016
 */

#ifndef _H_GRB_TIMERRESULTS
#define _H_GRB_TIMERRESULTS


namespace grb {

	namespace utils {

		/**
		 * A structure holding benchmark timing results.
		 *
		 * It keeps track of initial io, a preamble time for setup, a useful time for
		 * actual processing, and a postamble time for cleaning up.
		 */
		struct TimerResults {
			double io;
			double preamble;
			double useful;
			double postamble;
			void set( double val ) {
				io = val;
				preamble = val;
				useful = val;
				postamble = val;
			}
			void accum( TimerResults &times ) {
				io += times.io;
				preamble += times.preamble;
				useful += times.useful;
				postamble += times.postamble;
			}
			void normalize( const size_t loops_in ) noexcept {
				const double loops = static_cast< double >( loops_in );
				io /= loops;
				preamble /= loops;
				useful /= loops;
				postamble /= loops;
			}
			void min( const TimerResults &times ) noexcept {
				io = ( times.io < io ) ? times.io : io;
				preamble = ( times.preamble < preamble ) ? times.preamble : preamble;
				useful = ( times.useful < useful ) ? times.useful : useful;
				postamble = ( times.postamble < postamble ) ? times.postamble : postamble;
			}
			void max( const TimerResults &times ) noexcept {
				io = ( times.io > io ) ? times.io : io;
				preamble = ( times.preamble > preamble ) ? times.preamble : preamble;
				useful = ( times.useful > useful ) ? times.useful : useful;
				postamble = ( times.postamble > postamble ) ? times.postamble : postamble;
			}
		};

	} // namespace utils

} // namespace grb

#endif // ``_H_GRB_TIMERRESULTS''

