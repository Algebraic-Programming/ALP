
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

#ifndef _H_GRB_TIMER
#define _H_GRB_TIMER

#include <time.h> //clock_gettime, CLOCK_MONOTONIC, struct timespec
#include <assert.h>

namespace grb {
	namespace utils {

		/**
		 * A class for high-precision timers.
		 *
		 * Can handle parallel codes. Does not support multiple threads calling the
		 * same instance, however-- this class is not thread-safe. The intended way
		 * to use this class is from a Single Program, Multiple Data perspective:
		 * every thread operates on its own Timer instance.
		 *
		 * This implementation depends on the POSIX Realtime extension for high
		 * precision resolutions.
		 */
		class Timer {

		private:
			/** The POSIX Realtime timer to use. */
			static constexpr clockid_t clock_id = CLOCK_MONOTONIC;

			/** Containts the start time of this timer. */
			struct timespec start;

		public:
			/**
			 * Creates a new timer and sets the start point to now.
			 *
			 * The resolution is of course affected by the time taken for this call,
			 * which, in turn, depends on your implementation of the POSIX Realtime.
			 */
			Timer() {
				const int rc = clock_gettime( clock_id, &start );
#ifdef NDEBUG
				(void)rc;
#else
				assert( rc == 0 );
#endif
			}

			/**
			 * Reset the timer start point to now.
			 *
			 * The resolution is of course affected by the time taken for this call,
			 * which, in turn, depends on your implementation of the POSIX Realtime.
			 */
			void reset() {
				const int rc = clock_gettime( clock_id, &start );
#ifdef NDEBUG
				(void)rc;
#else
				assert( rc == 0 );
#endif
			}

			/**
			 * @returns Elapsed time since timer start in milliseconds.
			 *
			 * The timer start time can be modified in two ways:
			 * @see Timer()
			 * @see reset()
			 */
			double time() {
				struct timespec stop;
				const int rc = clock_gettime( clock_id, &stop );
#ifdef NDEBUG
				(void)rc;
#else
				assert( rc == 0 );
#endif
				double ret = ( stop.tv_sec - start.tv_sec ) * 1000.0;
				ret += ( stop.tv_nsec - start.tv_nsec ) / 1000000.0;
				return ret;
			}
		}; // end Timer

	} // namespace utils
} // namespace grb

#endif // end `_H_GRB_TIMER'
