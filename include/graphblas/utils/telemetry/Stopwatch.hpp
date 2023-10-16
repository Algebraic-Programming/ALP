
/*
 *   Copyright 2023 Huawei Technologies Co., Ltd.
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
 * @file Stopwatch.hpp
 * @author Alberto Scolari (alberto.scolar@huawei.com)
 *
 * Definition for the Stopwatch class.
 */

#ifndef _H_GRB_UTILS_TELEMETRY_STOPWATCH
#define _H_GRB_UTILS_TELEMETRY_STOPWATCH

#include <chrono>

#include "TelemetryBase.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			/**
			 * Type to store time duration in nanoseconds, which is the default time granularity.
			 */
			using duration_nano_t = size_t;

			/**
			 * Duration as floating point type, for time granularities coarser than nanoseconds.
			 */
			using duration_float_t = double;

			/**
			 * Base class for Stopwatch, with common logic.
			 */
			class StopwatchBase {
			public:

				/**
				 * Convert nanoseconds to microseconds, returned as floating point type duration_float_t.
				 */
				static inline duration_float_t nano2Micro( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000UL;
				}

				/**
				 * Convert nanoseconds to milliseconds, returned as floating point type duration_float_t.
				 */
				static inline duration_float_t nano2Milli( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000000UL;
				}

				/**
				 * Convert nanoseconds to seconds, returned as floating point type duration_float_t.
				 */
				static inline duration_float_t nano2Sec( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000000000UL;
				}
			};

			/**
			 * Class with functionalities to measure elapsed time for telemetry purposes: start, stop, reset.
			 *
			 * The time granularity is nanoseconds.
			 *
			 * Copy semantics is not available.
			 *
			 * This implementation assumes telemetry is enabled and the active state is controlled via
			 * a telemetry controller of type \p TelControllerType.
			 *
			 * @tparam TelControllerType underlying telemetry controller type
			 * @tparam enabled whether it is compile-time enabled
			 */
			template<
				typename TelControllerType,
				bool enabled = TelControllerType::enabled
			> class Stopwatch :
				public StopwatchBase, public TelemetryBase< TelControllerType, enabled > {

				typedef typename std::chrono::high_resolution_clock clock_t;

				typedef typename std::chrono::nanoseconds duration_t;

				typedef typename std::chrono::high_resolution_clock::time_point time_point_t;

				duration_t elapsedTime; ///< measured elapsed time so far, i.e.,
				///< accumulated time periods between successive calls to #start() and #stop()

				time_point_t beginning; ///< time instant of last call to #start()

			public:
				/**
				 * Construct a new Stopwatch object from a telemetry controller.
				 *
				 * @param tt underlying telemetry controller, to be (de)activated at runtime
				 */
				Stopwatch( const TelControllerType & tt ) :
					StopwatchBase(),
					TelemetryBase< TelControllerType, true >( tt ),
					elapsedTime( duration_t::zero() ) {}

				Stopwatch( const Stopwatch< TelControllerType, enabled > &  ) = delete;

				/**
				 * Start measuring time.
				 *
				 * Subsequent calls to this method "reset" the measure of elapsed time: if the user calls #start()
				 * twice and then #stop(), the elapsed time accumulated internally after the call to #stop() is
				 * the time elapsed from the \b second call of #start() to the call to #stop().
				 */
				inline void start() {
					if( this->is_active() ) {
						beginning = clock_t::now();
					}
				}

				/**
				 * Stops time measurement, returning the elapsed time since the last #start() invocation.
				 * Elapsed time is internally accounted only if this method is invoked.
				 */
				inline duration_nano_t stop() {
					duration_nano_t count = 0;
					if( this->is_active() ) {
						time_point_t end = clock_t::now();
						duration_t d = end - beginning;
						count = d.count();
						elapsedTime += d;
					}
					return count;
				}

				/**
				 * Returns the elapsed time, which is accounted \b only if #stop() is called.
				 *
				 * The value of the elapsed time is not erased, so that successive calls return
				 * the same value.
				 */
				inline duration_nano_t getElapsedNano() const {
					return static_cast< duration_nano_t >( elapsedTime.count() );
				}

				/**
				 * To be called on a stopped watch, it returns the elapsed time and sets it to 0.
				 */
				inline duration_nano_t reset() {
					duration_nano_t r = getElapsedNano();
					if( this->is_active() ) {
						elapsedTime = duration_t::zero();
					}
					return r;
				}

				/**
				 * Stops the watch, sets the elapsed time to 0, starts it again
				 * and returns the time elapsed between the previous #start()
				 * and the #stop() internally called.
				*/
				inline duration_nano_t restart() {
					stop();
					duration_nano_t r = reset();
					start();
					return r;
				}
			};

			/**
			 * Template specialization of Stopwatch<TelControllerType, enabled> for disabled telemetry:
			 * no state is stored, all functions are inactive.
			 */
			template<
				typename TelControllerType
			> class Stopwatch< TelControllerType, false > :
				public StopwatchBase, public TelemetryBase< TelControllerType, false > {
			public:
				Stopwatch( const TelControllerType & tt ) :
					StopwatchBase(),
					TelemetryBase< TelControllerType, false >( tt ) {}

				Stopwatch( const Stopwatch< TelControllerType, false > & ) = delete;

				constexpr inline void start() {}

				constexpr inline duration_nano_t stop() {
					return static_cast< duration_nano_t >( 0 );
				}

				constexpr inline duration_nano_t getElapsedNano() const {
					return static_cast< duration_nano_t >( 0 );
				}

				constexpr inline duration_nano_t reset() {
					return static_cast< duration_nano_t >( 0 );

				}

				constexpr inline duration_nano_t restart() {
					return static_cast< duration_nano_t >( 0 );
				}

			};

			/**
			 * Always active stopwatch, requiring no telemetry controller for construction.
			 * Mainly for debugging purposes.
			 */
			class ActiveStopwatch : public Stopwatch< TelemetryControllerAlwaysOn, true > {
			public:

				using base_t = Stopwatch< TelemetryControllerAlwaysOn, true >;

				ActiveStopwatch():
					base_t( tt ),
					tt( true ) {}

				ActiveStopwatch( const ActiveStopwatch & ) = delete;

			private:
				TelemetryControllerAlwaysOn tt;
			};

		} // namespace telemetry
	}     // namespace utils
} // namespace grb

#endif // _H_GRB_UTILS_TELEMETRY_STOPWATCH
