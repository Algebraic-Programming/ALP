
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

/*
 * @author Alberto Scolari
 * @date 14th February, 2023
 */

#ifndef _H_GRB_UTILS_TELEMETRY_STOPWATCH
#define _H_GRB_UTILS_TELEMETRY_STOPWATCH

#include <chrono>

#include "TelemetryBase.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			using duration_nano_t = size_t;

			using duration_float_t = double;

			class StopwatchBase {
			public:
				static inline duration_float_t nano2Micro( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000UL;
				}

				static inline duration_float_t nano2Milli( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000000UL;
				}

				static inline duration_float_t nano2Sec( duration_nano_t nano ) {
					return static_cast< duration_float_t >( nano ) / 1000000000UL;
				}

			};

			template<
				typename TelTokenType,
				bool enabled = TelTokenType::enabled
			> class Stopwatch:
				public StopwatchBase, public TelemetryBase< TelTokenType, enabled > {
			public:
				Stopwatch( const TelTokenType & tt ) :
					StopwatchBase(),
					TelemetryBase< TelTokenType, enabled >( tt )
					{}

				Stopwatch( const Stopwatch & ) = default;

				constexpr inline void start() {}

				constexpr inline duration_nano_t stop() {
					return static_cast< duration_nano_t >( 0 );
				}

				constexpr inline duration_nano_t reset() {
					return static_cast< duration_nano_t >( 0 );
				}

				constexpr inline duration_nano_t getElapsedNano() const {
					return static_cast< duration_nano_t >( 0 );
				}
			};


			template<
				typename TelTokenType
			> class Stopwatch< TelTokenType, true >:
				public StopwatchBase, public TelemetryBase< TelTokenType, true > {

				typedef typename std::chrono::high_resolution_clock clock_t;

				typedef typename std::chrono::nanoseconds duration_t;

				typedef typename std::chrono::high_resolution_clock::time_point time_point_t;

				duration_t elapsedTime;

				time_point_t beginning;

			public:
				Stopwatch( const TelTokenType & tt ) :
					StopwatchBase(),
					TelemetryBase< TelTokenType, true >( tt ),
					elapsedTime( duration_t::zero() )
					{}

				Stopwatch( const Stopwatch & s ) = default;

				inline void start() {
					if ( this->is_active() ) {
						beginning = clock_t::now();
					}
				}

				inline duration_nano_t stop() {
					duration_nano_t count = 0;
					if ( this->is_active() ) {
						time_point_t end = clock_t::now();
						duration_t d = end - beginning;
						count = d.count();
						elapsedTime += d;
					}
					return count;
				}

				inline duration_nano_t reset() {
					duration_t r = duration_t::zero();
					if ( this->is_active() ) {
						r = elapsedTime;
						elapsedTime = duration_t::zero();
					}
					return static_cast< duration_nano_t >( r.count() );
				}

				inline duration_nano_t getElapsedNano() const {
					return static_cast< duration_nano_t >( elapsedTime.count() );
				}
			};

			using StaticStopwatch = Stopwatch< TelemetryTokenAlwaysOn, true >;
		}
	}
}

#endif // _H_GRB_UTILS_TELEMETRY_STOPWATCH
