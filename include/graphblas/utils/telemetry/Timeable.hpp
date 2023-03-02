
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

#ifndef _H_GRB_UTILS_TIMEABLE
#define _H_GRB_UTILS_TIMEABLE

#include "Stopwatch.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			template<
				typename TelControllerType,
				bool enabled = TelControllerType::enabled
			> class Timeable {
			public:
				using self_t = Timeable< TelControllerType, enabled >;

				Timeable( const TelControllerType & tt ) {
					(void) tt;
				}

				Timeable( const self_t & ) = default;

				Timeable& operator=( const self_t & ) = delete;

				constexpr inline duration_nano_t getElapsedNano() const {
					return static_cast< duration_nano_t >( 0 );
				}

				constexpr inline duration_nano_t reset() {
					return static_cast< duration_nano_t >( 0 );
				}

			protected:
				inline void start() {}

				constexpr inline duration_nano_t stop() {
					return static_cast< duration_nano_t >( 0 );
				}

			};

			template< typename TelControllerType > class Timeable< TelControllerType, true > {
			public:
				using self_t = Timeable< TelControllerType, true >;

				Timeable( const TelControllerType & tt ) : swatch( tt ) {}

				Timeable( const self_t & ) = default;

				Timeable& operator=( const self_t & ) = delete;

				inline duration_nano_t getElapsedNano() const {
					return swatch.getElapsedNano();
				}

				inline duration_nano_t reset() {
					return swatch.reset();
				}

			protected:
				inline void start() {
					swatch.start();
				}

				inline duration_nano_t stop() {
					return swatch.stop();
				}

			private:
				Stopwatch< TelControllerType > swatch;
			};

			using StaticTimeable = Timeable< TelemetryControllerAlwaysOn, true >;

		}
	}
}

#endif // _H_GRB_UTILS_TIMEABLE
