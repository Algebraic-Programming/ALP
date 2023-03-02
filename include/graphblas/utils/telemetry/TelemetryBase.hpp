
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
 * @date 1st March, 2023
 */

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE

#include "TelemetryController.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			/**
			 *
			 *
			 * @tparam TelControllerType
			 * @tparam enabled
			 */
			template<
				typename TelControllerType,
				bool enabled = TelControllerType::enabled
			> class TelemetryBase {
			public:
				static_assert( is_telemetry_controller< TelControllerType >::value,
					"type TelControllerType does not implement Telemetry Controller interface" );

				using self_t = TelemetryBase< TelControllerType, enabled >;

				TelemetryBase() = default;

				TelemetryBase( const TelControllerType & tt ) {
					( void ) tt;
				}

				TelemetryBase( const self_t & ) = default;

				self_t & operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return false; }
			};


			template<
				typename TelControllerType
			> class TelemetryBase< TelControllerType, true > {

				const TelControllerType & telemetry_Controller;

			public:
				static_assert( is_telemetry_controller< TelControllerType >::value,
					"type TelControllerType does not implement Telemetry Controller interface" );

				using self_t = TelemetryBase< TelControllerType, true >;

				TelemetryBase( const TelControllerType & tt ): telemetry_Controller( tt ) {}

				TelemetryBase( const self_t & tb ) : telemetry_Controller( tb.telemetry_Controller ) {}

				self_t & operator=( const self_t & ) = delete;

				bool is_active() const { return telemetry_Controller.is_active(); }
			};

			// always actibe base, especially for prototyping scenarios
			template<> class TelemetryBase< TelemetryControllerAlwaysOn, true > {
			public:
				static_assert( is_telemetry_controller< TelemetryControllerAlwaysOn >::value,
					"type TelControllerType does not implement Telemetry Controller interface" );

				using self_t = TelemetryBase< TelemetryControllerAlwaysOn, true >;

				TelemetryBase( const TelemetryControllerAlwaysOn & tt ) { (void) tt; }

				TelemetryBase( const self_t & tb ) = default;

				self_t & operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return true; }
			};

		}
	}
}

#endif // _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE
