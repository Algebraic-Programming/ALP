
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
 * @file TelemetryBase.hpp
 * @author Alberto Scolari (alberto.scolar@huawei.com)
 *
 * Definition for the TelemetryBase class.
 */

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE

#include "TelemetryController.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			/**
			 * Base class provided as a convenience, exposing whether the telemetry is active.
			 *
			 * Default contruction is unavailable, because telemetry functionalities need an
			 * underlying telemetry controller to know whether they are enabled and active.
			 *
			 * Instead, copy construction is available for inheriting classes to easily implement copy semantics
			 * if needed; the copy shares the same telemetry controller of the original object via a reference.
			 *
			 * This implementation corresponds to enabled telemetry and stores an actual
			 * telemetry controller at runtime to be notified about its active state.
			 *
			 * @tparam TelControllerType type of the underlying telemetry controller,
			 * 	usually derived from TelemetryControllerBase
			 * @tparam enabled whther the current type is enabled (usually equals to TelControllerType::enabled)
			 */
			template<
				typename TelControllerType,
				bool enabled = TelControllerType::enabled
			> class TelemetryBase {

				const TelControllerType & telemetry_Controller;

			public:
				static_assert( is_telemetry_controller< TelControllerType >::value,
					"type TelControllerType does not implement Telemetry Controller interface" );

				using self_t = TelemetryBase< TelControllerType, enabled >;

				TelemetryBase( const TelControllerType & tt ): telemetry_Controller( tt ) {}

				TelemetryBase( const self_t & tb ) : telemetry_Controller( tb.telemetry_Controller ) {}

				self_t & operator=( const self_t & ) = delete;

				bool is_active() const { return telemetry_Controller.is_active(); }
			};

			/**
			 * Template specialization for disabled telemetry: no state, no activity.
			 *
			 * @tparam TelControllerType
			 */
			template <
				typename TelControllerType
			> class TelemetryBase< TelControllerType, false > {
			public:
				static_assert( is_telemetry_controller< TelControllerType >::value,
					"type TelControllerType does not implement Telemetry Controller interface" );

				using self_t = TelemetryBase< TelControllerType, false >;

				TelemetryBase() = default;

				TelemetryBase( const TelControllerType & ) {}

				TelemetryBase( const self_t & ) = default;

				self_t & operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return false; }
			};

			/**
			 * Specialization of TelemetryControllerBase for enabled and always active telemetry,
			 * mainly for debugging purposes: it is always active.
			 *
			 * For API compliance, it accepts an always-on telemetry controller, but does not store it.
			 */
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
