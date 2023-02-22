
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

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE

#include "TelemetryToken.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			template<
				typename TelTokenType,
				bool enabled = TelTokenType::enabled
			> class TelemetryBase {
			public:
				static_assert( is_telemetry_token< TelTokenType >::value,
					"type TelTokenType does not implement Telemetry Token interface" );

				using self_t = TelemetryBase< TelTokenType, enabled >;

				TelemetryBase() = default;

				TelemetryBase( const TelTokenType & tt ) {
					( void ) tt;
				}

				TelemetryBase( const self_t & ) = default;

				self_t & operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return false; }
			};


			template<
				typename TelTokenType
			> class TelemetryBase< TelTokenType, true > {

				const TelTokenType & telemetry_token;

			public:
				static_assert( is_telemetry_token< TelTokenType >::value,
					"type TelTokenType does not implement Telemetry Token interface" );

				using self_t = TelemetryBase< TelTokenType, true >;

				TelemetryBase( const TelTokenType & tt ): telemetry_token( tt ) {}

				TelemetryBase( const self_t & tb ) : telemetry_token( tb.telemetry_token ) {}

				self_t & operator=( const self_t & ) = delete;

				bool is_active() const { return telemetry_token.is_active(); }
			};

			// always actibe base, especially for prototyping scenarios
			template<> class TelemetryBase< TelemetryTokenAlwaysOn, true > {
			public:
				static_assert( is_telemetry_token< TelemetryTokenAlwaysOn >::value,
					"type TelTokenType does not implement Telemetry Token interface" );

				using self_t = TelemetryBase< TelemetryTokenAlwaysOn, true >;

				TelemetryBase( const TelemetryTokenAlwaysOn & tt ) { (void) tt; }

				TelemetryBase( const self_t & tb ) = default;

				self_t & operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return true; }
			};

		}
	}
}

#endif // _H_GRB_UTILS_TELEMETRY_TELEMETRY_BASE
