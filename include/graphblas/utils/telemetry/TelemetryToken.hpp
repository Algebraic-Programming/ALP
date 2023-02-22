
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

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY_TOKEN
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY_TOKEN

#include <type_traits>
#include <utility>

namespace grb {
	namespace utils {
		namespace telemetry {

			template< typename T > constexpr bool is_token_enabled() { return false; }

			// OFF
			template< bool en > class TelemetryTokenBase {
			public:
				using self_t = TelemetryTokenBase< en >;

				TelemetryTokenBase( bool _enabled ) {
					(void) _enabled;
				}

				TelemetryTokenBase() = delete;

				TelemetryTokenBase( const self_t & ) = delete;

				TelemetryTokenBase& operator=( const self_t & ) = delete;

				constexpr bool is_active() const { return false; }

				static constexpr bool enabled = false;
			};

			using TelemetryTokenAlwaysOff = TelemetryTokenBase< false >;

			template<> class TelemetryTokenBase< true > {
			public:
				using self_t = TelemetryTokenBase< true >;

				TelemetryTokenBase( bool _active ) : active( _active ) {}

				TelemetryTokenBase() = delete;

				TelemetryTokenBase( const self_t & ) = delete;

				TelemetryTokenBase& operator=( const self_t & ) = delete;

				bool is_active() const { return this->active; }

				static constexpr bool enabled = true;

			protected:
				const bool active;
			};

			// always active token, especially for prototyping scenarios
			class TelemetryTokenAlwaysOn {
			public:
				TelemetryTokenAlwaysOn( bool _enabled ) {
					(void) _enabled;
				}

				TelemetryTokenAlwaysOn() = delete;

				TelemetryTokenAlwaysOn( const TelemetryTokenAlwaysOn & ) = delete;

				TelemetryTokenAlwaysOn& operator=( const TelemetryTokenAlwaysOn & ) = delete;

				constexpr bool is_active() const { return true; }

				static constexpr bool enabled = true;
			};


			template< typename T > struct is_telemetry_token {
			private:
				template< typename U > static constexpr bool has_enabled_field(
					typename std::enable_if<
						std::is_same< typename std::decay< decltype( U::enabled ) >::type, bool >::value,
							bool * >::type
					) {
						return true;
					}

				template< typename U > static constexpr bool has_enabled_field( ... ) { return false; }

				template< typename U > static constexpr bool has_is_active_method(
					typename std::enable_if<
						std::is_same< typename std::decay< decltype( std::declval< U >().is_active() ) >::type, bool >::value,
						bool * >::type
				) {
					return true;
				}

				template< typename U > static constexpr bool has_is_active_method( ... ) { return false; }

			public:
				static constexpr bool value = has_enabled_field< T >( nullptr ) && has_is_active_method< T >( nullptr );
			};
		}

	}
}

#define __TELEMETRY_TOKEN_ENABLER_NAME( name ) __ ## name ## Enabler
#define __TELEMETRY_TOKEN_NAME( name ) name

#define DECLARE_TELEMETRY_TOKEN( name ) 																			\
	class __TELEMETRY_TOKEN_ENABLER_NAME( name ) {};																\
	template< typename T > class __TELEMETRY_TOKEN_NAME( name ) :													\
		public grb::utils::telemetry::TelemetryTokenBase< grb::utils::telemetry::is_token_enabled< T >() > {		\
	public:																											\
		using base_t = grb::utils::telemetry::TelemetryTokenBase< grb::utils::telemetry::is_token_enabled< T >() >;	\
		__TELEMETRY_TOKEN_NAME( name )( bool _enabled ) : base_t( _enabled ) {}										\
	};


#define ACTIVATE_TOKEN( name ) namespace grb { namespace utils { namespace telemetry {					\
	template<> constexpr bool is_token_enabled< __TELEMETRY_TOKEN_ENABLER_NAME( name ) >() { return true; } \
} } }

#define TELEMETRY_TOKEN_TYPE( name ) __TELEMETRY_TOKEN_NAME( name )< __TELEMETRY_TOKEN_ENABLER_NAME( name ) >

#endif // _H_GRB_UTILS_TELEMETRY_TELEMETRY_TOKEN
