
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
 * @author Alberto Scolari
 * @date 1st March, 2023
 *
 * This file defines the basic functionalities for <b>Telemetry Controllers</b>, i.e.,
 * objects that enable/disable telemetry at compile-time and runtime.
 *
 * A telemetry controller can be \b enabled (at compile-time) to produce the code for telemetry and must be
 * \b activated at runtime to emit actual telemetry information. Activation depends on runtime information
 * (e.g., user's input) and may change dynamically \a after the controller is instantiated.
 * If a controller is \b disabled, no code for compile-time is generated in any compliant telemetry functionality;
 * hence, any (de)activation of a disabled telemetry controller is simply ignored and produces no result.
 * In any case, the code must compile under all conditions, in order to avoid verbose
 * pre-processing \a #if conditions.
 *
 * A typical instantiation of a telemetry controller in a user's application looks as follows:
 *
 * \code{.cpp}
 * ENABLE_TELEMETRY_CONTROLLER( my_controller_t )
 * DEFINE_TELEMETRY_CONTROLLER( my_controller_t )
 *
 * int main() {
 * 		my_controller_t my_controller( true );
 * 		if( my_controller.is_active() ) {
 * 			std::cout << "my_controller is active";
 * 		} else {
 * 			std::cout << "my_controller is NOT active";
 * 			if( !my_controller_t::enabled ) {
 * 				std::cout << ", because it was deactivated at compile-time";
 * 			}
 * 		}
 * 		std::cout << std::endl;
 * 		return 0;
 * }
 * \endcode
 *
 * where the activation directive \a ENABLE_TELEMETRY_CONTROLLER is present only if the controller
 * is to be activated. Users should indeed comment/uncomment this directive do disable/enable telemetry
 * while debugging, or may add extra pre-processing logic to control it during compilation, like
 *
 * \code{.cpp}
 * #ifdef __I_WANT_my_controller_t_ENABLED__
 * 		ENABLE_TELEMETRY_CONTROLLER( my_controller_t )
 * #endif
 * DEFINE_TELEMETRY_CONTROLLER( my_controller_t )
 * \endcode
 *
 * Note that the \a ENABLE_TELEMETRY_CONTROLLER directive (if present) must come \b before the
 * \a DEFINE_TELEMETRY_CONTROLLER directive, otherwise compilation errors occur.
 */

#ifndef _H_GRB_UTILS_TELEMETRY_TELEMETRY_CONTROLLER
#define _H_GRB_UTILS_TELEMETRY_TELEMETRY_CONTROLLER

#include <type_traits>
#include <utility> // std::declval< T >()

namespace grb {
	namespace utils {
		namespace telemetry {

			/**
			 * Returns whether a telemetry controller is enabled <b>at compile-time</b>. By default
			 * it is \b not.
			 *
			 * @tparam T type associated to the telemetry controller
			 * @return true never
			 * @return false always
			 */
			template< typename T > constexpr bool is_controller_enabled() { return false; }

			/**
			 * Class that encapsulates the logic to enable/disable telemetry at compile-time
			 * or at runtime.
			 *
			 * Telemetry can be completely disabled at compile-time (e.g., to avoid any code generation
			 * and overhead) or can be controlled at runtime, based on external conditions (e.g.,
			 * user's input, cluster node number, ...).
			 *
			 * In the following, the field #enabled encodes the compile-time information, while
			 * the field \a active (if present) and the corresponding getter #is_active() tell
			 * whether the controller is \a active at runtime. Hence, users of telemetry controllers should always
			 * use the #is_active() method to check whether telemetry is active, while implementations
			 * of telemetry controllers should implement this method also based on the value of the #enabled
			 * field, possibly "short-circuiting" when #enabled is \a false. This implementation does
			 * exactly this, disabling telemetry at compile-time and ignoring any runtime information.
			 *
			 * @tparam en whether telemetry is enabled (\p en = \a true has a dedicated template specialization)
			 */
			template< bool en > class TelemetryControllerBase {
			public:
				using self_t = TelemetryControllerBase< en >;

				/**
				 * Construct a new Telemetry Controller Base object with runtime information.
				 *
				 * HEre, runtime information is ignored, as this implementation disables any telemetry.
				 *
				 * @param _enabled whether telemetry is runtime-enabled (ignored here)
				 */
				TelemetryControllerBase( bool _enabled ) {
					(void) _enabled;
				}

				TelemetryControllerBase() = delete;

				TelemetryControllerBase( const self_t & ) = delete;

				TelemetryControllerBase& operator=( const self_t & ) = delete;

				/**
				 * Whether telemetry is runtime-active.
				 *
				 * @return true never here
				 * @return false always
				 */
				constexpr bool inline is_active() const { return false; }

				/**
				 * Set the active status of the telemetry controller.
				 *
				 * This \a disabled implementation ignores the input \p _active.
				 */
				void inline set_active( bool _active ) {
					( void ) _active;
				}

				/**
				 * Whether telemetry is compile-time active (never here).
				 */
				static constexpr bool enabled = false;
			};

			/**
			 * Convenience definition fo an always-off telemetry controller.
			 */
			using TelemetryControllerAlwaysOff = TelemetryControllerBase< false >;

			/**
			 * Template specialization for compile-time enabled telemetry, which
			 * can be controlled at runtime.
			 *
			 * The controller is \b enabled by default, and its \a active status can be controlled
			 * at runtime via the constructor and the #set_active(bool) method.
			 */
			template<> class TelemetryControllerBase< true > {
			public:
				using self_t = TelemetryControllerBase< true >;

				/**
				 * Construct a new Telemetry oCntroller Base object, specifying the \a active state.
				 *
				 * @param _active whether the controller is \a active or not
				 */
				TelemetryControllerBase( bool _active ) : active( _active ) {}

				TelemetryControllerBase() = delete;

				TelemetryControllerBase( const self_t & ) = default;

				TelemetryControllerBase& operator=( const self_t & ) = delete;

				/**
				 * Tells whether the controller is \a active.
				*/
				bool is_active() const { return this->active; }

				/**
				 * Set the \a active status of the controller at runtime.
				 *
				 * @param _active whether to activate the controller
				 */
				void inline set_active( bool _active ) {
					this->active = _active;
				}

				/**
				 * Whether telemetry is compile-time active (here always).
				*/
				static constexpr bool enabled = true;

			protected:
				bool active;
			};

			/**
			 * Always active controller, useful especially for prototyping scenarios.
			 */
			class TelemetryControllerAlwaysOn {
			public:
				TelemetryControllerAlwaysOn( bool _enabled ) {
					(void) _enabled;
				}

				TelemetryControllerAlwaysOn() = default;

				TelemetryControllerAlwaysOn( const TelemetryControllerAlwaysOn & ) = default;

				TelemetryControllerAlwaysOn& operator=( const TelemetryControllerAlwaysOn & ) = delete;

				/**
				 * Tells whether the controller is \a active, which is in this case always true.
				*/
				constexpr bool is_active() const { return true; }

				/**
				 * Set the active status of the telemetry controller.
				 *
				 * This \a disabled implementation ignores the input \p _active.
				 */
				void inline set_active( bool _active ) {
					( void ) _active;
				}

				/**
				 * Whether telemetry is compile-time active (here always).
				 */
				static constexpr bool enabled = true;
			};

			/**
			 * SFINAE-based structure to check whether \p T is a telemetry controller, i.e.
			 *   - it has a \a constexpr static field named \a enabled
			 *   - it has an \a is_active() method
			 *   - it has a \a set_active(bool) method
			 */
			template< typename T > struct is_telemetry_controller {
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
						std::is_same< typename std::decay<decltype( std::declval< U >().is_active() )
							>::type, bool >::value, bool * >::type
				) {
					return true;
				}

				template< typename U > static constexpr bool has_is_active_method( ... ) { return false; }

				template< typename U > static constexpr bool has_set_active_method(
					typename std::enable_if<
						std::is_same< decltype( std::declval< U >().set_active( true ) ), void >::value,
						bool * >::type
				) {
					return true;
				}

				template< typename U > static constexpr bool has_set_active_method( ... ) { return false; }

			public:
				static constexpr bool value = has_enabled_field< T >( nullptr )
					&& has_is_active_method< T >( nullptr ) && has_set_active_method< T >( nullptr ) ;
			};
		}

	}
}

// Name of the Controller Enabler, i.e., a type that controls whether a telemetry controller is enabled
#define __TELEMETRY_CONTROLLER_ENABLER_NAME( name ) __ ## name ## _Enabler

// Name of the Telemetry Controller type
#define __TELEMETRY_CONTROLLER_NAME( name ) name ## _cls

/**
 * Defines a telemetry controller, i.e., a custom type derived from TelemetryControllerBase.
 *
 * This declaration requires the declaration of an associated controller enabler type, which controls
 * whether the controller is enabled at compile-time; the controller is by default \b deactivated.
 */
#define DEFINE_TELEMETRY_CONTROLLER( name ) 																\
	class __TELEMETRY_CONTROLLER_ENABLER_NAME( name ) {};												\
	using name = class __TELEMETRY_CONTROLLER_NAME( name ) :												\
		public grb::utils::telemetry::TelemetryControllerBase<											\
			grb::utils::telemetry::is_controller_enabled< __TELEMETRY_CONTROLLER_ENABLER_NAME( name ) >() > {	\
	public:																							\
		using base_t = grb::utils::telemetry::TelemetryControllerBase<									\
			grb::utils::telemetry::is_controller_enabled< __TELEMETRY_CONTROLLER_ENABLER_NAME( name ) >() >;	\
		__TELEMETRY_CONTROLLER_NAME( name )( bool _enabled ) : base_t( _enabled ) {}						\
	};

/**
 * Enables a telemetry controller through its associated enabler type.
 *
 * Once enabled, it can be runtime activated.
 */
#define ENABLE_TELEMETRY_CONTROLLER( name ) class __TELEMETRY_CONTROLLER_ENABLER_NAME( name );	\
	namespace grb { namespace utils { namespace telemetry {						\
		template<> constexpr bool is_controller_enabled<								\
			__TELEMETRY_CONTROLLER_ENABLER_NAME( name ) >() { return true; } 		\
	} } }

#endif // _H_GRB_UTILS_TELEMETRY_TELEMETRY_CONTROLLER
