
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

#ifndef _H_GRB_UTILS_TELEMETRY_OUTPUT_STREAM
#define _H_GRB_UTILS_TELEMETRY_OUTPUT_STREAM

#include <ostream>
#include <type_traits>
#include <utility>
#include <functional>

#include "TelemetryBase.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			template< typename T > struct is_ostream_input {

				template< typename U > static constexpr bool is_input(
					typename std::enable_if< std::is_same<
						// this means that the expression std::cout << obj is valid, where obj is of type T
						decltype( std::declval< std::ostream& >() << std::declval< U >() ),
						std::ostream& >::value, nullptr_t >::type
				) {
					return true;
				}

				template< typename U > static constexpr bool is_input( ... ) {
					return false;
				}

				static constexpr bool value = is_input< T >( nullptr );
			};

			class OutputStreamLazy {
				constexpr char operator()() const { return '\0'; }
			};

			template<
				typename TelTokenType,
				bool enabled = TelTokenType::enabled
			> class OutputStream : public TelemetryBase< TelTokenType, enabled > {
			public:
				using self_t = OutputStream< TelTokenType, enabled >;

				OutputStream() = default;

				OutputStream( const TelTokenType & _tt, std::ostream & _out ) :
					TelemetryBase< TelTokenType, enabled >( _tt )
				{
					( void ) _out;
				}

				OutputStream( const self_t & _out ) = default;

				OutputStream & operator=( const self_t & _out ) = delete;

				template< typename T > inline typename std::enable_if<
					is_ostream_input< T >::value,
				self_t & >::type operator<<( T&& v ) {
					( void ) v;
					return *this;
				}

				inline self_t & operator<<( std::ostream& (*func)( std::ostream& ) ) {
					( void ) func;
					return *this;
				}

				template< class F > inline typename std::enable_if<
					is_ostream_input< decltype( std::declval< F >()() ) >::value
					&& std::is_base_of< OutputStreamLazy, F >::value,
				self_t & >::type operator<<( F&& fun ) {
					( void ) fun;
					return *this;
				}
			};

			template< typename TelTokenType > class OutputStream< TelTokenType, true > :
				public TelemetryBase< TelTokenType, true > {
			public:
				using self_t = OutputStream< TelTokenType, true >;

				using base_t = TelemetryBase< TelTokenType, true >;

				OutputStream( const TelTokenType & _tt, std::ostream & _out ) :
					TelemetryBase< TelTokenType, true >( _tt ),
					out( _out )
				{}

				OutputStream( const self_t & _outs ) = default;

				OutputStream & operator=( const self_t & _out ) = delete;

				template< typename T > inline typename std::enable_if<
					is_ostream_input< T >::value,
				self_t & >::type operator<<( T&& v ) {
					if ( this->is_active() ) {
						out << std::forward< T >( v );
					}
					return *this;
				}

				inline self_t & operator<<( std::ostream& (*func)( std::ostream& ) ) {
					if ( this->is_active() ) {
						out << func;
					}
					return *this;
				}

				template< class F > inline typename std::enable_if<
					is_ostream_input< decltype( std::declval< F >()() ) >::value
					&& std::is_base_of< OutputStreamLazy, F >::value,
				self_t & >::type operator<<( F&& fun ) {
					if ( this->is_active() ) {
						out << fun();
					}
					return *this;
				}

			private:
				std::ostream & out;
			};

			using OutputStreamOff = OutputStream< TelemetryTokenAlwaysOff, false >;

			using OutputStreamOn = OutputStream< TelemetryTokenAlwaysOn, true >;
		}
	}
}

#endif // _H_GRB_UTILS_TELEMETRY_OUTPUT_STREAM
