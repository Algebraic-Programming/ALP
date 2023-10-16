
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
 * @file OutputStream.hpp
 * @author Alberto Scolari (alberto.scolar@huawei.com)
 *
 * Definition for the OutputStream class.
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

			/**
			 * SFINAE-based class to check whether the type \p T can be input to an std::ostream
			 * via the \a << operator.
			 */
			template< typename T > struct is_ostream_input {
			private:

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

			public:
				static constexpr bool value = is_input< T >( nullptr );
			};

			/**
			 * Telemetry-controllable output stream with basic interface, based on the \a << operator.
			 *
			 * It accepts in input any type \a std::ostream accepts. In addition, it also accepts
			 * the internl #OutputStreamLazy<RetType> type, which marks callable objects and allows
			 * lazy evaluation of their result if the telemetry is active; if not, the object is
			 * not called, avoiding runtime costs. This functionality allows paying time and memory
			 * costs of computation only if really needed.
			 *
			 * @tparam TelControllerType type of the telemetry controller
			 * @tparam enabled whether telemetry is enabled for this type
			 */
			template<
				typename TelControllerType,
				bool enabled = TelControllerType::enabled
			> class OutputStream : public TelemetryBase< TelControllerType, enabled > {
			public:
				using self_t = OutputStream< TelControllerType, enabled >;

				using base_t = TelemetryBase< TelControllerType, enabled >;

				/**
				 * Marker object to indicate that the stored callable object is to be called
				 * in a lazy way, i.e., only if output is active.
				 *
				 * @tparam RetType return type of the collable object, to be printed
				 */
				template< typename RetType > class OutputStreamLazy {

					const std::function< RetType() > f;

				public:
					static_assert( is_ostream_input< RetType >::value );

					template< class F > OutputStreamLazy( F&& _f ) : f( std::forward< F >( _f ) ) {}

					RetType operator()() const { return f(); }
				};

				/**
				 * Convenience function to create an #OutputStreamLazy<RetType> object from
				 * a callable one, inferring all template parameters automatically.
				 *
				 * @tparam CallableType type of the given callable object
				 * @tparam RetType return type of the callable object, to be printed
				 * @param f callable object
				 * @return OutputStreamLazy< RetType > object marking lazy evaluation for printing
				 */
				template<
					typename CallableType,
					typename RetType = decltype( std::declval< CallableType >()() )
				> static OutputStreamLazy< RetType > makeLazy( CallableType&& f ) {
					static_assert( is_ostream_input< RetType >::value );
					return OutputStreamLazy< RetType >( std::forward< CallableType >( f ) );
				}

				/**
				 * Construct a new Output Stream object from a telemetry controller \p -tt
				 * and an output stream \p _out (usually \a std::cout)
				 */
				OutputStream(
					const TelControllerType & _tt,
					std::ostream & _out
				) :
					TelemetryBase< TelControllerType, enabled >( _tt ),
					out( _out )
				{}

				/**
				 * Copy constructor.
				 */
				OutputStream( const self_t & _outs ) = default;

				OutputStream & operator=( const self_t & _out ) = delete;

				/**
				 * Stream input operator, enabled for all types std::ostream supports.
				 */
				template< typename T > inline typename std::enable_if< is_ostream_input< T >::value,
					self_t & >::type operator<<( T&& v ) {
					if ( this->is_active() ) {
						out << std::forward< T >( v );
					}
					return *this;
				}

				/**
				 * Specialization of the \a << operator for stream manipulators, to support
				 * \a std::endl and similar manipulators.
				 */
				inline self_t & operator<<( std::ostream& (*func)( std::ostream& ) ) {
					if ( this->is_active() ) {
						out << func;
					}
					return *this;
				}

				/**
				 * Specialization of the \a << operator for lazy evaluation of callable objects.
				 *
				 * A callable object can be wrapped into an #OutputStreamLazy<F> object in order
				 * to be called only if necessary, i.e., only if the stream \a this is active.
				 * In this case, the internal callable object is called, its result is materialized
				 * and sent into the stream.
				 *
				 * To conveniently instantiate an #OutputStreamLazy<F> to pass to this operator,
				 * see #makeLazy(CallableType&&).
				 *
				 * @tparam F type of the callable object
				 * @param fun callable object
				 * @return self_t & the stream itself
				 */
				template< class F > inline typename std::enable_if<
					is_ostream_input< decltype( std::declval< OutputStreamLazy< F > >()() ) >::value,
				self_t & >::type operator<<( const OutputStreamLazy< F >& fun ) {
					if ( this->is_active() ) {
						out << fun();
					}
					return *this;
				}

			private:
				std::ostream & out;
			};

			/**
			 * Template specialization of OutputStream<TelControllerType,enabled>
			 * for deactivated telemetry: no information is stored, no output produced.
			 */
			template<
				typename TelControllerType
			> class OutputStream< TelControllerType, false > :
				public TelemetryBase< TelControllerType, false > {
			public:
				using self_t = OutputStream< TelControllerType, false >;


				template< typename RetType > struct OutputStreamLazy {

					static_assert( is_ostream_input< RetType >::value );

					template< class F > OutputStreamLazy( F&& ) {}

					constexpr char operator()() const { return '\0'; }
				};

				template<
					typename CallableType,
					typename RetType = decltype( std::declval< CallableType >()() )
				> static OutputStreamLazy< RetType > makeLazy( CallableType&& f ) {
					static_assert( is_ostream_input< RetType >::value );
					return OutputStreamLazy< RetType >( std::forward< CallableType >( f ) );
				}

				OutputStream() = default;

				OutputStream( const TelControllerType & _tt, std::ostream & ) :
					TelemetryBase< TelControllerType, false >( _tt ) {}

				OutputStream( const self_t & _out ) = default;

				OutputStream & operator=( const self_t & _out ) = delete;

				inline self_t & operator<<( std::ostream& (*)( std::ostream& ) ) {
					return *this;
				}

				/**
				 * All-capturing implementation for the input stream operator, printing nothing.
				 *
				 * This operator is convenient especially for debugging cases.
				 * In case of "normal" stream types used with custom data types, the user
				 * must extend them manually to print the custom data type. If the user uses a
				 * deactivated stream (for example as a default template parameter to disable
				 * logging by default), she needs not extend it for custom types in order
				 * to make it compile, which is especially nonsensical when the output is deactivated.
				*/
				template< typename T > self_t & operator<<( T&& ) {
					return *this;
				}
			};

			/// Always active output stream, mainly for debugging purposes.
			using OutputStreamOn = OutputStream< TelemetryControllerAlwaysOn, true >;

			/// Always inactive output stream
			using OutputStreamOff = OutputStream< TelemetryControllerAlwaysOff, false >;

		}
	}
}

#endif // _H_GRB_UTILS_TELEMETRY_OUTPUT_STREAM
