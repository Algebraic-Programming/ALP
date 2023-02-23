
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
 * @file CSVWriter.hpp
 * @author Alberto Scolari (alberto.scolar@huawei.com)
 *
 * Definition for the CSVWriter class.
 */

#ifndef _H_GRB_UTILS_TELEMETRY_CSV_WRITER
#define _H_GRB_UTILS_TELEMETRY_CSV_WRITER

#include <fstream>
#include <initializer_list>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "TelemetryBase.hpp"

namespace grb {
	namespace utils {
		namespace telemetry {

			/// standard CSV separator
			static constexpr char STD_CSV_SEP = ',';

			template< class U, class... Us >
			struct __is_csv_printable {
				static constexpr bool value = std::is_arithmetic< U >::value;
			};

			template< class U1, class U2, class... Us >
			struct __is_csv_printable< U1, U2, Us... > {
				static constexpr bool value = __is_csv_printable< U1 >::value
					&& __is_csv_printable< U2, Us... >::value;
			};

			/**
			 * Class to store numerical information in form of lines and emit it as a CSV, with
			 * heading, field separator and newlines.
			 *
			 * The user should add an line at once via #add_line( UTypes && ) and can
			 * then output it to an \a std::ostream or a file, together with the
			 * heading specified at construction. The output is a fully compliant CSV file
			 * that can be read by common tools like spreadsheets and parsers (e.g. Pandas,
			 * https://pandas.pydata.org/). This class allows easily emitting telemetry
			 * information and importing them into advanced tools for thourough analysis.
			 *
			 * This implementation assumes telemetry is enabled, since a specialization for
			 * disabled telemetry follows.
			 * It internally allocates memory dynamically to store the lines.
			 * Only numerical information can be stored.
			 *
			 * @tparam TelControllerType type for the telemetry controller
			 * @tparam enabled whether telemetry is enabled
			 * @tparam T1 numerical type of the first field to store (at least one is required)
			 * @tparam Ts numerical types of the following fields to store
			 */
			template<
				typename TelControllerType,
				bool enabled,
				class T1,
				class... Ts
			> class CSVWriter :
				public TelemetryBase< TelControllerType, enabled > {
			public:
				static_assert( __is_csv_printable< T1, Ts... >::value, "not all types are printable" );

				using self_t = CSVWriter< TelControllerType, enabled, T1, Ts... >;

				using base_t = TelemetryBase< TelControllerType, enabled >;

				class CSVLastTuple {
				public:
					CSVLastTuple( const self_t & _csv ) : csv( _csv ) {}

					CSVLastTuple( const CSVLastTuple & clt ) : csv( clt.csv ) {}

					inline friend std::ostream & operator<<( std::ostream & stream, const CSVLastTuple & t ) {
						return t.csv.write_last_line_to_stream( stream );
					}

				private:
					const self_t & csv;
				};

				CSVWriter() = delete;

				/**
				 * Full constructor for a CSVWriter.
				 *
				 * @param tt telemetry controller
				 * @param _headers CSV headers, whose number must match the number of T types to print
				 * @param _separator field separator for printing
				 * @param size hint size for initial memory allocation (dynamic allocation may occur anyway)
				 */
				CSVWriter(
					const TelControllerType & tt,
					std::initializer_list< const char * > _headers,
					char _separator,
					size_t size
				) :
					base_t( tt ),
					separator( _separator )
				{
					if( _headers.size() != NUM_FIELDS ) {
						throw std::runtime_error( "wrong number of headers, it must match the unmber of line elements" );
					}
					// emplace anyway, so that the object is always in a consistent state and can be
					// activated/deactivated at runtime
					for( const auto & h : _headers ) {
						headers.emplace_back( h );
					}
					if( ! tt.is_active() ) {
						return;
					}
					lines.reserve( size );
					// zero to force physical allocation
					// std::memset( reinterpret_cast< void * >( lines.data() ), 0, lines.size() * sizeof( tuple_t ) );
				}

				/**
				 * Construct a new CSVWriter object assuming a comma separator and an initial
				 * amount of lines to store.
				 */
				CSVWriter(
					const TelControllerType & tt,
					std::initializer_list< const char * > _headers
				) : CSVWriter( tt, _headers, STD_CSV_SEP, 10 ) {}

				CSVWriter( const self_t & ) = delete;

				CSVWriter( self_t && ) = delete;

				self_t & operator=( const self_t & ) = delete;

				self_t & operator=( self_t && ) = delete;

				/**
				 * Add a line to the CSV, i.e., store the numerical information internally.
				 *
				 * @tparam UTypes information types whose number must match the number of fields in the CSV;
				 * 	these types must also be implicitly convertible to the corresponding T1, Ts... types
				 */
				template< class... UTypes >
				void add_line( UTypes &&... vs ) {
					if( this->is_active() ) {
						lines.emplace_back( std::forward< UTypes >( vs )... );
					}
				}

				/**
				 * Remove all lines from the CSV.
				 */
				void clear() {
					lines.clear();
				}

				/**
				 * Emit the last line of the CSV into \p stream as actual text, i.e. with the fields separated.
				 * Does not print the newline.
				 *
				 * If there is no line stored, the behavior is undefined.
				 *
				 * @param stream stream to write into
				 * @return std::ostream& \p stream itself
				 */
				std::ostream & write_last_line_to_stream( std::ostream & stream ) const {
					if( lines.size() > 0 && this->is_active() ) {
						write_line( stream, lines.back() );
					}
					return stream;
				}

				/**
				 * Returns an object that can be streamed into an std::cout stream via the \a << operator
				 * in order to print the last line stored.
				 *
				 * If there is no line stored, the behavior is undefined.
				 */
				CSVLastTuple last_line() const {
					if( lines.size() == 0 ) {
						throw std::runtime_error( "no measures" );
					}
					return CSVLastTuple( *this );
				}

				/**
				 * Write the entire CSV into \p stream, with heading (heading, separated fields with newline).
				 */
				std::ostream & write_to_stream( std::ostream & stream ) const {
					if( ! this->is_active() ) {
						return stream;
					}
					write_header( stream );
					stream << NEW_LINE;
					for( const tuple_t & line : lines ) {
						write_line( stream, line );
						stream << NEW_LINE;
					}
					return stream;
				}

				/**
				 * Creates a new file named \p name (or overwrites an existing one) and stores the entire CSV
				 * into it.
				 */
				void write_to_file( const char * name ) const {
					if( ! this->is_active() ) {
						return;
					}
					std::ofstream file( name );
					if( ! file.is_open() ) {
						throw std::runtime_error( "cannot open file" );
					}
					write_to_stream( file );
					file.close();
				}

			private:
				static constexpr char NEW_LINE = '\n';

				static constexpr size_t NUM_FIELDS = sizeof...( Ts ) + 1;

				using tuple_t = std::tuple< T1, Ts... >;

				std::vector< std::string > headers;
				const char separator;
				std::vector< tuple_t > lines;

				std::ostream & write_header( std::ostream & stream ) const {
					stream << headers[ 0 ];
					for( size_t i = 1; i < headers.size(); i++ ) {
						stream << separator << headers[ i ];
					}
					return stream;
				}

				void write_line( std::ostream & stream, const tuple_t & line ) const {
					write_val< 0 >( stream, line );
				}

				// recursive case
				template< size_t OFFS >
				inline void write_val( std::ostream & stream, typename std::enable_if < OFFS< NUM_FIELDS - 1, const tuple_t & >::type _tup ) const {
					stream << std::get< OFFS >( _tup ) << separator;
					write_val< OFFS + 1 >( stream, _tup ); // tail recursion
				}

				// base case
				template< size_t OFFS >
				inline void write_val( std::ostream & stream, typename std::enable_if< OFFS == NUM_FIELDS - 1, const tuple_t & >::type _tup ) const {
					(void)separator;
					stream << std::get< OFFS >( _tup );
				}
			};

			/**
			 * Temaplate specialization that assumes disabled telemetry: no state is kept,
			 * operations produce no result when invoked (no output into streams, no file creation).
			 *
			 * @tparam TelControllerType
			 * @tparam T1
			 * @tparam Ts
			 */
			template<
				typename TelControllerType,
				class T1,
				class... Ts
			> class CSVWriter< TelControllerType, false, T1, Ts... > :
				public TelemetryBase< TelControllerType, false > {
			public:
				static_assert( __is_csv_printable< T1, Ts... >::value, "not all types are printable" );

				using self_t = CSVWriter< TelControllerType, false, T1, Ts... >;

				using base_t = TelemetryBase< TelControllerType, false >;

				CSVWriter() = delete;

				CSVWriter(
					const TelControllerType & tt,
					std::initializer_list< const char * >,
					char,
					size_t
				) : base_t( tt ) {}

				CSVWriter(
					const TelControllerType & tt,
					std::initializer_list< const char * > _headers
				) : CSVWriter( tt, _headers, STD_CSV_SEP, 10 ) {}

				CSVWriter( const self_t & ) = delete;

				CSVWriter( self_t && ) = delete;

				self_t & operator=( const self_t & ) = delete;

				self_t & operator=( self_t && ) = delete;

				template< class... UTypes > void add_line( UTypes &&... ) {
					static_assert( sizeof...( UTypes ) == sizeof...( Ts ) + 1 );
				}

				void clear() {}

				std::ostream & write_last_line_to_stream( std::ostream & stream ) const {
					return stream;
				}

				char last_line() const {
					return '\0';
				}

				std::ostream & write_to_stream( std::ostream & stream ) const {
					return stream;
				}

				void write_to_file( const char * name ) const {
					(void)name;
				}
			};

			/**
			 * Implementation of CSVWriter for enabled telemetry, with implemented operations.
			 */
			template< class T1, class... Ts >
			using StaticCSVWriter = CSVWriter< TelemetryControllerAlwaysOn, true, T1, Ts... >;

		} // namespace telemetry
	}     // namespace utils
} // namespace grb

#endif // _H_GRB_UTILS_TELEMETRY_CSV_WRITER
