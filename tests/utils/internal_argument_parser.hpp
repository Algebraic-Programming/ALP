
/*
 *   Copyright 2021 Huawei Technologies Co., Ltd.
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
 * @file internal_argument_parser.hpp
 * @author Alberto Scolari (alberto.scolari@huawei.com)
 * @brief Definitions of the data structures for argument parsing from command line.
 *
 * The data types being accepted and parsed as command-line arguments are stored in the
 * parsed_types.hpp file, and the corresponding methods and members are defined in the
 * classes defined in this file.
 * To generate such members conveniently, the file parsed_types.hpp is repeatedly included
 * and the internal macro (\c PARSED_TYPE()) is repeatedly re-defined according to the member
 * to be generated. This avoids a lot of code repetitions, as most of these members differ only
 * for a few data type information.
 *
 * The definitions here depend on the macros defined in internal_argument_parser_defs.hpp.
 * To add one more parsed type:
 * -# add the corresponding \c typedef below to ensure the type name has no whitespaces
 * -# add the parsed type in file parsed_types.hpp, using the \c PARSED_TYPE(<newtype_parse_t>)
 *      macro
 * -# add the corresponding parsing lambda inside argument_parser.cpp, as from the examples within
 *
 * Parsing lambdas are assumed to:
 * - parse the value string
 * - throw exceptions in case it is not valid
 * - store the argument inside the target variable, possibly casting it to the correct pointer type
 *
 * Lambdas to print the default value are also generated automatically via header inclusion and pre-processing
 * directives, under the assumption that the corresponding
 * \code operator<<(std::ostream& os, const newtype& var) \endcode
 * exists. If this assumption is not valid, the user has to specify a custom default printer lambda,
 * as done for the \c bool type.
 *
 * @date 2021-04-30
 */

#ifndef _H_INTERNAL_UTILS_ARG_PARSER
#define _H_INTERNAL_UTILS_ARG_PARSER

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <functional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "internal_argument_parser_defs.hpp"


// ================ parsed types ===================
typedef bool bool_parse_t;
typedef size_t size_parse_t;
typedef const char * str_parse_t;
typedef double double_parse_t;

// =================================================

/**
 * @brief Container of default values for each parsed type, implemented as a union.
 */
// replace in future with std::variant from C++17
union default_value_container {

// declare members
#define PARSED_TYPE( _t, _vdef ) _t DEF_CONT_MEMBER_NAME( _t );
#include "parsed_types.hpp"
#undef PARSED_TYPE

// declare constructors
#define PARSED_TYPE( _t, _vdef )                                              \
	default_value_container( _t CONCAT( _, DEF_CONT_MEMBER_NAME( _t ) ) ) {   \
		DEF_CONT_MEMBER_NAME( _t ) = CONCAT( _, DEF_CONT_MEMBER_NAME( _t ) ); \
	}
#include "parsed_types.hpp"
#undef PARSED_TYPE
};

// =================================================

/**
 * @brief Container of information for argument parsing, default and printing.
 */
struct argument_parse_info {

	using def_container_t = default_value_container;
	using parser_t = const std::function< void( const char *, void * ) >;
	using def_setter_t = const std::function< void( const def_container_t &, void * ) >;
	using def_printer_t = const std::function< void( const def_container_t &, std::ostream & ) >;

	void * const target;                         ///< where to store the parsed value
	parser_t & parser;                           ///< routine parsing the string from the command line and storing it into ::target
	bool const is_option;                        ///< whether it is an option, i.e. a command line switch
	default_value_container const default_value; ///< the default value for this argument (if optional)
	def_setter_t & default_setter;               ///< function to set the default value (which depends on the data type)
	def_printer_t & default_printer;             ///< function to print the default (which depends on the data type)
	const char * const description;              ///< argument description to show when choosing the \a -h switch

	argument_parse_info() = delete;

	/**
	 * @brief Standard copy contructor.
	 *
	 * @param o original object
	 */
	argument_parse_info( const argument_parse_info & o ):
		target(o.target),
		parser(o.parser),
		is_option(o.is_option),
		default_value(o.default_value),
		default_setter(o.default_setter),
		default_printer(o.default_printer),
		description(o.description) {}

	/**
	 * @brief Constructor from all the needed values.
	 */
	argument_parse_info( void * _target, parser_t & _parser, bool option, default_value_container def, def_setter_t & _default_setter, def_printer_t & default_printer, const char * desc );

	/**
	 * @brief Templated constructor wrapping a default value of type \p T into the corresponding default container.
	 *
	 * @tparam T the argument type and default value
	 */
	template< typename T >
	argument_parse_info( T & _target, parser_t & _parser, bool & option, T & def, def_setter_t & _default_setter, def_printer_t & _default_printer, const char * desc ) :
		argument_parse_info( static_cast< void * >( &_target ), _parser, option, default_value_container( def ), _default_setter, _default_printer, desc ) {}
};

// =================================================

/**
 * @brief The container of parsing information and logic, which the user can instantiate, populate with
 *          arguments/options to be parsed (with relative info) and call.
 *
 * Note on naming:
 * - argument: command line switch with \b necessary value, to be parsed; e.g. <em>-f input_file.in</em>
 * - \b mandatory argument: an argument that \b must be specified in the command line, with related
 *      value; if not specified, the parsing fails and the process terminates
 * - \b optional argument: an argument that \b can be specified in the command line, but is not mandatory;
 *      if not specified, the default value is set (which must be given when populating the parser object)
 * - option: command line switch \b without any value, which can optionally be specified; it corresponds to a
 *      \c bool target variable and needs a default value to be set in case the option is not encountered;
 *      if the option is encountered while parsing, the negation of the default value is stored; example:
 *      <em>--do-dry-run</em>
 *
 * The user installs command line arguments by calling \c add_{mandatory,optional}_argument(...) or options by
 * calling \c add_option(...) . Rules
 * - the argument string cannot be \c nullptr
 * - it \b must start with \a - (and have any number of \a - afterward or any other character); hence,
 *      no specific format is enforce and \a -a , \a --a , \a ---a ,  \a -a-long-text-arg are all valid
 * - \a -h is \b not a valid argument, as it is reserved to print the list of arguments; however, variants
 *      like \a --help are available
 * - the argument cannot contain whitespaces
 *
 * No copy is made of the passed pointers or references (for example argument, description, ...), so it is the
 * user's responsibility to ensure that the memory references remain valid throughout the parser's lifetime.
 *
 * Example usage (excerpt from HPCG test):
 *
 * \code{.cpp}
 * parser.add_optional_argument("--nx", in.nx, PHYS_SYSTEM_SIZE_DEF,
 *     "physical system size along x")
 *   .add_optional_argument("--ny", in.ny, PHYS_SYSTEM_SIZE_DEF,
 *     "physical system size along y")
 *   .add_optional_argument("--nz", in.nz, PHYS_SYSTEM_SIZE_DEF,
 *     "physical system size along z")
 *   .add_option("--evaluation-run", in.evaluation_run, false,
 *     "launch single run directly, without benchmrker (ignore repetitions)")
 *   .add_option("--no-conditioning", in.no_conditioning, false, "do not apply pre/post-conditioning");
 *
 * parser.parse(argc, argv);
 * \endcode
 */
class argument_parser {
public:
	argument_parser();

	// for safety, disable any copy semantics
	argument_parser( const argument_parser & o ) = delete;

	argument_parser & operator=( const argument_parser & ) = delete;

// declare methods for mandatory arguments
#define MANDATORY_SIGNATURE( _t, arg, target, descr ) add_mandatory_argument( const char * arg, _t & target, const char * descr )
// declare methods for optional arguments
#define OPTIONAL_SIGNATURE( _t, arg, target, def, descr ) add_optional_argument( const char * arg, _t & target, _t def, const char * descr )

#define NO_BOOL
#define PARSED_TYPE( _t, _vdef )                                               \
	argument_parser & MANDATORY_SIGNATURE( _t, arg, target, descr = nullptr ); \
	argument_parser & OPTIONAL_SIGNATURE( _t, arg, target, def, descr = nullptr );
#include "parsed_types.hpp"
#undef PARSED_TYPE
#undef NO_BOOL

	/**
	 * @brief Adds an option to the parser (with optional description).
	 */
	argument_parser & add_option( const char * arg, bool & target, bool def, const char * desc = nullptr );

	/**
	 * @brief Runs the parser with the inputs from the command line; in case of error, the process is terminated.
	 */
	void parse( int argc, const char * const * argv );

	/**
	 * @brief Prints a list of all arguments (including \a -h first) in the given order,
	 *          with description (if present), whether it is an option and whether it is optional or mandatory.
	 */
	void print_all_arguments();

private:
	using parser_t = argument_parse_info::parser_t;
	using def_setter_t = argument_parse_info::def_setter_t;
	using def_printer_t = argument_parse_info::def_printer_t;
	using def_container_t = argument_parse_info::def_container_t;

	//================ PREPROCESSOR-GENERATED MEMBERS ==================

#define NO_BOOL
#define PARSED_TYPE( _t, _vdef ) static parser_t const PARSER_NAME( _t );
#include "parsed_types.hpp"
#undef PARSED_TYPE
#undef NO_BOOL

#define PARSED_TYPE( _t, _vdef )                  \
	static def_setter_t const DEF_SET_NAME( _t ); \
	static def_printer_t const DEF_PRINT_NAME( _t );
#include "parsed_types.hpp"
#undef PARSED_TYPE

	//============================================================

	static def_setter_t const nothing_setter; ///< does not set any value

	static def_printer_t const nothing_printer; ///< does not print any value

	/**
	 * @brief Prints a single argument \p arg, with parsing information \p p and also print
	 *          whether it is mandatory according to \p mandatory.
	 */
	static void print_argument( const char * arg, const argument_parse_info & p, bool mandatory );

	static parser_t const option_parser; ///< parser for option, storing into the target the negated default

	// with C++17, we may use std::basic_string_view not no create a full string (and avoid data copies)
	using arg_map = std::unordered_map< std::string, size_t >;
	using mandatory_set = std::set< size_t >;
	std::vector< argument_parse_info > parsers; ///< vector of parsing information, in insertion order;
	                                            ///< the order \b cannot be changed

	//====== state objects with parsing info, to be filled only when adding arguments ======

	arg_map args_info;                ///< maps for <argument string, offset of parsing info inside ::parsers>
	std::vector< const char * > args; ///< vector of all arguments passed, to print them for the help
	mandatory_set mandatory_args;     ///< set of mandatory arguments, to check which one have not been encountered
	///< during parsing (and fail)

	std::vector< bool > found; ///< vector to track the arguments that have been found during parsing,
	///< to set the default value in case an optional one has not been encountered
	std::remove_const< parser_t >::type help_parser; ///< parser for \a -h option, which prints all available
	                                                 ///< asrguments and terminates the process
	bool help_target;                                ///< fictitious member to store the result of parsing -h

	/**
	 * @brief Private method to store an argument with its information into the internal data structures.
	 *
	 * This method checks whether \p arg is valid.
	 *
	 * @tparam T type of the target, hence type of the default value
	 * @param arg argument string
	 * @param target reference to store the parsed value to
	 * @param parser parser lambda
	 * @param option whether it is an option
	 * @param def default value for the argument
	 * @param default_setter lambda setting the default value
	 * @param default_printer lambda printing the default value
	 * @param desc description of the argument, to be printed with \a -h (optional)
	 * @param mandatory whether it is a mandatory argument
	 * @return argument_parser& returns \c *this
	 */
	template< typename T >
	argument_parser &
	__add_argument( const char * arg, T & target, parser_t & parser, bool option, T def, def_setter_t & default_setter, def_printer_t & default_printer, const char * desc, bool mandatory ) {
		if( arg == nullptr ) {
			throw std::invalid_argument( "the argument cannot be null" );
		}
		if( strnlen( arg, 1000 ) == 0 ) {
			throw std::invalid_argument( "the argument cannot be empty" );
		}
		if( arg[ 0 ] != '-' ) {
			throw std::invalid_argument( "the argument must start with '-'" );
		}
		bool has_whitespace { std::any_of( arg, arg + strlen( arg ), isspace ) };
		if( has_whitespace ) {
			throw std::invalid_argument( "passed argument contains a "
										 "whitespace" );
		}
		if( strcmp( "-h", arg ) == 0 ) {
			throw std::invalid_argument( "\"-h\" is not a valid argument" );
		}
		return __add_argument_unsafe( arg, target, parser, option, def, default_setter, default_printer, desc, mandatory );
	}

	/**
	 * @brief Unsafe version of ::__add_argument(), checking only that \p arg has not been added yet.
	 *          Use with care!
	 */
	template< typename T >
	argument_parser &
	__add_argument_unsafe( const char * arg, T & target, parser_t & parser, bool option, T def, def_setter_t & default_setter, def_printer_t & default_printer, const char * desc, bool mandatory ) {
		std::string arg_string( arg );
		if( args_info.find( arg_string ) != args_info.end() ) {
			throw std::invalid_argument( arg_string + " is already present" );
		}
		parsers.emplace_back( target, parser, option, def, default_setter, default_printer, desc );
		size_t position { parsers.size() - 1 };
		args_info.emplace( std::move( arg_string ), position );
		args.push_back( arg );
		if( mandatory ) {
			mandatory_args.insert( position );
		}
		return *this;
	}
};

#endif // _H_INTERNAL_UTILS_ARG_PARSER

