#ifndef _H_ALP_ASCEND_UTILS
#define _H_ALP_ASCEND_UTILS

#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <atomic>

namespace alp {

	template< class T >
	constexpr T Zero = T( 0 );

	template< class T >
	constexpr T Infinity = std::numeric_limits< T >::infinity();

	// TODO: fix this so that mInfinity=-Infinity
	template< class T >
	constexpr T mInfinity = -Infinity< T >;

	enum class Datatype { FP16, FP32, VIEW_TYPE, NO_TYPE };

	namespace internal {

		enum class Rule {

				NONE,
				EWISE,
				BCAST,
				REDUCE
		};

		enum class Scope {

				GLOBAL,
				LOCAL,
				TEMP,
				VIEW
		};

		enum class Stagetype {

				GET_VIEW,
				STORE,
				IMPLICIT_FREE,
				SET_TENSOR,
				SET_SCALAR,
				APPLY_ADD,
				APPLY_MINUS,
				FOLDL_EXP,
				FOLDL_DIVIDE,
				FOLDL_MAX,
				FOLDL_TIMES,
				FOLDL_ADD
		};


		 std::string getDataType( const Datatype dtype );
		 std::string getScope( const Scope scope );
		 std::vector< int > vectorOfVectorsToVector( const std::vector< std::vector< int > > &vector_of_sets );
		 std::vector< int > vectorDifference( const std::vector< int > &vector1, const std::vector< int > &vector2 );
		 bool vectorSubset( const std::vector< int > &vector1, const std::vector< int > &vector2 );
		 std::vector< int > vectorUnion( const std::vector< int > &vector1, const std::vector< int > &vector2 );

	} // namespace internal


	static std::atomic_int axes_counter{0};


	static inline int getAxisId( const std::string &axis ) {
		static std::unordered_map<std::string, int> associations;

		if (associations.find(axis) == associations.end()) {
			associations[axis] = axes_counter++;
		}

		return associations[axis];
	}

	template< typename IntegralType = int >
	static inline int getAxisId( 
		const IntegralType axis,
		typename std::enable_if< std::is_integral< IntegralType >::value, int >::type* = 0
	) {
		return static_cast< int >( axis );
	}

	static inline int getAxisId( const char* axis ) {
		return getAxisId( std::string( axis ) );
	}

	template< typename = void >
	static inline std::vector< int > make_axes( ) {
		return std::vector< int >(0);
	}

	template< typename AxisType >
	static inline std::vector< int > make_axes( AxisType axis ) {
		return std::vector< int >{ getAxisId( axis ) };
	}

	template< typename AxisType, typename... AxisPackType >
	static std::vector< int > make_axes( const AxisType arg1, AxisPackType const... args ) {
		std::vector< int > axes{ getAxisId( arg1 ) };

		for( auto arg : { args... } ) {
			axes.push_back( getAxisId( arg ) );
		}

		return axes;
	}


} // namespace alp

#endif // _H_ALP_ASCEND_UTILS
