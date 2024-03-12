#ifndef _H_ALP_ASCEND_OPERATORS
#define _H_ALP_ASCEND_OPERATORS

#include <vector>

#include <graphblas.hpp>

//#include "graphblas/ascend/grid.hpp"
#include "graphblas/ascend/utils.hpp"


namespace alp
{
	class Tensor;


	Tensor getView( const Tensor &parent );

	// TODO extend to multiple containers
	void store( const Tensor &output );

	void set(
		Tensor &tout,
		Tensor &tin,
		const std::vector< int > &activeAxes = std::vector< int >()
	);

	void set(
		Tensor &tout,
		double alpha		// TODO: this is hardcoded datatype
	);

	void apply(
		Tensor &tout,
		Tensor &tin,
		const std::string &opName,
		const std::vector< int > &activeAxes = std::vector< int >()
	);

	void apply(
		Tensor &tout,
		Tensor &tin1,
		Tensor &tin2,
		const std::string &opName,
		const std::vector< int > &activeAxes = std::vector< int >()
	);

	void foldl(
		Tensor &tinout,
		Tensor &tin,
		const std::string &opName,
		const std::vector< int > &activeAxes = std::vector< int >()
	);

//	template< size_t sm, size_t pm >
	void foldl(
//		const Grid< sm, pm > &grid,
		Tensor &tinout,
		const std::string &opName,
		const std::vector< int > &activeAxes = std::vector< int >()
	);


	struct ReductionOperation {
		Tensor &input;
		const std::vector< int > axes;
		const internal::Stagetype opType;
		const std::string opName;

		ReductionOperation(
			Tensor &input,
			const std::vector< int > &axes,
			const internal::Stagetype &op,
			const std::string &opName
		) :
			input( input ),
			axes( axes ),
			opType( op ),
			opName( opName ) {}

	};

	/**
	 * Max-reduce operator
	 */
	template< typename AxisType >
	ReductionOperation max( Tensor &z, const AxisType axis ) {
		static_assert( 
			std::is_convertible< AxisType, int >::value || std::is_convertible< AxisType, std::string >::value,
			"AxisType must be convertible to int or std::string"
		);
		const int axisId = getAxisId( axis );
		return { z, { axisId }, internal::Stagetype::FOLDL_MAX, "max" };
	}

	/**
	 * Add-reduce operator
	 */
	template< typename AxisType >
	ReductionOperation add( Tensor &z, const AxisType axis ) {
		static_assert( 
			std::is_convertible< AxisType, int >::value || std::is_convertible< AxisType, std::string >::value,
			"AxisType must be convertible to int or std::string"
		);
		const int axisId = getAxisId( axis );
		return { z, { axisId }, internal::Stagetype::FOLDL_ADD, "add" };
	}

	struct ApplyOperation {
		Tensor& input1;
		Tensor& input2;
		const std::vector< int > axes;
		const std::string opName;

		ApplyOperation(
			Tensor &input1, Tensor &input2,
			const std::vector< int > &axes,
			const std::string &opName
		) :
			input1( input1 ),
			input2( input2 ),
			axes( axes ),
			opName( opName ) { }
	};

	/**
	 * Add-reduce operator
	 */
	template< typename AxisType >
	ApplyOperation add( Tensor &y, Tensor &z, const AxisType axis ) {
		static_assert( 
			std::is_convertible< AxisType, int >::value || std::is_convertible< AxisType, std::string >::value,
			"AxisType must be convertible to int or std::string"
		);
		const int axisId = getAxisId( axis );
		// std::vector<Tensor&> inputs = { y, z };
		return { y, z, { axisId }, "add" };
	}



	/**
	 * Minus operator
	 */
	template< typename AxisType >
	ApplyOperation minus( Tensor &y, Tensor &z, const AxisType axis ) {
		static_assert( 
			std::is_convertible< AxisType, int >::value || std::is_convertible< AxisType, std::string >::value,
			"AxisType must be convertible to int or std::string"
		);
		const int axisId = getAxisId( axis );
		return { y, z, { axisId }, "minus" };
	}

} // namespace alp

#endif
