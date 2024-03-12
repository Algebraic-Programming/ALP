#ifndef _H_ALP_ASCEND_STAGE
#define _H_ALP_ASCEND_STAGE

#include <vector>
#include <string>

#include <graphblas/ascend/utils.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/pipeline.hpp>

namespace alp {

	namespace internal {

		class Stage {

			private:

				const AscendPipeline &pipeline;
				const Stagetype enum_op_type;
				const Rule rule;
				const Tensor tensor0;
				Tensor tensor1;
				Tensor tensor2;
				Tensor tensor3;
				std::string tensor0_offset;
				std::string tensor1_offset;
				std::string tensor2_offset;
//				std::string tensor3_offset;
				std::string stride;
				double alpha; 	//TODO double should be replaced by alp::Scalar
				const std::vector< int > activeAxes;
				const std::vector< int > forEachAxes;


			public:

				Stage( const AscendPipeline &parent, Stagetype _enum_op_type, Rule _rule,
						const Tensor &_tensor0, const double _alpha, const std::vector< int > &_forEachAxes );
				Stage( const AscendPipeline &parent, Stagetype _enum_op_type, Rule _rule,
						const Tensor &_tensor0,
						const std::vector< int > &_activeAxes, const std::vector< int > &_forEachAxes );
				Stage( const AscendPipeline &parent, Stagetype _enum_op_type, Rule _rule,
						const Tensor &_tensor0, const Tensor &_in_tensor,
						const std::vector< int > &_activeAxes, const std::vector< int > &_forEachAxes );
				Stage( const AscendPipeline &parent, Stagetype _enum_op_type, Rule _rule,
						const Tensor &_tensor0, const Tensor &_tensor1, const Tensor &_tensor2,
						const std::vector< int > &_activeAxes, const std::vector< int > &_forEachAxes );
//				Stage( const AscendPipeline &parent, Stagetype _enum_op_type, Rule _rule,
//						const Tensor &_tensor0, const Tensor &_tensor1, const Tensor &_tensor2, const Tensor &_tensor3,
//						const std::vector< int > &_activeAxes, const std::vector< int > &_forEachAxes );
				Stagetype getOpType() const;
				Rule getRule() const;
				const Tensor &getTensor0() const;
				const std::vector< int > &getAxes() const;
				const std::vector< int > &getForEachAxes() const;
				std::string getOp( const std::string &tabs ) const;
				std::string generateApplyMinusOp( const std::string &tabs ) const;
				std::string generateApplyAddOp( const std::string &tabs ) const;
				std::string generateFoldlDivideOp( const std::string &tabs ) const;
				std::string generateFoldlMaxOp( const std::string &tabs ) const;
				std::string generateFoldlTimesOp( const std::string &tabs ) const;
				std::string generateFoldlAddOp( const std::string &tabs ) const;
				std::string generateFoldlExpOp( const std::string &tabs ) const;
				std::string generateSetTensorOp( const std::string &tabs ) const;
				std::string generateSetScalarOp( const std::string &tabs ) const;
				std::string generateGetViewOp( const std::string &tabs ) const;
				std::string generateStoreOp( const std::string &tabs ) const;
				std::string generateImplicitFreeOp( const std::string &tabs ) const;
				std::string generateToDoOp( const std::string &tabs ) const;


			private:

				std::vector< int > computeOperatorAxes() const;
				void computeMemoryOffsets();
				void semanticsCheks();
		};
	}

}

#endif // end _H_ALP_ASCEND_STAGE

