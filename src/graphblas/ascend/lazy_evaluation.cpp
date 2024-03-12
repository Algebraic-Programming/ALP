#include <graphblas/ascend/lazy_evaluation.hpp>
#include <graphblas/ascend/stage.hpp>


namespace alp
{
	namespace internal
	{
		AscendLazyEvaluation ale;
	}
}

alp::internal::AscendLazyEvaluation::AscendLazyEvaluation() {

	num_pipelines = 0;
	addPipeline(); //TODO add the first pipeline
}

void alp::internal::AscendLazyEvaluation::addPipeline() {

	pipelines.emplace_back( AscendPipeline( num_pipelines ) );
	num_pipelines++;
}

void alp::internal::AscendLazyEvaluation::insertFreeInputTensorStages( const std::vector< int > &forEachAxes ) {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->insertFreeInputTensorStages( forEachAxes );
	}
}

const alp::Tensor &alp::internal::AscendLazyEvaluation::store( const alp::Tensor &output_tensor ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	return pipeline->store( output_tensor );
}

void alp::internal::AscendLazyEvaluation::clear() {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->clear();
	}
}

void alp::internal::AscendLazyEvaluation::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const alp::Tensor &tensor1, const double alpha, const std::vector< int > &forEachAxes ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	pipeline->addStage( op_type, rule, tensor1, alpha, forEachAxes );
}

void alp::internal::AscendLazyEvaluation::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const alp::Tensor &tensor1, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	pipeline->addStage( op_type, rule, tensor1, activeAxes, forEachAxes );
}

void alp::internal::AscendLazyEvaluation::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const alp::Tensor &tensor1, const alp::Tensor &tensor2, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	pipeline->addStage( op_type, rule, tensor1, tensor2, activeAxes, forEachAxes );
}

void alp::internal::AscendLazyEvaluation::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const alp::Tensor &tensor1, const alp::Tensor &tensor2, const alp::Tensor &tensor3, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	pipeline->addStage( op_type, rule, tensor1, tensor2, tensor3, activeAxes, forEachAxes );
}
/*
void alp::internal::AscendLazyEvaluation::addStage( alp::internal::Stagetype op_type, alp::internal::Rule rule, const alp::Tensor &tensor1, const alp::Tensor &tensor2, const alp::Tensor &tensor3, const alp::Tensor &tensor4, const std::vector< int > &activeAxes, const std::vector< int > &forEachAxes ) {

	//TODO: perhaps data dependence analysis will determine the right pipeline
	auto pipeline = pipelines.rbegin();
	pipeline->addStage( op_type, rule, tensor1, tensor2, tensor3, tensor4, activeAxes, forEachAxes );
}
*/
void alp::internal::AscendLazyEvaluation::generateDeclarations( std::stringstream &declarations ) {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->generateDeclarations( declarations );
	}
}
/*
void alp::internal::AscendLazyEvaluation::generateConstructor( std::stringstream &constructor ) {
	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->generateConstructor( constructor );
	}
}
*/
void alp::internal::AscendLazyEvaluation::generateHostBody( std::stringstream &os, std::stringstream &analyticModelArgs,
									std::stringstream &analyticModelFormalParams, std::stringstream &analyticModelDecls,
									std::stringstream &analyticModelConstrBody ) {
	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->generateHostBody( os, analyticModelArgs, analyticModelFormalParams, analyticModelDecls, analyticModelConstrBody );
	}
}

void alp::internal::AscendLazyEvaluation::generateInit( std::stringstream &init ) {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->generateInit( init );
	}
}

void alp::internal::AscendLazyEvaluation::generateProcess( std::stringstream &process, std::stringstream &processCall ) {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->generateProcess( process, processCall );
	}
}

void alp::internal::AscendLazyEvaluation::debug_print() const {

	for( auto it = pipelines.begin(); it != pipelines.end(); ++it ) {
		it->debug_print();
	}
}
