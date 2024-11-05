
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
 * @file
 *
 * The main header to include in order to use ALP/Ascend codegen.
 *
 * @author A. N. Yzelman.
 * @date 12th of September, 2023.
 */

#ifndef _H_ALPASCEND
#define _H_ALPASCEND

#include <functional>
#include <limits>
#include <cstddef>

#include <graphblas.hpp>
#include <graphblas/ascend/tensor.hpp>
#include <graphblas/ascend/operators.hpp>
#include <graphblas/ascend/grid.hpp>
#include <graphblas/ascend/opgen.hpp>
#include <graphblas/ascend/lazy_evaluation.hpp>
#include <graphblas/ascend/symbolTable.hpp>

/**
 * \defgroup ALPAscend ALP/Ascend
 *
 * This the ALP/Ascend module.
 *
 * @{
 */

namespace alp
{
	namespace internal
	{
		extern iGrid *igrid;
		extern AscendLazyEvaluation ale;
		extern SymbolTable symbols;
	}
}

/** The ALP/Ascend namespace */
namespace alp {

	using grb::RC;

	using grb::toString;

	namespace internal {

		template< size_t process_mesh_order, size_t problem_mesh_order >
		using AscendCodeFunction = void (*) (
				const alp::Grid< process_mesh_order, problem_mesh_order > &,
				alp::RC &
			);
	
	}

	template< size_t process_mesh_order, size_t problem_mesh_order >
	static grb::RC compile(
		const internal::AscendCodeFunction<
			process_mesh_order,
			problem_mesh_order
		> ascend_code,
		const std::string &kernel_name
	) {
		grb::RC ret = grb::PANIC;
		grb::Launcher< grb::EXEC_MODE::AUTOMATIC > launcher;
		alp::Grid< process_mesh_order, problem_mesh_order > grid;

		alp::internal::igrid =
			new alp::internal::iGrid( process_mesh_order, problem_mesh_order );

		internal::OpGen::kernel_id = kernel_name;

		std::ofstream output_device_code;
		output_device_code.open ( internal::OpGen::kernel_id + "_npu_op.cpp", std::ofstream::out | std::ofstream::trunc);

		std::ofstream output_host_log;
		output_host_log.open ( "generate_host_code_" + internal::OpGen::kernel_id + ".inp", std::ofstream::out | std::ofstream::trunc);

		output_host_log << "0";
		for( size_t i = 1; i < process_mesh_order; ++i ) {
			output_host_log << "," << i;
		}
		output_host_log << std::endl;

		output_host_log << "0";
		for( size_t i = 1; i < problem_mesh_order; ++i ) {
			output_host_log << "," << i;
		}
		output_host_log << std::endl;

		// TODO perhaps the processSize and problemSize members should be generated
		//		more than once, for every forEach
		//		only the tile_num is the same?

		// const uint32_t _p0
		internal::OpGen::hostFormalParam << "const uint32_t _" << alp::internal::igrid->processSize( 0 );

		// , const uint32_t _p1, const uint32_t _p2, const uint32_t _p3 ...
		for( size_t i = 1; i < process_mesh_order; ++i ) {
			internal::OpGen::hostFormalParam << ", const uint32_t _" << alp::internal::igrid->processSize( i );
		}

		// , const uint32_t _n0, const uint32_t _n1, const uint32_t _n2
		for( size_t i = 0; i < problem_mesh_order; ++i ) {
			internal::OpGen::hostFormalParam << ", const uint32_t _" << alp::internal::igrid->problemSize( i );
		}

		// _p0
		internal::OpGen::hostArg << "_" << alp::internal::igrid->processSize( 0 );

		// , _p1, _p2, _p3 ...
		for( size_t i = 1; i < process_mesh_order; ++i ) {
			internal::OpGen::hostArg << ", _" << alp::internal::igrid->processSize( i );
		}

		// , _n0, _n1, _n2 ...
		for( size_t i = 0; i < problem_mesh_order; ++i ) {
			internal::OpGen::hostArg << ", _" << alp::internal::igrid->problemSize( i );
		}

		// p0 = _p0;
		// p1 = _p1;
		// p2 = _p2;
		// ...
		// when i < process_mesh_order
		for( size_t i = 0; i < process_mesh_order; ++i ) {
			internal::OpGen::constrBody << "\n";
			internal::OpGen::constrBody << "\t\t\t"
				<< alp::internal::igrid->processSize( i )
				<< " = _" << alp::internal::igrid->processSize( i )
				<< ";";
		}

		// p1 = 1;
		// p2 = 1;
		// ...
		// when process_mesh_order <= i < problem_mesh_order
		for( size_t i = process_mesh_order; i < problem_mesh_order; ++i ) {
			internal::OpGen::constrBody << "\n";
			internal::OpGen::constrBody << "\t\t\t"
				<< alp::internal::igrid->processSize( i )
				<< " = 1;";
		}

		internal::OpGen::constrBody << "\n";

		// n0 = _n0;
		// n1 = _n1;
		// n2 = _n2;
		// ...
		for( size_t i = 0; i < problem_mesh_order; ++i ) {
			internal::OpGen::constrBody << "\n";
			internal::OpGen::constrBody << "\t\t\t"
				<< alp::internal::igrid->problemSize( i ) << " = _"
				<< alp::internal::igrid->problemSize( i ) << ";";
		}

		internal::OpGen::constrBody << "\n";

		// uint32_t p0;
		// uint32_t p1;
		// uint32_t p2;
		for( size_t i = 0; i < problem_mesh_order; ++i ) {
			internal::OpGen::classMembers << "\t\tuint32_t "
				<< alp::internal::igrid->processSize( i ) << ";\n";
		}

		internal::OpGen::classMembers << "\n";

		// uint32_t n0;
		// uint32_t n1;
		// uint32_t n2;
		for( size_t i = 0; i < problem_mesh_order; ++i ) {
			internal::OpGen::classMembers << "\t\tuint32_t "
				<< alp::internal::igrid->problemSize( i ) << ";\n";
		}

		internal::OpGen::classMembers << "\n";

		const RC launch_rc = launcher.exec<
			alp::Grid< process_mesh_order, problem_mesh_order >,
			alp::RC
		> (
			ascend_code, grid, ret, true
		);
		if( launch_rc != grb::SUCCESS ) {
			throw std::runtime_error( "Launching codegen FAILED" );
		}

		// ANALYTIC MODEL
		{
			std::stringstream analyticModelArgs;
			std::stringstream analyticModelFormalParams;
			std::stringstream analyticModelDecls;
			std::stringstream analyticModelConstrBody;

			// host body generation appends to hostArgs, so the below line must follow the previous one(!)
			 alp::internal::ale.generateHostBody( internal::OpGen::hostBody,
									analyticModelArgs, analyticModelFormalParams,
									analyticModelDecls, analyticModelConstrBody );

			internal::OpGen::hostArg << analyticModelArgs.str();
			internal::OpGen::analyticModelFormalParams << analyticModelFormalParams.str();
			internal::OpGen::classMembers << analyticModelDecls.str();
			internal::OpGen::constrBody << analyticModelConstrBody.str();
		}

		/*
		 * Only once we are here we have execute all the forEach,
		 * and thus we have all the information we need to generate
		 * code and performs optimizations, especially across
		 * different forEach, and including handling multiple
		 * pipelines that may be built by the same forEach
		 *
		 */

//		alp::internal::symbols.debug_print();
//		alp::internal::ale.debug_print();

		// CLASS MEMBER DECLARATIONS
		{
			std::stringstream decl;
			alp::internal::ale.generateDeclarations( decl );
			internal::OpGen::declarations << decl.str();
		}

		// CONSTRUCTOR BODY
//		{
//			std::stringstream constructor;
//			alp::internal::ale.generateConstructor( constructor );
//			internal::OpGen::constrBody << constructor.str();
//		}

		// INIT BODY
		{
			if( alp::internal::symbols.existsTBufTensorDecl() == true ) {

				//TODO I should make the datatype a parameter
				std::string temp_data_type = "half";
				std::stringstream max_n;
/*
				max_n << "std::max( { " << alp::internal::igrid->problemSize( 0 );

				for( size_t i = 1; i < problem_mesh_order; ++i ) {
					max_n << ", " << alp::internal::igrid->problemSize( i );
				}

				// close all open parentheses
				max_n << " } )";
*/
				if( problem_mesh_order == 1 ) {
					max_n << "" << alp::internal::igrid->problemSize( 0 ) << "";
				} else {
					max_n << "alp::max( " << alp::internal::igrid->problemSize( 0 ) << ", ";

					for( size_t i = 1; i < problem_mesh_order - 1; ++i ) {
						max_n << "alp::max( " << alp::internal::igrid->problemSize( i ) << ", ";
					}

					// this corresponds to the last one, which is a special case
					// since it doesn't open a new recursive std::max
					max_n << alp::internal::igrid->problemSize( problem_mesh_order - 1 );

					// close all open parentheses
					for( size_t i = 1; i < problem_mesh_order; ++i ) {
						max_n << " )";
					}
				}

				internal::OpGen::initBody << "\n";
				internal::OpGen::initBody << "\t\t\tint32_t totWorkSpaceSize = alp::computeBufferSize( " << max_n.str() << ", sizeof( " << temp_data_type << " ) );\n";

			}

			std::stringstream init;
			alp::internal::ale.generateInit( init );
			internal::OpGen::initBody << init.str();

			if( alp::internal::symbols.existsTBufTensorDecl() == true ) {
				std::stringstream temp_local_init;
				alp::internal::symbols.generateTempLocalInit( temp_local_init );
				internal::OpGen::initBody << temp_local_init.str();
			}
		}

		// PROCESS
		{
			std::stringstream process, processCall;
			alp::internal::ale.generateProcess( process, processCall );
			internal::OpGen::processFunc.push_back( std::move( process ) );
			internal::OpGen::genericProcessBody << processCall.str();
		}

		alp::internal::OpGen::generate( output_device_code );

		std::stringstream listOfGlobalTensors;
		alp::internal::symbols.printHostLogFile( listOfGlobalTensors );
		output_host_log << listOfGlobalTensors.str() << std::endl;

		output_host_log << internal::OpGen::kernel_id << std::endl;

		output_host_log << internal::OpGen::analyticModelFormalParams.str() << std::endl;

		output_host_log << "$BEGIN_ANALYTIC_MODEL" << std::endl;
		output_host_log << internal::OpGen::hostBody.str();
		output_host_log << "$END_ANALYTIC_MODEL" << std::endl;

		output_device_code.close();
		output_host_log.close();

		internal::OpGen::compileClear();

		delete alp::internal::igrid;

		return ret;
	}

}

/** @} */

#endif // end _H_ALPASCEND

