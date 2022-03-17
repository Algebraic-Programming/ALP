#include <iostream>
#include <fstream>

#include <Dialects/LinalgTransform/Passes.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Dialect/PDL/IR/PDLOps.h>

namespace grb {
	namespace jit {

		template< typename T >
		RC JitContext::executeFn( llvm::StringRef funcName, llvm::SmallVector< T > args ) {
			// read the execution tactic.
			const std::ifstream input_stream( "pdl.txt", std::ios_base::binary );

			if( input_stream.fail() ) {
				throw std::runtime_error( "Failed to open file" );
			}

			std::stringstream buffer;
			buffer << input_stream.rdbuf();

			auto tactic = buffer.str();
			mlir::OwningOpRef< mlir::ModuleOp > moduleTactic( mlir::parseSourceString< mlir::ModuleOp >( tactic, &ctx ) );
			mlir::OpBuilder builder( &ctx );
			mlir::OpBuilder::InsertionGuard guard( builder );
			builder.setInsertionPointToEnd( module->getBody() );
			// clone into original module.
			for( mlir::Operation & op : moduleTactic->getBody()->getOperations() )
				builder.clone( op );

			// initialize pass manager and run passes to lower from linalg to llvm.
			mlir::PassManager pm( &ctx );
			pm.addNestedPass< mlir::FuncOp >( mlir::createLinalgChainPass() );
			pm.addPass( mlir::createLinalgTransformInterpreterPass() );

			if( mlir::failed( pm.run( *module ) ) ) {
				std::cout << "module verification error!\n";
				return FAILED;
			}
			
			// Remove pdl and linalg_transform dialects
			builder.setInsertionPointToStart( module->getBody() );
			module->walk( [ & ]( mlir::pdl::PatternOp op ) {
				op->erase();
			} );
			module->walk( [ & ]( mlir::linalg::transform::SequenceOp op ) {
				op->erase();
			} );
			
			module->dump();

			llvm::InitializeNativeTarget();
			llvm::InitializeNativeTargetAsmPrinter();

			// Register the translation from MLIR to LLVM IR, which must happen before we
			// can JIT-compile.
			mlir::registerLLVMDialectTranslation( ctx );

			// TODO: user-defined option?
			bool enableOpt = false;
			// An optimization pipeline to use within the execution engine.
			auto optPipeline = mlir::makeOptimizingTransformer(
				/*optLevel=*/enableOpt ? 1 : 0, /*sizeLevel=*/0,
				/*targetMachine=*/nullptr );

			mlir::ExecutionEngineOptions engineOpts;
			engineOpts.transformer = optPipeline;
			auto maybeEngine = mlir::ExecutionEngine::create( *module, engineOpts );
			assert( maybeEngine && "failed to construct an execution engine!" );
			if( ! maybeEngine ) {
				return FAILED;
			}
			auto & engine = maybeEngine.get();
			auto invocationResult = engine->invoke( funcName, args );

			if( invocationResult ) {
				std::cout << "JIT invocation failed!\n";
				return FAILED;
			}

			return SUCCESS;
		}

	} // end namespace jit
} // end namespace grb
