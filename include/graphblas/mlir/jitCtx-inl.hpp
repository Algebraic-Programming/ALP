#include <fstream>
#include <iostream>

#include <Dialects/LinalgTransform/Passes.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/LegacyPassNameParser.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Dialect/PDL/IR/PDLOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

struct Options {
	llvm::cl::OptionCategory optFlags { "opt-like flags" };
	//   CLI list of pass information
	llvm::cl::list< const llvm::PassInfo *, bool, llvm::PassNameParser > llvmPasses { llvm::cl::desc( "LLVM passes to run" ), llvm::cl::cat( optFlags ) };
};

namespace grb {
	namespace jit {

		template< typename T >
		RC JitContext::executeFn( llvm::StringRef funcName, llvm::SmallVector< T > args ) {
			// read the execution tactic.
			std::string errorMessage;
			auto memoryBuffer = mlir::openInputFile( "pdl.txt", &errorMessage );
			if( ! memoryBuffer ) {
				llvm::errs() << errorMessage << "\n";
				return FAILED;
			}
			// Tell sourceMgr about this buffer, the parser will pick it up.
			llvm::SourceMgr sourceMgr;
			sourceMgr.AddNewSourceBuffer( std::move( memoryBuffer ), llvm::SMLoc() );
			mlir::OwningOpRef< mlir::ModuleOp > moduleTactic( mlir::parseSourceFile< mlir::ModuleOp >( sourceMgr, &ctx ) );

			mlir::OpBuilder builder( &ctx );
			mlir::OpBuilder::InsertionGuard guard( builder );
			builder.setInsertionPointToEnd( module->getBody() );

			// TODO: Can we avoid this copy?
			// See: runTransformModuleOnOperation on the sanbox.
			// clone into original module.
			for( mlir::Operation & op : moduleTactic->getBody()->getOperations() )
				builder.clone( op );

			// initialize pass manager and run passes to lower from linalg to llvm.
			mlir::PassManager pm( &ctx );
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
			bool enableOpt = true;

			auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
			if( ! tmBuilderOrError ) {
				llvm::errs() << "Failed to create a JITTargetMachineBuilder for the host\n";
				return FAILED;
			}
			auto tmOrError = tmBuilderOrError->createTargetMachine();
			if( ! tmOrError ) {
				llvm::errs() << "Failed to create a TargetMachine for the host\n";
				return FAILED;
			}

			// Options for machine code generation
			const char * llc_options[] = { "llc", "--loop-prefetch-writes" };

			Options options;
			llvm::cl::ParseCommandLineOptions( 2, llc_options, "LLC options\n" );

			// Generate vector of pass information
			mlir::SmallVector< const llvm::PassInfo *, 2 > passes;

			for( unsigned i = 0, e = options.llvmPasses.size(); i < e; ++i )
				passes.push_back( options.llvmPasses[ i ] );

			// An optimization pipeline to use within the execution engine.
			auto optPipeline = mlir::makeLLVMPassesTransformer( passes, enableOpt ? 3 : 0, tmOrError->get() );

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
