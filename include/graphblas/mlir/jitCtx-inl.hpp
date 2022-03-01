#include <iostream>

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

namespace grb {
	namespace jit {

		template< typename T >
		RC JitContext::executeFn( llvm::StringRef funcName, llvm::SmallVector< T > args ) {
			// initialize pass manager and run passes to lower from linalg to llvm.
			mlir::PassManager pm( &ctx );
			pm.addNestedPass< mlir::FuncOp >( mlir::createLinalgChainPass() );
			pm.addNestedPass< mlir::FuncOp >( mlir::createConvertLinalgToLoopsPass() );
			pm.addPass( mlir::createConvertSCFToCFPass() );
			pm.addPass( mlir::createMemRefToLLVMPass() );
			pm.addNestedPass< mlir::FuncOp >( mlir::arith::createConvertArithmeticToLLVMPass() );
			pm.addPass( mlir::createLowerToLLVMPass() );
			pm.addPass( mlir::createReconcileUnrealizedCastsPass() );

			if( mlir::failed( pm.run( *module ) ) ) {
				std::cout << "module verification error!\n";
				return FAILED;
			}
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
