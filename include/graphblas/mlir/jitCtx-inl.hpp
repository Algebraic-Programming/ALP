#include <iostream>

#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Parser/Parser.h>
#include <Dialects/LinalgTransform/Passes.h>

namespace grb {
	namespace jit {

		template< typename T >
		RC JitContext::executeFn( llvm::StringRef funcName, llvm::SmallVector< T > args ) {
      // read the execution tactic.
      auto tactic = R"(
      pdl.pattern @pdl_target : benefit(1) {
        %args = operands
        %results = types
        %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
        apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
        // TODO: we don't want this, but it is the required terminator for pdl.pattern
        rewrite %0 with "linalg_transform.apply"
      }

      linalg_transform.sequence {
        %0 = match @pdl_target
        tile %0 {sizes = [4, 4, 4], pad = false}
      }
      )";
      mlir::OwningOpRef< mlir::ModuleOp > moduleTactic( 
        mlir::parseSourceString< mlir::ModuleOp >( tactic, &ctx ));
      mlir::OpBuilder builder (&ctx);
      mlir::OpBuilder::InsertionGuard guard( builder );
      builder.setInsertionPointToEnd( module->getBody() );
      // clone into original module.
      for (mlir::Operation &op : moduleTactic->getBody()->getOperations())
        builder.clone(op); 

			// initialize pass manager and run passes to lower from linalg to llvm.
			mlir::PassManager pm( &ctx );
			pm.addNestedPass< mlir::FuncOp >( mlir::createLinalgChainPass() );
      pm.addPass( mlir::createLinalgTransformInterpreterPass() );
    
      if( mlir::failed( pm.run( *module ) ) ) {
        std::cout << "module verification error!\n";
        return FAILED;
      }
      module->dump();  
      mlir::PassManager pm2( &ctx );  
      pm2.addPass( mlir::createDropScheduleFromModulePass() );
			pm2.addNestedPass< mlir::FuncOp >( mlir::createConvertLinalgToLoopsPass() );
			pm2.addPass( mlir::createConvertSCFToCFPass() );
			pm2.addPass( mlir::createMemRefToLLVMPass() );
			pm2.addNestedPass< mlir::FuncOp >( mlir::arith::createConvertArithmeticToLLVMPass() );
			pm2.addPass( mlir::createConvertFuncToLLVMPass() );
			pm2.addPass( mlir::createReconcileUnrealizedCastsPass() );

			if( mlir::failed( pm2.run( *module ) ) ) {
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
