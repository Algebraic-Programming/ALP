
/*
 *   Copyright 2022 Huawei Technologies Co., Ltd.
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

#include <iostream>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include <graphblas.hpp>
#include <utils/assertions.hpp>

using namespace grb;

static mlir::LogicalResult lowerToLLVMDialect( mlir::OwningOpRef< mlir::ModuleOp > & module ) {
	mlir::PassManager pm( module->getContext() );
	pm.addPass( mlir::createMemRefToLLVMPass() );
	pm.addNestedPass< mlir::FuncOp >( mlir::arith::createConvertArithmeticToLLVMPass() );
	pm.addPass( mlir::createLowerToLLVMPass() );
	pm.addPass( mlir::createReconcileUnrealizedCastsPass() );
	return pm.run( *module );
}

static mlir::LogicalResult checkContentMemRef( mlir::OwningMemRef< float, 2 > & m ) {
	for( int64_t i = 0; i < 5; i++ ) {
		for( int64_t j = 0; j < 5; j++ ) {
			if( ( ( i == 2 && j == 1 ) || ( i == 1 && j == 2 ) ) && ( *m )[ i ][ j ] != 42. )
				return mlir::failure();
		}
	}
	return mlir::success();
}

void grb_memref( const size_t & n, RC & rc ) {

	mlir::DialectRegistry registry;
	mlir::MLIRContext context( registry );
	// register dialects
	context.getOrLoadDialect< mlir::StandardOpsDialect >();
	context.getOrLoadDialect< mlir::scf::SCFDialect >();
	context.getOrLoadDialect< mlir::arith::ArithmeticDialect >();
	context.getOrLoadDialect< mlir::LLVM::LLVMDialect >();
	context.getOrLoadDialect< mlir::memref::MemRefDialect >();
	context.disableMultithreading();

	auto str = R"(
    func @rank2_memref(%arg0 : memref<?x?xf32>, 
                       %arg1 : memref<?x?xf32>) attributes { llvm.emit_c_interface } {
    %x = arith.constant 2 : index
    %y = arith.constant 1 : index
    %cst42 = arith.constant 42.0 : f32
    memref.store %cst42, %arg0[%y, %x] : memref<?x?xf32>
    memref.store %cst42, %arg1[%x, %y] : memref<?x?xf32>
    return
  }
  )";

	mlir::OwningOpRef< mlir::ModuleOp > module( mlir::parseSourceString< mlir::ModuleOp >( str, &context ) );
	if( mlir::failed( lowerToLLVMDialect( module ) ) ) {
		llvm::errs() << "module verification error!\n";
		rc = FAILED;
		return;
	}
	module->dump();

	// callback to init memref.
	auto init = [ = ]( float & elt, llvm::ArrayRef< int64_t > indices ) {
		assert( indices.size() == 2 );
		elt = 23;
	};

	// 2-ranked memref.
	int64_t shape[] = { 5, 5 };
	mlir::OwningMemRef< float, 2 > a( shape, {}, init );

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Register the translation from MLIR to LLVM IR, which must happen before we
	// can JIT-compile.
	mlir::registerLLVMDialectTranslation( *module->getContext() );

	bool enableOpt = false;
	// An optimization pipeline to use within the execution engine.
	auto optPipeline = mlir::makeOptimizingTransformer(
		/*optLevel=*/enableOpt ? 1 : 0, /*sizeLevel=*/0,
		/*targetMachine=*/nullptr );

  mlir::ExecutionEngineOptions engineOpts;
  engineOpts.transformer = optPipeline;
	auto maybeEngine = mlir::ExecutionEngine::create( *module, engineOpts );
	assert( maybeEngine && "failed to construct an execution engine" );
	if( ! maybeEngine ) {
		rc = FAILED;
		return;
	}
	auto & engine = maybeEngine.get();
	auto invocationResult = engine->invoke( "rank2_memref", &*a, &*a );
	if( invocationResult ) {
		llvm::errs() << "JIT invocation failed\n";
		rc = FAILED;
		return;
	}

	if( mlir::failed( checkContentMemRef( a ) ) ) {
		llvm::errs() << "Content check failed!\n";
		rc = FAILED;
		return;
	}

	rc = SUCCESS;
}

void grb_constant( const size_t & n, RC & rc ) {
	mlir::DialectRegistry registry;
	mlir::MLIRContext context( registry );
	// register dialects
	context.getOrLoadDialect< mlir::StandardOpsDialect >();
	context.getOrLoadDialect< mlir::scf::SCFDialect >();
	context.getOrLoadDialect< mlir::arith::ArithmeticDialect >();
	context.getOrLoadDialect< mlir::LLVM::LLVMDialect >();
	context.disableMultithreading();

	auto str = R"(
    func @_mlir_ciface_foo(%arg0 : i32) -> i32 {
      return %arg0 : i32
    }
  )";
	mlir::OwningOpRef< mlir::ModuleOp > module( mlir::parseSourceString< mlir::ModuleOp >( str, &context ) );
	mlir::PassManager pm( &context );

	pm.addPass( mlir::createLowerToLLVMPass() );

	if( mlir::failed( pm.run( *module ) ) ) {
		llvm::errs() << "module verification error!\n";
		rc = FAILED;
		return;
	}
	module->dump();

	llvm::InitializeNativeTarget();
	llvm::InitializeNativeTargetAsmPrinter();

	// Register the translation from MLIR to LLVM IR, which must happen before we
	// can JIT-compile.
	mlir::registerLLVMDialectTranslation( *module->getContext() );

	bool enableOpt = false;
	// An optimization pipeline to use within the execution engine.
	auto optPipeline = mlir::makeOptimizingTransformer(
		/*optLevel=*/enableOpt ? 0 : 0, /*sizeLevel=*/0,
		/*targetMachine=*/nullptr );

  mlir::ExecutionEngineOptions engineOpts;
  engineOpts.transformer = optPipeline;
	auto maybeEngine = mlir::ExecutionEngine::create( *module, engineOpts );
	assert( maybeEngine && "failed to construct an execution engine" );
	if( ! maybeEngine ) {
		rc = FAILED;
		return;
	}
	auto & engine = maybeEngine.get();

	// Invoke the JIT-compiled function.
	int32_t r = 0;
	int32_t i = 42;
	auto rr = mlir::ExecutionEngine::Result< int32_t >( r );
	auto invocationResult = engine->invoke( "foo", i, rr );
	if( invocationResult ) {
		llvm::errs() << "JIT invocation failed\n";
		rc = FAILED;
		return;
	}

	if( r != 42 ) {
		rc = FAILED;
		return;
	}
	rc = SUCCESS;
}

int main( int argc, char ** argv ) {
	std::cout << "This is a functional test for MLIR jitter\n";
	Launcher< AUTOMATIC > launcher;
	RC out;
	// TODO: Can I avoid to pass this in if I dont' use it?
	size_t in = 100;
	if( auto er = launcher.exec( &grb_constant, in, out, true ) ) {
		ASSERT_RC_SUCCESS( er );
		std::cerr << "Launching test grb_constant FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		return 255;
	}
	if( launcher.exec( &grb_memref, in, out, true ) != SUCCESS ) {
		std::cerr << "Launching test grb_memref FAILED\n";
		return 255;
	}
	if( out != SUCCESS ) {
		return 255;
	}
	std::cout << "Test OK" << std::endl;
	return 0;
}
