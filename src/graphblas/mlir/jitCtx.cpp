#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>

#include <graphblas/mlir/jitCtx.hpp>
#include <graphblas/mlir/matrix.hpp>

using namespace grb::jit;
using namespace grb;

JitContext & JitContext::getCurrentJitContext() {
	thread_local JitContext instance;
	return instance;
}

// Build a memref of type f32.
// TODO: Double and others.
static mlir::Type getTensorType( Matrix< float, Backend::mlir > & buff, mlir::OpBuilder & builder ) {
	llvm::SmallVector< int64_t > dims { buff.m, buff.n };
	return mlir::RankedTensorType::get( dims, builder.getF32Type() );
}

// build mxm (aka linalg.matmul) based on memref.
static void buildMxmBody( mlir::OpBuilder & builder, mlir::Location loc, mlir::FuncOp fn, mlir::Block * block ) {
	builder.create< mlir::linalg::MatmulOp >( loc, mlir::ValueRange { fn.getArgument( 1 ), fn.getArgument( 2 ) }, fn.getArgument( 0 ) );
	builder.create< mlir::func::ReturnOp >( loc );
}

static mlir::FunctionType
buildMxmFunctionDef( mlir::OpBuilder & builder, mlir::OwningOpRef< mlir::ModuleOp > & module, const llvm::ArrayRef< mlir::Type > typeOperands, const llvm::ArrayRef< mlir::Type > typeResults = {} ) {
	mlir::OpBuilder::InsertionGuard guard( builder );
	builder.setInsertionPointToEnd( module->getBody() );
	auto fnType = mlir::FunctionType::get( builder.getContext(), typeOperands, typeResults );
	mlir::FuncOp funcOp = builder.create< mlir::FuncOp >( module->getLoc(), "mxm", fnType, llvm::ArrayRef< mlir::NamedAttribute > {} );
	funcOp->setAttr( "llvm.emit_c_interface", mlir::UnitAttr::get( module->getContext() ) );
	mlir::SymbolTable::setSymbolVisibility( funcOp, mlir::SymbolTable::Visibility::Private );
	mlir::Block * entryBlock = funcOp.addEntryBlock();
	builder.setInsertionPointToStart( entryBlock );
	buildMxmBody( builder, module->getLoc(), funcOp, entryBlock );
	return fnType;
}

void JitContext::buildMxm( Matrix< float, Backend::mlir > & C, Matrix< float, Backend::mlir > & B, Matrix< float, Backend::mlir > & A ) {
	mlir::OpBuilder builder( &ctx );

	llvm::SmallVector< mlir::Type, 3 > typeOperands;
	typeOperands.push_back( getTensorType( C, builder ) );
	typeOperands.push_back( getTensorType( B, builder ) );
	typeOperands.push_back( getTensorType( A, builder ) );

	mlir::FunctionType fnType = mlir::FunctionType::get( builder.getContext(), typeOperands, {} );
	// first time we build mxm.
	if( fnInModule.count( "mxm" ) == 0 ) {
		fnType = buildMxmFunctionDef( builder, module, typeOperands );
		fnInModule[ "mxm" ].push_back( fnType );
	} else { // check if we already built the function.
		for( mlir::FunctionType t : fnInModule[ "mxm" ] )
			if( t == fnType )
				return;
		fnType = buildMxmFunctionDef( builder, module, typeOperands );
		fnInModule[ "mxm" ].push_back( fnType );
	}
	// module->dump();
}

RC JitContext::registerMxm( Matrix< float, Backend::mlir > & C, Matrix< float, Backend::mlir > & B, Matrix< float, Backend::mlir > & A ) {
	queue.push( { C, B, A } );
	return SUCCESS;
}

// TODO: remove duplicate code.
RC JitContext::buildAndExecute() {
	// Bind StridedMemRef's basePtr with their position in the function
	// definition.
	llvm::ScopedHashTable< void *, size_t > symTab;
	llvm::ScopedHashTable< size_t, mlir::Value > changeMap;
	llvm::ScopedHashTableScope< void *, size_t > scope( symTab );
	llvm::ScopedHashTableScope< size_t, mlir::Value > scope2( changeMap );

	mlir::OpBuilder builder( &ctx );
	llvm::SmallVector< mlir::Type, 5 > typeOperands;
	mlir::Type typeOutput;
	llvm::SmallVector< void * > descriptors;

	// Make a copy of the current queue. We use the copy to codegenerate the
	// GemmNodes. The original queue is used to build the function definition.
	std::queue< GemmNode > newQueue;
	size_t posInFuncArg = 0;
	while( ! queue.empty() ) {
		GemmNode curr = queue.front();
		queue.pop();
		void * basePtr = &*( curr.C.storage )->basePtr;
		void * descriptor = &*( curr.C.storage );
		if( symTab.count( basePtr ) == 0 ) {
			symTab.insert( basePtr, posInFuncArg++ );
			descriptors.push_back( descriptor );
			typeOperands.push_back( getTensorType( curr.C, builder ) );
			typeOutput = getTensorType( curr.C, builder );
		}
		basePtr = &*( curr.B.storage )->basePtr;
		descriptor = &*( curr.B.storage );
		if( symTab.count( basePtr ) == 0 ) {
			symTab.insert( basePtr, posInFuncArg++ );
			descriptors.push_back( descriptor );
			typeOperands.push_back( getTensorType( curr.B, builder ) );
		}
		basePtr = &*( curr.A.storage )->basePtr;
		descriptor = &*( curr.A.storage );
		if( symTab.count( basePtr ) == 0 ) {
			symTab.insert( basePtr, posInFuncArg++ );
			descriptors.push_back( descriptor );
			typeOperands.push_back( getTensorType( curr.A, builder ) );
		}
		newQueue.push( curr );
	}

	mlir::OpBuilder::InsertionGuard guard( builder );
	builder.setInsertionPointToEnd( module->getBody() );
	auto fnType = mlir::FunctionType::get( &ctx, typeOperands, typeOutput );
	std::string funcName = "moduleFn" + std::to_string( counter++ );
	mlir::FuncOp funcOp = builder.create< mlir::FuncOp >( module->getLoc(), funcName, fnType, llvm::ArrayRef< mlir::NamedAttribute > {} );
	funcOp->setAttr( "llvm.emit_c_interface", mlir::UnitAttr::get( module->getContext() ) );
	mlir::SymbolTable::setSymbolVisibility( funcOp, mlir::SymbolTable::Visibility::Private );
	mlir::Block * entryBlock = funcOp.addEntryBlock();
	builder.setInsertionPointToStart( entryBlock );
	mlir::Location loc = module->getLoc();
	mlir::Value ret = nullptr;
	while( ! newQueue.empty() ) {
		GemmNode curr = newQueue.front();
		newQueue.pop();
		void * basePtr = &*( curr.C.storage )->basePtr;
		size_t posC = symTab.lookup( basePtr );
		basePtr = &*( curr.B.storage )->basePtr;
		size_t posB = symTab.lookup( basePtr );
		basePtr = &*( curr.A.storage )->basePtr;
		size_t posA = symTab.lookup( basePtr );
		mlir::linalg::MatmulOp mxm;
		// Look for indirections of the arguments
		auto A = changeMap.lookup( posA );
		if( ! A )
			A = funcOp.getArgument( posA );
		auto B = changeMap.lookup( posB );
		if( ! B )
			B = funcOp.getArgument( posB );
		auto C = changeMap.lookup( posC );
		if( ! C )
			C = funcOp.getArgument( posC );
		mxm = builder.create< mlir::linalg::MatmulOp >( loc, mlir::ValueRange { A, B }, C );
		ret = mxm.getResult( 0 );
		// Update the indirection of the output
		changeMap.insert( posC, ret );
	}
	// Return the last computation
	builder.create< mlir::func::ReturnOp >( loc, mlir::ValueRange { ret } );
	llvm::errs() << "------Module pre optimization/lowering------\n";
	module->dump();
	llvm::errs() << "--------------------------------------------\n";
	return executeFn( funcName, descriptors );
}
