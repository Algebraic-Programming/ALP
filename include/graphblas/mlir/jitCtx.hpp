#ifndef ALP_JIT_CTX
#define ALP_JIT_CTX

#include <queue>
#include <unordered_set>

#include <Dialects/LinalgTransform/LinalgTransformOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>

#include <graphblas/backends.hpp>
#include <graphblas/rc.hpp>

namespace grb {
	template< typename D, Backend >
	class Matrix;
}

namespace grb {

	namespace jit {

		// TODO: template
		/// Store a GEMM operation with it's arguments.
		struct GemmNode {
			Matrix< float, Backend::mlir > & C;
			Matrix< float, Backend::mlir > & B;
			Matrix< float, Backend::mlir > & A;
		};

		/// Object to keep an MLIR context and a reference to a module. The idea is
		/// that every alp method (i.e., mxm) can retrieve the current jit context using
		/// `getCurrentJitContext`, which instantiates a static JitContext object or
		/// returns the object itself if already available. JitContext stores a queue of
		/// register methods used for code generation and, just in time, code execution.
		/// It also caches the method already introduced in the module for utility
		/// purposes.
		class JitContext {
		public:
			JitContext() : ctx( mlir::DialectRegistry(), mlir::MLIRContext::Threading::DISABLED ), module( mlir::ModuleOp::create( mlir::OpBuilder( &ctx ).getUnknownLoc() ) ) {
				ctx.getOrLoadDialect< mlir::func::FuncDialect >();
				ctx.getOrLoadDialect< mlir::scf::SCFDialect >();
				ctx.getOrLoadDialect< mlir::arith::ArithmeticDialect >();
				ctx.getOrLoadDialect< mlir::LLVM::LLVMDialect >();
				ctx.getOrLoadDialect< mlir::memref::MemRefDialect >();
				ctx.getOrLoadDialect< mlir::linalg::LinalgDialect >();
				ctx.getOrLoadDialect< mlir::pdl_interp::PDLInterpDialect >();
				ctx.getOrLoadDialect< mlir::linalg::transform::LinalgTransformDialect >();
			};
			~JitContext() = default;
			JitContext( const JitContext & ) = delete;
			JitContext & operator=( const JitContext & ) = delete;

			static JitContext & getCurrentJitContext();

			// build and execute the entire module by code generating all the functions in
			// 'queue' and the jit compiling and executing the entire MLIR module.
			grb::RC buildAndExecute();

			// execute a specific 'funcName' in module passing 'args'.
			template< typename T >
			grb::RC executeFn( llvm::StringRef funcName, llvm::SmallVector< T > args );

			// register a GEMM operation to the queue.
			grb::RC registerMxm( Matrix< float, Backend::mlir > & C, Matrix< float, Backend::mlir > & B, Matrix< float, Backend::mlir > & A );

		private:
			// counter to have unique name for compiled function.
			size_t counter = 1;
			// current MLIR context.
			mlir::MLIRContext ctx;
			// current MLIR module.
			mlir::OwningOpRef< mlir::ModuleOp > module;
			// store the function to be inserted into the module.
			std::queue< GemmNode > queue;
			// cache for already inserted functions in the module.
			llvm::DenseMap< mlir::FunctionType, mlir::FlatSymbolRefAttr > fnInModule;
			// hard-coded flag to generate ranked but unknown dimensions.
			bool castToUnknownDims = false;

			// build a call op to a matmul func.
			mlir::func::CallOp buildMatmulImpl( mlir::OpBuilder & builder, mlir::ValueRange operands, mlir::TypeRange resultType );

			// build or return a func with name 'fnName'.
			mlir::FlatSymbolRefAttr buildOrGetFunc( mlir::OpBuilder & builder, mlir::ValueRange operands, mlir::TypeRange resultType, std::string fnName );
		};

	} // end namespace jit
} // end namespace grb

#include <graphblas/mlir/jitCtx-inl.hpp>

#endif
