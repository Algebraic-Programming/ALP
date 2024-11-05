#include <kernel_operator.h>

using namespace AscendC;

namespace alp {

	__aicore__ inline int32_t max( const int32_t a, const int32_t b ) {
		if( a > b) {
			return a;
		}
		return b;
	}

	__aicore__ inline int32_t RoundUp(int32_t a, int32_t b) {
		return (a + b - 1) / b;
	}

	__aicore__ inline int32_t computeBufferSize( const uint32_t max_n, const uint32_t data_size )
	{
			// Initializing data required by temporary Tensors
			int32_t ascend_el_per_blk = ONE_BLK_SIZE / data_size;
			int32_t elementsPerRepeat = ONE_REPEAT_BYTE_SIZE / data_size;
			int32_t firstMaxRepeat = max_n / elementsPerRepeat;
			int32_t iter1OutputCount = firstMaxRepeat * 2;
			int32_t tmpBufsColsReduce = RoundUp( iter1OutputCount, ascend_el_per_blk ) * ascend_el_per_blk;
			int32_t totWorkSpaceSize = ( ascend_el_per_blk + tmpBufsColsReduce + max_n );

			return totWorkSpaceSize;
	}

	template< typename T3 = half, typename T1, typename T2 >
	__aicore__ inline void DataMove(
		T1 tensorOut,
		T2 tensorIn,
		const uint32_t blocklen
	) {
		DataCopy<T3>( tensorOut, tensorIn, blocklen );
	}

	template< typename T3 = half, typename T1, typename T2 >
	__aicore__ inline void DataMove(
		T1 tensorOut,
		T2 tensorIn,
		const uint32_t nblocks,
		const uint32_t blocklen,
		const uint32_t src_stride,
		const uint32_t dst_stride
	) {
		DataCopyParams dcp;
		dcp.blockCount = nblocks;
		dcp.blockLen   = sizeof( T3 ) * blocklen / 32 ;
		dcp.srcStride  = sizeof( T3 ) * ( src_stride - blocklen ) / 32;
		dcp.dstStride  = sizeof( T3 ) * ( dst_stride - blocklen ) / 32;
		DataCopy<T3>( tensorOut, tensorIn, dcp );
	}

	// Bock (matrix) versions

	__aicore__ inline void BlockSet(
		AscendC::LocalTensor< half > tensorOut,
		half value,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		Duplicate( tensorOut, value, nblocks * blocklen );
	}

	__aicore__ inline void BlockSet(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		DataCopy( tensorOut, tensorIn, nblocks * blocklen );
	}

	__aicore__ inline void BlockExp(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			Exp( tensorOut[ k * blocklen ], tensorIn[ k * blocklen ], blocklen );
		}
	}

	__aicore__ inline void BlockReduceSum(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		AscendC::LocalTensor< half > Work,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		// for( uint32_t k = 0; k < nblocks ; ++k ) {
		// 	ReduceSum( tensorOut[ k ], tensorIn[ k * blocklen ], Work, blocklen );
		// }
		uint32_t repeat = nblocks;
		uint32_t srcRepStride = blocklen;
		srcRepStride = ( sizeof( half ) * srcRepStride ) / 32;
		uint32_t nr = repeat/255;
		if( repeat % 255 ) nr++;
		for( uint32_t ir = 0; ir < nr ; ++ir  ) {
			uint32_t locrepeat = 255;
			if( ir == nr - 1 ) locrepeat = repeat - ir * 255;
			WholeReduceSum<half>(
				tensorOut[ ir * 255 ],
				tensorIn[ ir * 255 * blocklen ],
				blocklen,   // mask
				locrepeat,  // repeat
				1,    // dstStride
				1,    // srcBlkStride
				srcRepStride// srcRepStride
			);
		}

	}

	__aicore__ inline void BlockReduceMax(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		AscendC::LocalTensor< half > Work,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
#ifdef ASCEND910B
		uint32_t repeat = nblocks;
		uint32_t srcRepStride = blocklen;
		srcRepStride = ( sizeof( half ) * srcRepStride ) / 32;
		uint32_t nr = repeat/255;
		if( repeat % 255 ) nr++;
		for( uint32_t ir = 0; ir < nr ; ++ir  ) {
			uint32_t locrepeat = 255;
			if( ir == nr - 1 ) locrepeat = repeat - ir * 255;
			WholeReduceMax<half>(
				tensorOut[ ir * 255 ],
				tensorIn[ ir * 255 * blocklen ],
				blocklen,   // mask
				locrepeat,  // repeat
				1,    // dstStride
				1,    // srcBlkStride
				srcRepStride, // srcRepStride
				ReduceOrder::ORDER_ONLY_VALUE
			);
		}
#else
		// TODO replace with better
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			ReduceMax( tensorOut[ k ], tensorIn[ k * blocklen ], Work, blocklen );
		}
#endif
	}

	__aicore__ inline void BlockBcastMinus(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			Duplicate( Work, tensorInB[ k ].GetValue( 0 ), blocklen ); // broadcast
			Sub( tensorOut[ k * blocklen ], tensorInA[ k * blocklen ], Work, blocklen );
		}
	}

	__aicore__ inline void BlockEwiseMinus(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		Sub( tensorOut, tensorInA, tensorInB, nblocks * blocklen );
	}

	__aicore__ inline void BlockEwiseSum(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		Add( tensorOut, tensorInA, tensorInB, nblocks * blocklen );
	}

	__aicore__ inline void BlockEwiseMax(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		Max( tensorOut, tensorInA, tensorInB, nblocks * blocklen );
	}

	__aicore__ inline void BlockBcastDivide(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			Duplicate( Work, tensorInB[ k ].GetValue( 0 ), blocklen ); // broadcast
			Div( tensorOut[ k * blocklen ], tensorInA[ k * blocklen ], Work, blocklen );
		}
	}

	__aicore__ inline void BlockBcastMultiply(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			Duplicate( Work, tensorInB[ k ].GetValue( 0 ), blocklen ); // broadcast
			Mul( tensorOut[ k * blocklen ], tensorInA[ k * blocklen ], Work, blocklen );
		}
	}

	__aicore__ inline void BlockEwiseMultiply(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t nblocks,
		const uint32_t blocklen
	) {
		Mul( tensorOut, tensorInA, tensorInB, nblocks * blocklen );
	}

	// Vector versions

	__aicore__ inline void VectorSet(
		AscendC::LocalTensor< half > tensorOut,
		half value,
		const uint32_t blocklen
	) {
		Duplicate( tensorOut, value, blocklen );
	}

	__aicore__ inline void VectorSet(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		const uint32_t blocklen
	) {
		DataCopy( tensorOut, tensorIn, blocklen );
	}

	__aicore__ inline void VectorExp(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		const uint32_t blocklen
	) {
		Exp( tensorOut, tensorIn, blocklen );
	}

	__aicore__ inline void VectorReduceSum(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		AscendC::LocalTensor< half > Work,
		const uint32_t blocklen
	) {
		ReduceSum( tensorOut, tensorIn, Work, blocklen );
	}

	__aicore__ inline void VectorReduceMax(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorIn,
		AscendC::LocalTensor< half > Work,
		const uint32_t blocklen
	) {
		ReduceMax( tensorOut, tensorIn, Work, blocklen );
	}

	__aicore__ inline void VectorBcastMinus(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t blocklen
	) {
		Duplicate( Work, tensorInB.GetValue( 0 ), blocklen ); // broadcast
		Sub( tensorOut, tensorInA, Work, blocklen );
	}

	__aicore__ inline void VectorEwiseMinus(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t blocklen
	) {
		Sub( tensorOut, tensorInA, tensorInB, blocklen );
	}

	__aicore__ inline void VectorEwiseSum(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t blocklen
	) {
		Add( tensorOut, tensorInA, tensorInB, blocklen );
	}

	__aicore__ inline void VectorEwiseMax(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t blocklen
	) {
		Max( tensorOut, tensorInA, tensorInB, blocklen );
	}

	__aicore__ inline void VectorBcastDivide(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t blocklen
	) {
		Duplicate( Work, tensorInB.GetValue( 0 ), blocklen ); // broadcast
		Div( tensorOut, tensorInA, Work, blocklen );
	}

	__aicore__ inline void VectorBcastMultiply(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		AscendC::LocalTensor< half > Work,
		const uint32_t blocklen
	) {
		Duplicate( Work, tensorInB.GetValue( 0 ), blocklen ); // broadcast
		Mul( tensorOut, tensorInA, Work, blocklen );
	}

	__aicore__ inline void VectorEwiseMultiply(
		AscendC::LocalTensor< half > tensorOut,
		AscendC::LocalTensor< half > tensorInA,
		AscendC::LocalTensor< half > tensorInB,
		const uint32_t blocklen
	) {
		Mul( tensorOut, tensorInA, tensorInB, blocklen );
	}
}
