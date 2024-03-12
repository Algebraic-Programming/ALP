/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

__aicore__ inline int32_t RoundUp(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

//template < typename T >
class KernelSoftmax {
public:
	__aicore__ inline KernelSoftmax( const uint32_t _p0, const uint32_t _n0, const uint32_t _n1, const uint32_t _n2 ) {
		p0 = _p0;
		p1 = 1;
		p2 = 1;

		n0 = _n0;
		n1 = _n1;
		n2 = _n2;

		block_length0 = ( n0 * n1 * n2 ) / ( p0 * p1 * p2 );
		tile_length0 = ( n1 * n2 ) / BUFFER_NUM;

	}

  __aicore__ inline void Init( GM_ADDR tensor0, GM_ADDR tensor1 )
    {

		// get start index for current core, core parallel
		_tensor0_0Gm.SetGlobalBuffer( (__gm__ half *)tensor0 +  block_length0 * GetBlockIdx(), block_length0);
		_tensor1_0Gm.SetGlobalBuffer((__gm__ half *)tensor1 + block_length0 * GetBlockIdx(), block_length0);

		// Min workspace for reduction ops.
		// Taking the largest btw MaxReduce and SumReduce (ie, MaxReduce) as specified in the AscendC manual
		// at Secs. 8.1.5.10.1 and 8.1.5.10.3
		ascend_el_per_blk = ONE_BLK_SIZE / sizeof( half );
		int32_t elementsPerRepeat = ONE_REPEAT_BYTE_SIZE / sizeof( half );
		int32_t firstMaxRepeat = n2 / elementsPerRepeat;
		int32_t iter1OutputCount = firstMaxRepeat * 2;
		int32_t tmpBufsColsReduce = RoundUp( iter1OutputCount, ascend_el_per_blk ) * ascend_el_per_blk;

		totWorkSpaceSize = (
			ascend_el_per_blk + tmpBufsColsReduce // Output + workspace for Max/SumReduce
		+ n2
		) * sizeof( half );

		pipe.InitBuffer( inQueue_tensor0_0, BUFFER_NUM,  n2 * sizeof( half ) );
		pipe.InitBuffer( outQueue_tensor1_0, BUFFER_NUM, n2 * sizeof( half ) );

    }

    __aicore__ inline void Process()
    {
		// loop count ( including effect of using BUFFER_NUM )
		const uint32_t loopCount0 = ( n0 * BUFFER_NUM ) / p0;
		for (uint32_t i0 = 0; i0 < loopCount0; i0++) {
			uint32_t i = i0;

		pipe.InitBuffer( tempBuf_tensor5_0, totWorkSpaceSize );
		_tensor5_0temp = tempBuf_tensor5_0.Get< half >( );
		pipe.InitBuffer( tempBuf_tensor6_0, totWorkSpaceSize );
		_tensor6_0temp = tempBuf_tensor6_0.Get< half >( );

		pipe.InitBuffer( localBuf_tensor4_0, n1 );
		_tensor4_0Gm = localBuf_tensor4_0.Get< half >( ); // _tensor4_0Gm comes from API


		// This loop comes from axis 1, does not need data movement
		// For now process a tile row by row
		const uint32_t loopCount1 = n1 / BUFFER_NUM;
		for( uint32_t i1 = 0; i1 < n1 ; ++i1 ) {
			CopyIn0(i0,i1);
			Compute0( i1 );
			CopyOut0(i0,i1);
		}
		// free input tensors for reuse
		// inQueue_tensor0_0.FreeTensor( _tensor0Local );
		}
    }


private:

	__aicore__ inline void CopyIn0(
		uint32_t _i0, uint32_t _i1
	) {
		// alloc tensor from queue memory
		_tensor0Local = inQueue_tensor0_0.AllocTensor< half >();
		// copy progress_th tile from global tensor to local tensor
		DataCopy( _tensor0Local, _tensor0_0Gm[ _i0 * n1 * n2 + _i1 * n2  ], n2 );
		// enque input tensors to VECIN queue
		inQueue_tensor0_0.EnQue( _tensor0Local );

		// deque input tensors from VECIN queue
		_tensor0Local = inQueue_tensor0_0.DeQue< half >();
		_tensor1Local = outQueue_tensor1_0.AllocTensor< half >();

	}
    __aicore__ inline void Compute0(uint32_t _i1)
    {
		// apply( _tensor4_0Gm, S_block_in, "max", make_axes(2)  )
		ReduceMax( _tensor5_0temp, _tensor0Local, _tensor5_0temp[ ascend_el_per_blk ], n2, false );
		half max_ = _tensor5_0temp.GetValue( 0 );
		Duplicate( _tensor4_0Gm, max_, n2 ); // broadcast

		// apply( S_block_out, S_block_in, _tensor4_0Gm, "minus", make_axes(2) );
		Sub( _tensor1Local, _tensor0Local,  _tensor4_0Gm, n2 );

		// foldl( S_block_out, "exp" );
		Exp( _tensor1Local, _tensor1Local, n2 );

		// apply( _tensor4_0Gm, S_block_out, "add", make_axes(2) );
		ReduceSum( _tensor6_0temp, _tensor1Local, _tensor6_0temp[ ascend_el_per_blk ], n2 );
		half rec_sum_ = _tensor6_0temp.GetValue( 0 );
		Duplicate( _tensor4_0Gm, rec_sum_, n2 ); // broadcast

		// foldl( S_block_out, _tensor4_0Gm, "divide", make_axes(2) );
		Div( _tensor1Local, _tensor1Local, _tensor4_0Gm, n2 );

    }
	__aicore__ inline void CopyOut0(
		uint32_t _i0, uint32_t _i1
	) {
		outQueue_tensor1_0.EnQue< half >( _tensor1Local );
		// free input tensors for reuse
		inQueue_tensor0_0.FreeTensor( _tensor0Local );

		// deque output tensor from VECOUT queue
		_tensor1Local = outQueue_tensor1_0.DeQue< half >();
		DataCopy( _tensor1_0Gm[ _i0 * n1 * n2 + _i1 * n2 ], _tensor1Local, n2 );
		// free output tensor for reuse
		outQueue_tensor1_0.FreeTensor( _tensor1Local );
    }

	private:
	TPipe pipe;
	// create queues for input, in this case depth is equal to buffer num
	TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_tensor0_0;
	// create queue for output, in this case depth is equal to buffer num
	TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_tensor1_0;

	uint32_t p0, p1, p2, n0, n1, n2;
	uint32_t block_length0, tile_length0;
	int32_t ascend_el_per_blk, totWorkSpaceSize;

	GlobalTensor< half > _tensor0_0Gm, _tensor1_0Gm;
	LocalTensor< half > _tensor0Local;
	LocalTensor< half > _tensor1Local;
	LocalTensor< half > _tensor5_0temp;
	LocalTensor< half > _tensor6_0temp;
	LocalTensor< half > _tensor4_0Gm;

	TBuf< QuePosition::VECCALC > tempBuf_tensor5_0;
	TBuf< QuePosition::VECCALC > tempBuf_tensor6_0;
	TBuf< QuePosition::VECCALC > localBuf_tensor4_0;

};

extern "C" __global__ __aicore__ void custom_KernelSoftmax(
    GM_ADDR in, GM_ADDR out,
	uint32_t _p, uint32_t _n0, uint32_t _n1, uint32_t _n2 ) {
    KernelSoftmax op(_p, _n0, _n1, _n2 );
    op.Init( in, out );
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void custom_KernelSoftmax_do( uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* in, uint8_t* out, uint32_t _p, uint32_t _n0, uint32_t _n1, uint32_t _n2 )
{
  custom_KernelSoftmax<<< blockDim, l2ctrl, stream >>>( in, out, _p, _n0, _n1, _n2 );
}
#endif
