/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "ascendlib.hpp"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

__aicore__ inline int32_t RoundUp(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

//template < typename T >
class KernelSoftmax {
public:
/*
	__aicore__ inline KernelSoftmax(
		const uint32_t _p0,
		const uint32_t _n0,
		const uint32_t _n1,
		const uint32_t _n2,
		const uint32_t _n3,
		const uint32_t _n4,
		const uint32_t _n5
	) {
		p0 = _p0;
		p1 = 1;
		p2 = 1;
		p3 = 1;
		p4 = 1;
		p5 = 1;

		n0 = _n0;
		n1 = _n1;
		n2 = _n2;
		n3 = _n3;
		n4 = _n4;
		n5 = _n5;

		block_length0 = ( n0 * n1 * n2 * n3 * n4 * n5 ) / ( p0 * p1 * p2 * p3 * p4 * p5 );
		tile_length0 = ( n1 * n2 * n3 * n4 * n5 ) / BUFFER_NUM;

	}
*/
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
		int32_t firstMaxRepeat = n3 / elementsPerRepeat;
		int32_t iter1OutputCount = firstMaxRepeat * 2;
		int32_t tmpBufsColsReduce = RoundUp( iter1OutputCount, ascend_el_per_blk ) * ascend_el_per_blk;

		totWorkSpaceSize = (
			ascend_el_per_blk + tmpBufsColsReduce // Output + workspace for Max/SumReduce
		+ n3
		) * sizeof( half );

		pipe.InitBuffer( inQueue_tensor0_0, BUFFER_NUM,  tile_length0 * sizeof( half ) );
		pipe.InitBuffer( outQueue_tensor1_0, BUFFER_NUM, tile_length0 * sizeof( half ) );

    }

    __aicore__ inline void Process()
    {
	    tempBuffInit();

	    // loop count ( including effect of using BUFFER_NUM )

	    const uint32_t loopCount0 =  n0  / p0;
	    for (uint32_t i0 = 0; i0 < loopCount0; i0++) {

		    for (uint32_t i1 = 0; i1 < n1; i1++) {
			    // no loop i2
			    for (uint32_t i3 = 0; i3 < n3; i3++) {
				    for (uint32_t i4 = 0; i4 < n4; i4++) {
					    // no loop i5


					    uint32_t gm_pointer = i0*n1*n2*n3*n4*n5  +  i1*n2*n3*n4*n5   +  i3*n4*n5  +  i4*n5;
					    uint32_t blocklen=n5;
					    uint32_t stride=n3*n4*n5;
					    uint32_t nblocks=n2;

					    CopyIn0(gm_pointer,blocklen,stride,nblocks);

					    alp::BlockReduceMax( _tensor4_0Gm, _tensor0Local, _tensor5_0temp[ ascend_el_per_blk ], nblocks, blocklen );

					    alp::BlockBcastMinus( _tensor1Local, _tensor0Local, _tensor4_0Gm, _tensor5_0temp, nblocks, blocklen );

					    alp::BlockExp( _tensor1Local, _tensor1Local, nblocks, blocklen );

					    alp::BlockReduceSum( _tensor4_0Gm, _tensor1Local, _tensor5_0temp[ ascend_el_per_blk ], nblocks, blocklen );

					    alp::BlockBcastDivide( _tensor1Local, _tensor1Local, _tensor4_0Gm, _tensor5_0temp, nblocks, blocklen );

					    CopyOut0(gm_pointer,blocklen,stride,nblocks);

				    }
			    }
		    }
	    }
    }


private:


	__aicore__ inline void tempBuffInit() {

		pipe.InitBuffer( tempBuf_tensor5_0, totWorkSpaceSize );
		_tensor5_0temp = tempBuf_tensor5_0.Get< half >( );

		// pipe.InitBuffer( tempBuf_tensor6_0, totWorkSpaceSize );
		// _tensor6_0temp = tempBuf_tensor6_0.Get< half >( );

		pipe.InitBuffer( localBuf_tensor4_0, n2 );
		_tensor4_0Gm = localBuf_tensor4_0.Get< half >( ); // _tensor4_0Gm comes from API
	}


	__aicore__ inline void CopyIn0(
		uint32_t gm_pointer, uint32_t  blocklen, uint32_t stride, uint32_t nblocks
	)
    {
		// alloc tensor from queue memory
		_tensor0Local = inQueue_tensor0_0.AllocTensor< half >();
		// copy progress_th tile from global tensor to local tensor

		// DataCopyParams dcpy_param;
		// dcpy_param.blockCount=nblocks;
		// dcpy_param.blockLen  =blocklen;
		// dcpy_param.srcStride =stride;
		// dcpy_param.dstStride =0;
		// DataCopy( _tensor0Local, _tensor0_0Gm[ gm_pointer ], dcpy_param );
		// DataCopy( _tensor0Local, _tensor0_0Gm[ gm_pointer ], blocklen );
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			DataCopy( _tensor0Local[ k*blocklen ], _tensor0_0Gm[ gm_pointer + k*stride ], blocklen );
		}

		// enque input tensors to VECIN queue
		inQueue_tensor0_0.EnQue( _tensor0Local );

		// deque input tensors from VECIN queue
		_tensor0Local = inQueue_tensor0_0.DeQue< half >();
		_tensor1Local = outQueue_tensor1_0.AllocTensor< half >();


    }

    __aicore__ inline void CopyOut0(
	    		uint32_t gm_pointer, uint32_t  blocklen, uint32_t stride, uint32_t nblocks
    )
    {
		outQueue_tensor1_0.EnQue< half >( _tensor1Local );
		// free input tensors for reuse
		inQueue_tensor0_0.FreeTensor( _tensor0Local );

		// deque output tensor from VECOUT queue
		_tensor1Local = outQueue_tensor1_0.DeQue< half >();

		// DataCopyParams dcpy_param;
		// dcpy_param.blockCount=nblocks;
		// dcpy_param.blockLen  =blocklen;
		// dcpy_param.srcStride =0;
		// dcpy_param.dstStride =stride;
		// DataCopy( _tensor1_0Gm[ gm_pointer ], _tensor1Local, dcpy_param );
		// DataCopy( _tensor1_0Gm[ gm_pointer ], _tensor1Local, blocklen );
		for( uint32_t k = 0; k < nblocks ; ++k ) {
			DataCopy( _tensor1_0Gm[ gm_pointer + k*stride ], _tensor1Local[ k*blocklen ], blocklen );
		}

		// free output tensor for reuse
		outQueue_tensor1_0.FreeTensor( _tensor1Local );
    }

	private:
	TPipe pipe;
	// create queues for input, in this case depth is equal to buffer num
	TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_tensor0_0;
	// create queue for output, in this case depth is equal to buffer num
	TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_tensor1_0;

	uint32_t p0, p1, p2, p3, p4, p5;
	uint32_t n0, n1, n2, n3, n4, n5;
	uint32_t block_length0, tile_length0;
	int32_t ascend_el_per_blk, totWorkSpaceSize;

	GlobalTensor< half > _tensor0_0Gm, _tensor1_0Gm;
	LocalTensor< half > _tensor0Local;
	LocalTensor< half > _tensor1Local;
	LocalTensor< half > _tensor5_0temp;
	// LocalTensor< half > _tensor6_0temp;
	LocalTensor< half > _tensor4_0Gm;

	TBuf< QuePosition::VECCALC > tempBuf_tensor5_0;
	// TBuf< QuePosition::VECCALC > tempBuf_tensor6_0;
	TBuf< QuePosition::VECCALC > localBuf_tensor4_0;

};

extern "C" __global__ __aicore__ void custom_KernelSoftmax(
    GM_ADDR in, GM_ADDR out,
	uint32_t _p, uint32_t _n0, uint32_t _n1, uint32_t _n2, uint32_t _n3, uint32_t _n4, uint32_t _n5 ) {
    KernelSoftmax op(_p, _n0, _n1, _n2, _n3, _n4, _n5 );
    op.Init( in, out );
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void custom_KernelSoftmax_do(
	uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* in, uint8_t* out, uint32_t _p,
	uint32_t _n0, uint32_t _n1, uint32_t _n2, uint32_t _n3, uint32_t _n4, uint32_t _n5
) {
  custom_KernelSoftmax<<< blockDim, l2ctrl, stream >>>( in, out, _p, _n0, _n1, _n2, _n3, _n4, _n5 );
}
#endif
