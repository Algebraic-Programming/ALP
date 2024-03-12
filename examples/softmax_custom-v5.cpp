/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "ascendlib.hpp"

#define TMP_MXM

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

__aicore__ inline int32_t RoundUp(int32_t a, int32_t b) {
    return (a + b - 1) / b;
}

//template < typename T >
class KernelOnlineSoftmax {
public:
	__aicore__ inline KernelOnlineSoftmax( const uint32_t _p0, const uint32_t _n0, const uint32_t _n1, const uint32_t _n2, const uint32_t _n3 ) {
		p0 = _p0;
		p1 = 1;
		p2 = 1;
		p3 = 1;

		n0 = _n0; // Tr
		n1 = _n1; // Tc
		n2 = _n2; // Br
		n3 = _n3; // Bc  // Sij(Br,Bc)

		block_length_out1 = ( n0 * n2 ) / ( p0  * p2 );
		tile_length_out1 = ( n2 ) / BUFFER_NUM;

		block_length_out2 = ( n0 * n2 ) / ( p0  * p2 );
		tile_length_out2 = ( n2 ) / BUFFER_NUM;
	}

  __aicore__ inline void Init(
	  GM_ADDR tensorOut1, GM_ADDR tensorOut2,
	  GM_ADDR tensorS0, GM_ADDR tensorS1
  ) {

	  // get start index for current core, core parallel

	  _tensorOutm_Gm.SetGlobalBuffer( (__gm__ half *)tensorOut1 +  block_length_out1 * GetBlockIdx(), block_length_out1 );
	  _tensorOutl_Gm.SetGlobalBuffer( (__gm__ half *)tensorOut2 +  block_length_out2 * GetBlockIdx(), block_length_out2 );
	  pipe.InitBuffer( outQueue_tensor_l, BUFFER_NUM,  tile_length_out1 * sizeof( half ) );
	  pipe.InitBuffer( outQueue_tensor_m, BUFFER_NUM,  tile_length_out2 * sizeof( half ) );

	  uint32_t block_length_in_s = ( n0 * n1 * n2 * n3 ) / ( p0 * p1 * p2 * p3 );
	  uint32_t tile_length_in_s = ( n1 * n2 * n3 ) / BUFFER_NUM;

	  _tensorS0_Gm.SetGlobalBuffer( (__gm__ half *)tensorS0 + block_length_in_s * GetBlockIdx(), block_length_in_s );
	  _tensorS1_Gm.SetGlobalBuffer( (__gm__ half *)tensorS1 + block_length_in_s * GetBlockIdx(), block_length_in_s );
	  pipe.InitBuffer(  inQueue_tensor_S0,  BUFFER_NUM,  n1*n2*n3 * sizeof( half ) );
	  pipe.InitBuffer( outQueue_tensor_S1,  BUFFER_NUM,  n1*n2*n3 * sizeof( half ) );


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


	    pipe.InitBuffer( tempBuf_alltensors, totWorkSpaceSize + 3 * n2 );
	    _tensor_Work4     = tempBuf_alltensors.Get< half >();

	    // 0:
	    // ascend_el_per_blk: TEMP / HIDDEN 
	    // rowmaxS: totWorkSpaceSize
	    // mi_old: rowmaxS + n2;
	    // expmidiff: mi_old + n2;
	    //
	    //
	    expmidiff= totWorkSpaceSize;
	    mi_old = rowmaxS + n2;
	    rowmaxS = mi_old + n2;

    }

    __aicore__ inline void Process()
    {
	    half Zero = 0;

	    const uint32_t loopCount0 =  n0  / p0;
	    for (uint32_t i0 = 0; i0 < loopCount0; i0++) {

		    //*******************************//
		    // auto m_block_out = mtensorout.getView(); // T(2)
		    _tensor_m_i0 = outQueue_tensor_m.AllocTensor< half >();
		    outQueue_tensor_m.EnQue( _tensor_m_i0 );
		    _tensor_m_i0 = outQueue_tensor_m.DeQue< half >();


		    // alp::set( m_block_out, -alp::Infinity<double> );
		    half mInf = -65504.0;                 //----
		    Duplicate( _tensor_m_i0, mInf, n2 ); //----		//TODO SET scalar


		    // DataCopy here
		    //*******************************//
		    // auto l_block_out = ltensorout.getView(); // T(2)
		    _tensor_l_i0 = outQueue_tensor_l.AllocTensor< half >();
		    outQueue_tensor_l.EnQue( _tensor_l_i0 );
		    _tensor_l_i0 = outQueue_tensor_l.DeQue< half >();


		    // alp::set( l_block_out, alp::Zero<double>  );
		    Duplicate( _tensor_l_i0, Zero, n2 );  //----	//TODO SET scalar


		    // DataCopy here
		    //*******************************//

		    const uint32_t loopCount1 = n1 ;
		    for( uint32_t i1 = 0; i1 < n1; i1++ ) {

			    _tensorSijIn = inQueue_tensor_S0.AllocTensor< half >();
			    _tensorSijOut = outQueue_tensor_S1.AllocTensor< half >();

			    // alp::Tensor Sij(       alp::Datatype::FP16, alp::make_axes( 2, 3 ) );

			    // alp::Tensor Temp(      alp::Datatype::FP16, alp::make_axes( 2, 3 ) );

			    // alp::Tensor rowmaxS(   alp::Datatype::FP16, alp::make_axes( 2 ) );

			    // alp::Tensor mi_old(    alp::Datatype::FP16, alp::make_axes( 2 ) );

			    // alp::Tensor expmidiff( alp::Datatype::FP16, alp::make_axes( 2 ) );


			    DataCopy( _tensorSijIn, _tensorS0_Gm[ i0*n1*n2*n3 + i1*n2*n3 ],  n2*n3  );
			    inQueue_tensor_S0.EnQue( _tensorSijIn );
			    _tensorSijIn = inQueue_tensor_S0.DeQue< half >();

			    // +++++++++++++++++++++++++++++ //
			    // Online softmax

			    // set( mi_old, m_block_out);
			    DataCopy( _tensor_Work4[mi_old], _tensor_m_i0, n2 );

			    // apply( rowmaxS, S_block_in, "max", make_axes( 3 ) );
			    alp::BlockReduceMax( _tensor_Work4[rowmaxS], _tensorSijIn, _tensor_Work4[ ascend_el_per_blk ], n2, n3 );

			    // foldl( m_block_out, rowmaxS, "max" );
			    Max( _tensor_m_i0, _tensor_m_i0, _tensor_Work4[rowmaxS], n2 );

			    // // apply( S_block_out, S_block_in, m_block_out, "minus", make_axes( 3 ) );
			    alp::BlockBcastMinus( _tensorSijOut, _tensorSijIn, _tensor_m_i0, _tensor_Work4, n2, n3 );

			    // Si=np.exp(Si)
			    alp::BlockExp( _tensorSijOut, _tensorSijOut, n2, n3 );

			    // expmidiff=np.exp(mi_old-mtensor[i,:])
			    Duplicate( _tensor_Work4[expmidiff], Zero, n2 );  //----
			    Sub( _tensor_Work4[expmidiff], _tensor_Work4[mi_old], _tensor_m_i0, n2 );
			    Exp( _tensor_Work4[expmidiff], _tensor_Work4[expmidiff], n2 );

			    // foldl( l_block_out, expmidiff, "times" );
			    Mul( _tensor_l_i0, _tensor_l_i0, _tensor_Work4[expmidiff], n2 );

			    // foldl( l_block_out, S_block_out, "add", make_axes( 3 ) );
			    alp::BlockReduceSum( _tensor_Work4[rowmaxS], _tensorSijOut, _tensor_Work4[ ascend_el_per_blk ] , n2, n3 );
			    Add( _tensor_l_i0, _tensor_l_i0, _tensor_Work4[rowmaxS], n2 );

			    // +++++++++++++++++++++++++++++ //

			    outQueue_tensor_S1.EnQue( _tensorSijOut );
			    _tensorSijOut = outQueue_tensor_S1.DeQue< half >();
			    DataCopy( _tensorS1_Gm[ i0*n1*n2*n3 + i1*n2*n3 ], _tensorSijOut,  n2*n3  );

			    inQueue_tensor_S0.FreeTensor( _tensorSijIn );
			    outQueue_tensor_S1.FreeTensor( _tensorSijOut );
		    }

    			// // Uptade ltensor
    			// // CopyOUT ltensor & mtensor

		    DataCopy(  _tensorOutm_Gm[ i0 * n2 ], _tensor_m_i0, n2 );
		    DataCopy(  _tensorOutl_Gm[ i0 * n2 ], _tensor_l_i0, n2 );

		    outQueue_tensor_m.FreeTensor( _tensor_m_i0 );
		    outQueue_tensor_l.FreeTensor( _tensor_l_i0 );

    		}

    }


private:


	private:

	uint32_t p0, p1, p2, p3;
	uint32_t n0, n1, n2, n3;
	uint32_t block_length_out1, tile_length_out1;
	uint32_t block_length_out2, tile_length_out2;

	int32_t ascend_el_per_blk, totWorkSpaceSize;
	int32_t rowmaxS, mi_old, expmidiff;

	TPipe pipe;

	// create queue for output, in this case depth is equal to buffer num
	TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_tensor_S1;
	TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_tensor_S0;

	TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_tensor_m;
	TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_tensor_l;


	GlobalTensor< half > _tensorOutm_Gm;
	GlobalTensor< half > _tensorOutl_Gm;
	GlobalTensor< half >  _tensorS0_Gm;
	GlobalTensor< half >  _tensorS1_Gm;

	LocalTensor< half > _tensorSijOut;
	LocalTensor< half > _tensorSijIn;

	LocalTensor< half > _tensor_m_i0;
	LocalTensor< half > _tensor_l_i0;

	LocalTensor< half > _tensor_Work4;



	TBuf< QuePosition::VECCALC > tempBuf_alltensors;
};

extern "C" __global__ __aicore__ void custom_KernelOnlineSoftmax(
	GM_ADDR out1, GM_ADDR out2,
	GM_ADDR S0, GM_ADDR S1,
	uint32_t _p, uint32_t _n0, uint32_t _n1, uint32_t _n2, uint32_t _n3
) {
	KernelOnlineSoftmax op(_p, _n0, _n1, _n2, _n3 );
	op.Init(
		out1, out2,
		S0, S1
	);  // TODO fix Init
	op.Process(); // TODO fix Process
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void custom_KernelOnlineSoftmax_do(
	uint32_t blockDim, void* l2ctrl, void* stream,
	uint8_t* out1, uint8_t* out2, 
	uint8_t* s0, uint8_t* s1,
	uint32_t _p, uint32_t _n0, uint32_t _n1, uint32_t _n2, uint32_t _n3
) {
  custom_KernelOnlineSoftmax<<< blockDim, l2ctrl, stream >>>(
	  out1, out2,
	  s0, s1,
	  _p, _n0, _n1, _n2, _n3
  );
}
#endif
