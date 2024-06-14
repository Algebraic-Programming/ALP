
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

//#define _ANALYTIC_MODEL_

#ifdef _ANALYTIC_MODEL_
#include "analytic_model.hpp"
#endif

#include "data_utils.h"

#ifdef __CCE_KT_TEST__

#include "tikicpulib.h"
extern "C" __global__ __aicore__ void custom_##KERNELNAME##(
##CPUFRWDECTENSORALLLIST##,
##CPUFRWDECTHRDGRIDLIST##,
##CPUFRWDECTENSORSIZESLIST##
##ANALYTICMODELFORMALPARAMS##
);

#else

#include "acl/acl.h"
extern void custom_##KERNELNAME##_do(
	uint32_t coreDim, void* l2ctrl, void* stream,
##FRWDECTENSORALLLIST##,
##FRWDECTHRDGRIDLIST##,
##FRWDECTENSORSIZESLIST##
##ANALYTICMODELFORMALPARAMS##
);

#endif



#define DTYPE uint16_t

#define REPS ##REPEATS##

int32_t main(int32_t argc, char* argv[]){
	int rc = 0;
	uint32_t blockDim = ##NTHREADS##;
	uint32_t _p0 = ##NTHREADS##;
##DECLARESIZES##

##DECLARETENSORSIZES##

##DECLAREANALYTICMODELPARAMS##

#ifdef __CCE_KT_TEST__
##CPUDECLARETENSOR##
##CPUREADFILES##

	AscendC::SetKernelMode(KernelMode::AIV_MODE);
	ICPU_RUN_KF(
		custom_##KERNELNAME##,
		blockDim,
##CPUTENSORLIST##,
		blockDim,
##ALLDIMENSIONSLIST##, ##ANALYTICMODELPARAMS##
	); // run the Kernel

##CPUWRITETENSOR##

##CPUFREETENSOR##
#else

	CHECK_ACL(aclInit(nullptr));
	aclrtContext context;
	int32_t deviceId = ##DEVICEID##;
	CHECK_ACL(aclrtSetDevice(deviceId));
	CHECK_ACL(aclrtCreateContext(&context, deviceId));
	aclrtStream stream = nullptr;
	CHECK_ACL(aclrtCreateStream(&stream));

##HOSTDECLARETENSOR##
##HOSTREADFILES##
##DEVICEDECLARETENSOR##

	std::vector< double > meas_vec( REPS );

	for ( auto i = 0; i < REPS; ++i ) {
##HOST2DEVICEMOVE##
		std::cout << "custom_##KERNELNAME## rep " << i << std::endl;
		auto begin = std::chrono::high_resolution_clock::now();
		custom_##KERNELNAME##_do(
			blockDim, nullptr, stream,
##DEVICETENSORLIST##,
			blockDim,
##ALLDIMENSIONSLIST##, ##ANALYTICMODELPARAMS##
		);
		rc = aclrtSynchronizeStream(stream);
		CHECK_ACL(rc);
		if( rc != 0 ) {
			break;
		}
		auto end = std::chrono::high_resolution_clock::now();
		meas_vec[ i ] = static_cast< double >( std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() );
	}

	std::sort( meas_vec.begin(), meas_vec.end() );
	auto avg = std::accumulate( meas_vec.cbegin(), meas_vec.cend(), 0. ) / meas_vec.size();
	auto min = *( std::min_element( meas_vec.cbegin(), meas_vec.cend() ) );
	auto max = *( std::max_element( meas_vec.cbegin(), meas_vec.cend() ) );
	auto size = meas_vec.size();
	auto med = ( size % 2 == 0 ) ? ( meas_vec[ size / 2 - 1 ] + meas_vec[ size / 2 ] ) / 2 : meas_vec[ size / 2 ];
	std::cout << "Measured Time (avg, ms): " << avg * 1e-6 << std::endl;
	std::cout << "              (min, ms): " << min * 1e-6 << std::endl;
	std::cout << "              (max, ms): " << max * 1e-6 << std::endl;
	std::cout << "              (med, ms): " << med * 1e-6 << std::endl;

##DEVICE2HOSTMOVE##
##DEVICEFREETENSOR##
##WRITETENSOR##
##HOSTFREETENSOR##

	CHECK_ACL(aclrtDestroyStream(stream));
	CHECK_ACL(aclrtDestroyContext(context));
	CHECK_ACL(aclrtResetDevice(deviceId));
	CHECK_ACL(aclFinalize());
#endif
	if( rc != 0 ) {
		return 1;
	} else {
		return 0;
	}

}
