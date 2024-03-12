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

#include "data_utils.h"

#include "acl/acl.h"

extern void custom_KernelOnlineSoftmax_do(
	uint32_t coreDim, void* l2ctrl, void* stream,
	uint8_t *param_Sin, uint8_t *param_Sout, uint8_t *param_m, uint8_t *param_l,
	uint32_t _p, uint32_t n0,
	uint32_t n1, uint32_t n2, uint32_t n3 );

#define DTYPE uint16_t

constexpr uint32_t n0=16;
constexpr uint32_t n1=32;
constexpr uint32_t n2=16;
constexpr uint32_t n3=16;

constexpr uint32_t N2 = n0*n2;
constexpr uint32_t N3 = n0*n1*n2*n3;

#define REPS 20

int32_t main(int32_t argc, char* argv[])
{
    size_t param_m_FileSize = N2 * sizeof( DTYPE );
    size_t param_l_FileSize = N2 * sizeof( DTYPE );
    size_t param_Sin_FileSize = N3 * sizeof( DTYPE );
    size_t param_Sout_FileSize = N3 * sizeof( DTYPE );
    uint32_t blockDim = 4;

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    ///////////////   allocate on host ////////////////////////

    uint8_t *param_m_Host;
    CHECK_ACL(aclrtMallocHost((void**)(&param_m_Host), param_m_FileSize));

    uint8_t *param_l_Host;
    CHECK_ACL(aclrtMallocHost((void**)(&param_l_Host), param_l_FileSize));

    uint8_t *param_Sin_Host;
    CHECK_ACL(aclrtMallocHost((void**)(&param_Sin_Host), param_Sin_FileSize));
    ReadFile("./input/s0_gm.bin", param_Sin_FileSize, param_Sin_Host, param_Sin_FileSize);

    uint8_t *param_Sout_Host;
    CHECK_ACL(aclrtMallocHost((void**)(&param_Sout_Host), param_Sout_FileSize));

    ///////////////   allocate on device ////////////////////////

    uint8_t *param_m_Device;
    CHECK_ACL(aclrtMalloc((void**)&param_m_Device, param_m_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *param_l_Device;
    CHECK_ACL(aclrtMalloc((void**)&param_l_Device, param_l_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *param_Sin_Device;
    CHECK_ACL(aclrtMalloc((void**)&param_Sin_Device, param_Sin_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *param_Sout_Device;
    CHECK_ACL(aclrtMalloc((void**)&param_Sout_Device, param_Sout_FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector< double > meas_vec( REPS );

    for ( auto i = 0; i < REPS; ++i ) {
	CHECK_ACL(aclrtMemcpy(param_Sin_Device, param_Sin_FileSize, param_Sin_Host, param_Sin_FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

        std::cout << "Softmax rep " << i << std::endl;
        auto begin = std::chrono::high_resolution_clock::now();

        custom_KernelOnlineSoftmax_do(
		blockDim, nullptr, stream,
		param_Sin_Device, param_Sout_Device,
		param_m_Device, param_l_Device,
		blockDim, n0, n1, n2, n3
	);
	CHECK_ACL(aclrtSynchronizeStream(stream));

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


    CHECK_ACL(aclrtMemcpy(param_m_Host, param_m_FileSize, param_m_Device, param_m_FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(param_l_Host, param_l_FileSize, param_l_Device, param_l_FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(param_Sout_Host, param_Sout_FileSize, param_Sout_Device, param_Sout_FileSize, ACL_MEMCPY_DEVICE_TO_HOST));

    WriteFile("./output/output_s1.bin", param_Sout_Host, param_Sout_FileSize);
    CHECK_ACL(aclrtFreeHost(param_Sin_Host));
    CHECK_ACL(aclrtFreeHost(param_Sout_Host));

    CHECK_ACL(aclrtFree(param_Sin_Device));
    CHECK_ACL(aclrtFree(param_Sout_Device));

    WriteFile("./output/output_m.bin", param_m_Host, param_m_FileSize);
    WriteFile("./output/output_l.bin", param_l_Host, param_l_FileSize);
    CHECK_ACL(aclrtFreeHost(param_l_Host));
    CHECK_ACL(aclrtFreeHost(param_m_Host));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return 0;
}
