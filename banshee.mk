
#
#   Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

BANSHEE_PATH = /home/dan/workspace/Huawei/banshee

RISCV_PATH = $(BANSHEE_PATH)/riscv-32/bin

RISCV_XLEN    ?= 32
RISCV_ABI     ?= rv$(RISCV_XLEN)imafd
RISCV_PREFIX  ?= riscv$(RISCV_XLEN)-unknown-elf-
RISCV_CC      ?= $(RISCV_PREFIX)gcc
RISCV_CXX     ?= $(RISCV_PREFIX)g++
RISCV_OBJDUMP ?= $(RISCV_PREFIX)objdump
RISCV_OBJCOPY ?= $(RISCV_PREFIX)objcopy
RISCV_AS      ?= $(RISCV_PREFIX)as
RISCV_AR      ?= $(RISCV_PREFIX)ar
RISCV_LD      ?= $(RISCV_PREFIX)ld
RISCV_STRIP   ?= $(RISCV_PREFIX)strip

RISCV_FLAGS    = -march=$(RISCV_ABI)  -mabi=ilp32d -mno-fdiv -mcmodel=medany -g -O3 -ffast-math \
                 -fno-common -ffunction-sections -fno-builtin-printf -fno-exceptions \
                 -ffreestanding -nostdlib -flto -fno-fat-lto-objects \
                 -D_GRB_NO_STDIO -D_GRB_NO_EXCEPTIONS -DNDEBUG #-D_DEBUG
#RISCV_FLAGS    += -DSSR

RISCV_CCFLAGS  = $(RISCV_FLAGS)
RISCV_CXXFLAGS = -fpermissive -std=c++11 -fno-rtti $(RISCV_FLAGS)

BANSHEE_CFLAGS= -D_GRB_BACKEND=banshee -D_GRB_COORDINATES_BACKEND=banshee -D_GRB_NO_LIBNUMA -D_GRB_WITH_BANSHEE \
				${RISCV_CCFLAGS}
BANSHEE_CXXFLAGS= -D_GRB_BACKEND=banshee -D_GRB_COORDINATES_BACKEND=banshee -D_GRB_NO_LIBNUMA -D_GRB_WITH_BANSHEE \
				${RISCV_CXXFLAGS}

BANSHEE_TEST_CXXFLAGS= -D_GRB_BACKEND=banshee -D_GRB_COORDINATES_BACKEND=banshee -D_GRB_NO_LIBNUMA -D_GRB_WITH_BANSHEE \
				  	   -nostartfiles -Wl,-Ttext-segment=0x80000000  \
				       -fno-use-cxa-atexit \
				       ${RISCV_CXXFLAGS}
BOBJDUMP_FLAGS= -dhS --source-comment=\#


BC89 = ${RISCV_PATH}/${RISCV_CC}
BCPP11= ${RISCV_PATH}/${RISCV_CXX}
BAR =gcc-ar					# The RISC-V ar fails
BLD =${RISCV_PATH}/${RISCV_LD}
BOBJDUMP =${RISCV_PATH}/${RISCV_OBJDUMP}


GRAPHBLAS_BANSHEE_SOURCES=\
	src/graphblas/rc.cpp \
	src/graphblas/descriptors.cpp \
	src/graphblas/banshee/init.cpp

GRAPHBLAS_BANSHEE_OBJS1=${GRAPHBLAS_BANSHEE_SOURCES:%.cpp=%.banshee.o}
GRAPHBLAS_BANSHEE_OBJS=${GRAPHBLAS_BANSHEE_OBJS1:%.c=%.banshee.o}

BANSHEE_RUNTIME_SOURCES=\
	3rd/banshee/snRuntime/src/start_banshee.S \
	3rd/banshee/snRuntime/src/start_banshee.c \
	3rd/banshee/snRuntime/src/barrier.c \
	3rd/banshee/snRuntime/src/dma.c \
	3rd/banshee/snRuntime/src/memcpy.c \
	3rd/banshee/snRuntime/src/printf.c \
	3rd/banshee/snRuntime/src/team.c \
	3rd/banshee/snRuntime/src/ssr.c \
	3rd/banshee/snRuntime/src/ssr_v1.c

BANSHEE_RUNTIME_OBJS1=${BANSHEE_RUNTIME_SOURCES:%.c=%.runtime.c.o}
BANSHEE_RUNTIME_OBJS=${BANSHEE_RUNTIME_OBJS1:%.S=%.runtime.S.o}

GRAPHBLAS_INCLUDES+=include/graphblas/banshee/io.hpp \
	include/graphblas/banshee/exec.hpp \
	include/graphblas/banshee/init.hpp \
	include/graphblas/banshee/spmd.hpp \
	include/graphblas/banshee/alloc.hpp \
	include/graphblas/banshee/blas1.hpp \
	include/graphblas/banshee/blas2.hpp \
	include/graphblas/banshee/config.hpp \
	include/graphblas/banshee/matrix.hpp \
	include/graphblas/banshee/vector.hpp \
	include/graphblas/banshee/forward.hpp \
	include/graphblas/banshee/deleters.hpp \
	include/graphblas/banshee/benchmark.hpp \
	include/graphblas/banshee/blas1-raw.hpp \
	include/graphblas/banshee/properties.hpp \
	include/graphblas/banshee/collectives.hpp \
	include/graphblas/banshee/coordinates.hpp \
	include/graphblas/banshee/pinnedvector.hpp

GRAPHBLAS_INCLUDES+=3rd/banshee/snRuntime/include/snrt.h \
	3rd/banshee/vendor/riscv-opcodes/encoding.h \
	3rd/banshee/include/runtime.h

ISNFLAGS=-I3rd/banshee/snRuntime/include/ \
		 -I3rd/banshee/include/ \
		 -I3rd/banshee/vendor/riscv-opcodes/ \
		 -Idatasets/include/

BACKENDS+=banshee

UNITTESTS+=bin/tests/vmxa_banshee \
	bin/tests/vmx_banshee \
	bin/tests/emptyVector_banshee \
	bin/examples/sp_banshee \
	bin/tests/sparse_vxm_banshee \
	bin/tests/sparse_mxv_banshee \
	bin/tests/masked_vxm_banshee \
	bin/tests/masked_mxv_banshee \
	bin/tests/vxm_banshee \
	bin/tests/mxv_banshee \
	bin/tests/printf_simple_banshee \
	bin/tests/printf_fmtint_banshee \
	bin/tests/fcvt_banshee \
	bin/tests/1d_baseline \
	bin/tests/1d_ssr \
	bin/tests/2d_baseline \
	bin/tests/2d_ssr \
	bin/tests/matmul_baseline \
	bin/tests/matmul_ssr \
	bin/tests/simple_pagerank

LIBRARIES+=lib/banshee/libgraphblas.a lib/banshee/libsnRuntime.a

BANSHEE_LFLAGS=lib/banshee/libgraphblas.a lib/banshee/libsnRuntime.a \
               -lm -lgcc

CLEAN_OBJS+=${GRAPHBLAS_BANSHEE_OBJS}  ${BANSHEE_RUNTIME_OBJS}


# Compile the libraries
lib/banshee:
	mkdir -p lib/banshee || true

lib/banshee/libgraphblas.a: ${GRAPHBLAS_BANSHEE_OBJS} | lib/banshee
	${BAR} cr $@ ${GRAPHBLAS_BANSHEE_OBJS}

%.banshee.o: %.cpp ${GRAPHBLAS_INCLUDES}
	${BCPP11} ${BANSHEE_CXXFLAGS} ${IFLAGS} ${ISNFLAGS} $< -c -o $@

%.runtime.S.o: %.S
	${BC89} ${BANSHEE_CFLAGS} ${IFLAGS} ${ISNFLAGS} $< -c -o $@

%.runtime.c.o: %.c
	${BC89} ${BANSHEE_CFLAGS} ${IFLAGS} ${ISNFLAGS} $< -c -o $@

lib/banshee/libsnRuntime.a: ${BANSHEE_RUNTIME_OBJS} | lib/banshee
	${BAR} cr $@ ${BANSHEE_RUNTIME_OBJS}

bin/tests/printf_simple_banshee: 3rd/banshee/snRuntime/tests/printf_simple.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/printf_fmtint_banshee: 3rd/banshee/snRuntime/tests/printf_fmtint.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/fcvt_banshee: 3rd/banshee/tests/fcvt.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/1d_baseline: tests/banshee/1d_baseline.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/1d_ssr: tests/banshee/1d_ssr.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/2d_baseline: tests/banshee/2d_baseline.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/2d_ssr: tests/banshee/2d_ssr.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/matmul_baseline: 3rd/banshee/tests/matmul_baseline.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/matmul_ssr: 3rd/banshee/tests/matmul_ssr.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/blas_banshee: 3rd/banshee/tests/blas_banshee.c ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/emptyVector_banshee: tests/emptyVector.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/emptyVector.cpp -o $@ ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/vmx_banshee: tests/vmx.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/vmx.cpp -o $@ ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/vmx_bin_banshee: tests/banshee/vmx.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/banshee/raw_data_I_J_V_X_Y.S tests/banshee/vmx.cpp -o $@ ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/sparse_vxm_banshee: tests/sparse_vxm.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/sparse_mxv_banshee: tests/sparse_mxv.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/masked_vxm_banshee: tests/masked_vxm.cpp ${GRAPHBLAS_INCLUDES}  ${LIBRARIES+}| dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/masked_mxv_banshee: tests/masked_mxv.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/vmxa_banshee: tests/vmxa.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/vmxa.cpp -o $@ ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/pagerank_banshee: tests/banshee/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/banshee/raw_data_I_J.S "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/conjugate_gradient_banshee: tests/banshee/conjugate_gradient.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/banshee/raw_data_I_J_V.S "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/knn_banshee: tests/banshee/knn.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} tests/banshee/raw_data_I_J.S "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/examples/sp_banshee: examples/sp.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} examples/sp.cpp -o $@ ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/vxm_banshee: tests/launcher/vxm.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

bin/tests/mxv_banshee: tests/launcher/mxv.cpp ${GRAPHBLAS_INCLUDES} ${LIBRARIES+} | dirtree
	${BCPP11} ${BANSHEE_TEST_CXXFLAGS} ${WFLAGS} ${IFLAGS} ${ISNFLAGS} "$<" -o "$@" ${BANSHEE_LFLAGS}
	${BOBJDUMP} ${BOBJDUMP_FLAGS} "$@" >  "$@.S"

