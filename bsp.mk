
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
#
GRAPHBLAS_BSP_SOURCES=\
src/graphblas/bsp/collectives.cpp

GRAPHBLAS_HYB_SOURCES=\
src/graphblas/bsp/collectives.cpp

GRAPHBLAS_INCLUDES_BASE+=\
include/graphblas/bsp/spmd.hpp \
include/graphblas/bsp/config.hpp \
include/graphblas/bsp/collectives.hpp \
include/graphblas/bsp/collectives_blas1.hpp \
include/graphblas/bsp/internal-collectives.hpp \
include/graphblas/bsp/collectives_blas1_vec.hpp \
include/graphblas/bsp/collectives_blas1_raw.hpp

LPFengine=mpimsg #other options are: mpirma, ibverbs, and hybrid.
LPFCC=${LPF_INSTALL_PATH}/bin/lpfcc -engine ${LPFengine}   #this is a bit unfortunate-- because exec uses MPI
LPFCPP=${LPF_INSTALL_PATH}/bin/lpfcxx -engine ${LPFengine} #symbols, we cannot generate LPF universal binaries
LPFRUN=${LPF_INSTALL_PATH}/bin/lpfrun -engine ${LPFengine}
LPFCPP11=${LPFCPP} -std=c++11
MPICC=${LPF_INSTALL_PATH}/bin/lpfcc -engine ${LPFengine}
MPICPP=${LPF_INSTALL_PATH}/bin/lpfcxx -engine ${LPFengine}
MPICPP11=${MPICPP} -std=c++11
MANUALRUN=${LPFRUN} -np 1

PARFLAGS=-D_GRB_WITH_LPF -D_GRB_BACKEND=BSP1D -D_GRB_COORDINATES_BACKEND=reference
HYBRIDFLAGS=-D_GRB_WITH_LPF -D_GRB_BACKEND=BSP1D -D_GRB_BSP1D_BACKEND=reference_omp -D_GRB_COORDINATES_BACKEND=reference_omp
PAR_LFLAGS=lib/spmd/libgraphblas.a -llpf_hl -lpthread -lm -ldl -lnuma
HYB_LFLAGS=lib/hybrid/libgraphblas.a -llpf_hl -lpthread -lm -ldl -lnuma

KNNTESTS+=bin/tests/sequential_hook_knn

PRTESTS+=bin/tests/sequential_hook_simple_pagerank \
bin/tests/sequential_hook_pagerank \
bin/tests/automatic_hook_simple_pagerank-openmp

%.hyb.o: %.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${LPFCPP11} ${PERFLAGS} ${HYBRIDFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

%.bsp.o: %.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${LPFCPP11} ${PERFLAGS} ${PARFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

%.hyb.shared.o: %.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${LPFCPP11} -fPIC ${PERFLAGS} ${HYBRIDFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

%.bsp.shared.o: %.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${LPFCPP11} -fPIC ${PERFLAGS} ${PARFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

bin/tests/sequential_hook_knn: tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} -DGRB_LAUNCH_SEQUENTIAL tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/sequential_hook_pagerank: tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} -DGRB_LAUNCH_SEQUENTIAL tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/sequential_hook_simple_pagerank: tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} -DGRB_LAUNCH_SEQUENTIAL -DSIMPLE_PR_TEST tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_hook_simple_pagerank-openmp:  tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} -DGRB_LAUNCH_SEQUENTIAL -DSIMPLE_PR_TEST tests/sequential_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${SEQ_LFLAGS}

