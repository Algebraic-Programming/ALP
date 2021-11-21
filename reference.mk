
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

CFLAGS+=-D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP
OMPFLAGS=-D_GRB_BACKEND=reference_omp
SEQ_LFLAGS=lib/sequential/libgraphblas.a -lnuma

GRAPHBLAS_SOURCES=\
src/graphblas/rc.cpp \
src/graphblas/descriptors.cpp \
src/graphblas/reference/init.cpp \
src/graphblas/utils/hpparser.c \
src/graphblas/reference/config.cpp

GRAPHBLAS_OBJS1=${GRAPHBLAS_SOURCES:%.cpp=%.o}
GRAPHBLAS_OBJS=${GRAPHBLAS_OBJS1:%.c=%.o}
GRAPHBLAS_SHARED_OBJS=${GRAPHBLAS_OBJS:%.o=%.shared.o}

GRAPHBLAS_INCLUDES_BASE+=\
include/graphblas/omp/config.hpp \
include/graphblas/reference/io.hpp \
include/graphblas/reference/exec.hpp \
include/graphblas/reference/init.hpp \
include/graphblas/reference/spmd.hpp \
include/graphblas/reference/alloc.hpp \
include/graphblas/reference/blas1.hpp \
include/graphblas/reference/blas2.hpp \
include/graphblas/reference/blas3.hpp \
include/graphblas/reference/config.hpp \
include/graphblas/reference/matrix.hpp \
include/graphblas/reference/vector.hpp \
include/graphblas/reference/forward.hpp \
include/graphblas/reference/benchmark.hpp \
include/graphblas/reference/blas1-raw.hpp \
include/graphblas/reference/properties.hpp \
include/graphblas/reference/collectives.hpp \
include/graphblas/reference/coordinates.hpp \
include/graphblas/reference/pinnedvector.hpp \
include/graphblas/reference/compressed_storage.hpp

BACKENDS=reference reference_omp
BACKENDCOMPILER=\"${CXX}\" \"${CXX}\"
BACKENDCFLAGS=\"-D_GRB_BACKEND=reference\" \"-D_GRB_BACKEND=reference_omp\"
BACKENDLFLAGS=\"${GRB_INSTALL_PATH}/lib/sequential/libgraphblas.a -lnuma\" \"${GRB_INSTALL_PATH}/lib/sequential/libgraphblas.a -lnuma\"
BACKENDRUNENV=\"\" \"\"
BACKENDRUNNER=\"\" \"\"
COMMONCFLAGS=-D_GRB_WITH_REFERENCE -D_GRB_WITH_OMP -fopenmp

UNITTESTS=bin/tests/add15d \
bin/tests/mul15i \
bin/tests/mul15m \
bin/tests/add15m \
bin/tests/distribution_bsp1d \
bin/tests/vmxa_reference \
bin/tests/vmxa_reference_omp \
bin/tests/vmx_reference \
bin/tests/vmx_reference_omp \
bin/tests/thread_local_storage \
bin/tests/emptyVector_reference \
bin/tests/emptyVector_reference_omp \
bin/examples/sp_reference \
bin/examples/sp_reference_omp \
bin/tests/parserTest \
bin/tests/compareParserTest \
bin/tests/hpparser \
bin/tests/sparse_vxm_reference \
bin/tests/sparse_mxv_reference \
bin/tests/masked_vxm_reference \
bin/tests/masked_mxv_reference \
bin/tests/sparse_vxm_reference_omp \
bin/tests/sparse_mxv_reference_omp \
bin/tests/masked_vxm_reference_omp \
bin/tests/masked_mxv_reference_omp \
bin/tests/vxm_reference \
bin/tests/mxv_reference_omp \
bin/tests/vxm_reference_omp \
bin/tests/mxv_reference \
bin/tests/ewiseapply_reference \
bin/tests/ewiseapply_reference_omp \
bin/tests/zip_reference \
bin/tests/zip_reference_omp \
bin/tests/dot_reference \
bin/tests/dot_reference_omp \
bin/tests/mxm_reference \
bin/tests/mxm_reference_omp \
bin/tests/copyVector_reference \
bin/tests/copyVector_reference_omp \
bin/tests/swapVector_reference \
bin/tests/swapVector_reference_omp \
bin/tests/moveVector_reference \
bin/tests/moveVector_reference_omp \
bin/tests/moveMatrix_reference \
bin/tests/moveMatrix_reference_omp \
bin/tests/stdVector_reference \
bin/tests/stdVector_reference_omp \
bin/tests/RBGaussSeidel_reference \
bin/tests/RBGaussSeidel_reference_omp \
bin/tests/muladd_reference_omp \
bin/tests/muladd_reference \
bin/tests/masked_muladd_reference_omp \
bin/tests/masked_muladd_reference \
bin/tests/buildVector_reference \
bin/tests/buildVector_reference_omp \
bin/tests/copyAndAssignVectorIterator_reference \
bin/tests/copyAndAssignVectorIterator_reference_omp \
bin/tests/set_reference \
bin/tests/set_reference_omp \
bin/tests/vectorToMatrix_reference \
bin/tests/vectorToMatrix_reference_omp \
bin/tests/clearMatrix_reference \
bin/tests/clearMatrix_reference_omp \
bin/tests/argmin_reference \
bin/tests/argmin_reference_omp \
bin/tests/argmax_reference \
bin/tests/argmax_reference_omp \
bin/tests/automatic_launch_conjugate_gradient_reference \
bin/tests/automatic_launch_conjugate_gradient_reference_omp \
bin/tests/automatic_launch_graphchallenge_nn_single_inference_reference \
bin/tests/automatic_launch_graphchallenge_nn_single_inference_reference_omp \
bin/tests/hpcg_reference \
bin/tests/hpcg_reference_omp \
bin/tests/matrixIterator_reference \
bin/tests/matrixIterator_reference_omp \
bin/tests/matrixSet_reference \
bin/tests/matrixSet_reference_omp \
bin/tests/eWiseMatrix_reference \
bin/tests/eWiseMatrix_reference_omp \
bin/tests/kmeans_unit_reference \
bin/tests/kmeans_unit_reference_omp \
bin/tests/spy_reference \
bin/tests/spy_reference_omp \
bin/tests/dense_spmv_debug_reference \
bin/tests/dense_spmv_debug_reference_omp \
bin/tests/dense_spmv_reference \
bin/tests/dense_spmv_reference_omp

KNNTESTS=bin/tests/automatic_launch_knn_serial \
bin/tests/automatic_launch_knn_openmp

PRTESTS=bin/tests/automatic_launch_pagerank_serial \
bin/tests/automatic_launch_pagerank_openmp \

LABELTESTS=bin/tests/automatic_launch_label_serial \
bin/tests/automatic_launch_label_openmp

LABELPERFTESTS=bin/tests/automatic_launch_label_serial_test \
bin/tests/automatic_launch_label_openmp_test

SCALETESTS=bin/tests/automatic_launch_scaling_serial \
bin/tests/automatic_launch_scaling_openmp

LIBRARIES=lib/sequential/libgraphblas.a lib/sequential/libgraphblas.so lib/sequential/libgraphblas.so.${VERSION}

CLEAN_OBJS+=${GRAPHBLAS_OBJS} ${GRAPHBLAS_SHARED_OBJS}

BSPCPP11=echo DISABLED:
BSPRUN=echo DISABLED:
MANUALRUN=echo DISABLED:

bin/tests/hpparser: src/graphblas/utils/hpparser.c include/graphblas/utils/hpparser.h | dirtree
	${C89} ${WFLAGS} ${IFLAGS} ${CFLAGS} -DTEST_HPPARSER -D_GNU_SOURCE -D_DEBUG $< -o $@

bin/tests/parserTest: tests/utilParserTest.cpp ${GRAPHBLAS_INCLUDES} | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/utilParserTest.cpp -o $@

bin/tests/compareParserTest: tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} -DCOMPARE $< -o $@ ${SEQ_LFLAGS}

bin/tests/add15d: tests/add15d.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/add15d.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/mul15i: tests/mul15i.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/mul15i.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/mul15m: tests/mul15m.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/mul15m.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/add15m: tests/add15m.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/add15m.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/distribution_bsp1d: tests/distribution_bsp1d.cpp  ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/distribution_bsp1d.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/fma: tests/fma.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels.o | dirtree
	${CPP11} ${PERFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} tests/fma.cpp -o $@ -lrt ${SEQ_LFLAGS} tests/bench_kernels.o

bin/tests/fma-openmp: tests/fma.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels_omp.o | dirtree
	${CPP11} ${PERFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} tests/fma.cpp -o $@ -lrt ${SEQ_LFLAGS} tests/bench_kernels_omp.o

bin/tests/reduce: tests/reduce.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels.o | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} tests/reduce.cpp -o $@ ${SEQ_LFLAGS} tests/bench_kernels.o

bin/tests/reduce-openmp: tests/reduce.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels_omp.o | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} tests/reduce.cpp -o $@ ${SEQ_LFLAGS} tests/bench_kernels_omp.o

bin/tests/dot: tests/dot.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels.o | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} tests/dot.cpp -o $@ ${SEQ_LFLAGS} tests/bench_kernels.o

bin/tests/dot-openmp: tests/dot.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a tests/bench_kernels_omp.o | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} tests/dot.cpp -o $@ ${SEQ_LFLAGS} tests/bench_kernels_omp.o

bin/tests/zip_reference: tests/zip.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/zip_reference_omp: tests/zip.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/dot_reference: tests/dot_unit.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/dot_reference_omp: tests/dot_unit.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxm_reference: tests/mxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxm_reference_omp: tests/mxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/ewiseapply_reference: tests/ewiseapply.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/ewiseapply_reference_omp: tests/ewiseapply.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/thread_local_storage: tests/thread_local_storage.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/thread_local_storage.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_knn_serial: tests/launcher/knn.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} tests/launcher/knn.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_knn_openmp: tests/launcher/knn.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} tests/launcher/knn.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_pagerank_serial: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} tests/launcher/simple_pagerank.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_pagerank_openmp: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} tests/launcher/simple_pagerank.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_label_serial: tests/launcher/label.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} ${TEST_UTILS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${PERFLAGS} tests/launcher/label.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_label_openmp: tests/launcher/label.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} ${TEST_UTILS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${OMPFLAGS} ${PERFLAGS} tests/launcher/label.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_label_serial_test: tests/launcher/label_test.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} tests/launcher/label_test.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_label_openmp_test: tests/launcher/label_test.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} tests/launcher/label_test.cpp tests/parser.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_scaling_serial: tests/launcher/scaling.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} tests/launcher/scaling.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_scaling_openmp: tests/launcher/scaling.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} tests/launcher/scaling.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/sparse_vxm_reference: tests/sparse_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/sparse_mxv_reference: tests/sparse_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_vxm_reference: tests/masked_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_mxv_reference: tests/masked_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/vmx_reference: tests/vmx.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/vmx.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/vmxa_reference: tests/vmxa.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/vmxa.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/emptyVector_reference: tests/emptyVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} tests/emptyVector.cpp -o $@ ${SEQ_LFLAGS}

bin/examples/sp_reference: examples/sp.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} examples/sp.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/vxm_reference: tests/launcher/vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxv_reference: tests/launcher/mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/sparse_vxm_reference_omp: tests/sparse_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/sparse_mxv_reference_omp: tests/sparse_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_vxm_reference_omp: tests/masked_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_mxv_reference_omp: tests/masked_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/vmx_reference_omp: tests/vmx.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} tests/vmx.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/vmxa_reference_omp: tests/vmxa.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} tests/vmxa.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/emptyVector_reference_omp: tests/emptyVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${OMPFLAGS} tests/emptyVector.cpp -o $@ ${SEQ_LFLAGS}

bin/examples/sp_reference_omp: examples/sp.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} examples/sp.cpp -o $@ ${SEQ_LFLAGS}

bin/tests/vxm_reference_omp: tests/launcher/vxm.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxv_reference_omp: tests/launcher/mxv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/swapVector_reference: tests/swapVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/swapVector_reference_omp: tests/swapVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/copyVector_reference: tests/copyVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/copyVector_reference_omp: tests/copyVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/moveVector_reference_omp: tests/moveVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/moveVector_reference: tests/moveVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/moveMatrix_reference_omp: tests/moveMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/moveMatrix_reference: tests/moveMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/stdVector_reference_omp: tests/stdVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/stdVector_reference: tests/stdVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/RBGaussSeidel_reference_omp: tests/RBGaussSeidel.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/RBGaussSeidel_reference: tests/RBGaussSeidel.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/muladd_reference_omp: tests/muladd.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/muladd_reference: tests/muladd.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_muladd_reference_omp: tests/masked_muladd.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/masked_muladd_reference: tests/masked_muladd.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/buildVector_reference: tests/buildVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/buildVector_reference_omp: tests/buildVector.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/copyAndAssignVectorIterator_reference: tests/copyAndAssignVectorIterator.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/copyAndAssignVectorIterator_reference_omp: tests/copyAndAssignVectorIterator.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/vectorToMatrix_reference: tests/vectorToMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/vectorToMatrix_reference_omp: tests/vectorToMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/set_reference: tests/set.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/set_reference_omp: tests/set.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/clearMatrix_reference: tests/clearMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/clearMatrix_reference_omp: tests/clearMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/argmin_reference: tests/argmin.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/argmin_reference_omp: tests/argmin.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/argmax_reference: tests/argmax.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/argmax_reference_omp: tests/argmax.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/matrixIterator_reference: tests/matrixIterator.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/matrixIterator_reference_omp: tests/matrixIterator.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/outer_product_reference: tests/outerProduct.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/outer_product_reference_omp: tests/outerProduct.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxm_elementwise_reference: tests/mxm_elementwise.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/mxm_elementwise_reference_omp: tests/mxm_elementwise.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/kmeans_unit_reference: tests/kmeans_unit.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/kmeans_unit_reference_omp: tests/kmeans_unit.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/hpcg_reference: tests/launcher/hpcg_test.cpp tests/launcher/hpcg_matrix_building_utils.hpp tests/launcher/hpcg_system_building_utils.hpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a ${TEST_UTILS_INCLUDES} ${LIB_TEST_UTILS} | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} $< ${LIB_TEST_UTILS} -o "$@" ${SEQ_LFLAGS}

bin/tests/hpcg_reference_omp: tests/launcher/hpcg_test.cpp tests/launcher/hpcg_matrix_building_utils.hpp tests/launcher/hpcg_system_building_utils.hpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a ${TEST_UTILS_INCLUDES} ${LIB_TEST_UTILS} | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} $< ${LIB_TEST_UTILS} -o "$@" ${SEQ_LFLAGS}

bin/tests/automatic_launch_conjugate_gradient_reference: tests/launcher/conjugate_gradient.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} $< -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_conjugate_gradient_reference_omp: tests/launcher/conjugate_gradient.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} $< -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_graphchallenge_nn_single_inference_reference: tests/launcher/graphchallenge_nn_single_inference.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} $< -o $@ ${SEQ_LFLAGS}

bin/tests/automatic_launch_graphchallenge_nn_single_inference_reference_omp: tests/launcher/graphchallenge_nn_single_inference.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${OMPFLAGS} ${PERFLAGS} $< -o $@ ${SEQ_LFLAGS}

bin/tests/matrixSet_reference: tests/matrixSet.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/matrixSet_reference_omp: tests/matrixSet.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/eWiseMatrix_reference: tests/eWiseMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/eWiseMatrix_reference_omp: tests/eWiseMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/spy_reference: tests/spy.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/spy_reference_omp: tests/spy.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} "$<" -o "$@" ${SEQ_LFLAGS}

bin/tests/dense_spmv_debug_reference: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/dense_spmv_debug_reference_omp: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/dense_spmv_reference: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${PERFLAGS} $< -o "$@" ${SEQ_LFLAGS}

bin/tests/dense_spmv_reference_omp: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/sequential/libgraphblas.a | dirtree
	${CPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${CFLAGS} ${OMPFLAGS} ${PERFLAGS} $< -o "$@" ${SEQ_LFLAGS}

