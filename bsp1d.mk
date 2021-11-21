
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

GRAPHBLAS_BSP_SOURCES+=\
src/graphblas/bsp1d/exec.cpp \
src/graphblas/bsp1d/init.cpp \
src/graphblas/bsp1d/config.cpp

GRAPHBLAS_HYB_SOURCES+=\
src/graphblas/bsp1d/exec.cpp \
src/graphblas/bsp1d/init.cpp \
src/graphblas/bsp1d/config.cpp

GRAPHBLAS_INCLUDES_BASE+=\
include/graphblas/bsp1d/io.hpp \
include/graphblas/bsp1d/exec.hpp \
include/graphblas/bsp1d/init.hpp \
include/graphblas/bsp1d/spmd.hpp \
include/graphblas/bsp1d/alloc.hpp \
include/graphblas/bsp1d/blas1.hpp \
include/graphblas/bsp1d/blas2.hpp \
include/graphblas/bsp1d/blas3.hpp \
include/graphblas/bsp1d/config.hpp \
include/graphblas/bsp1d/matrix.hpp \
include/graphblas/bsp1d/vector.hpp \
include/graphblas/bsp1d/benchmark.hpp \
include/graphblas/bsp1d/properties.hpp \
include/graphblas/bsp1d/distribution.hpp \
include/graphblas/bsp1d/pinnedvector.hpp

BACKENDS+=bsp1d
BACKENDS+=hybrid
BACKENDCOMPILER+=\"${LPFCPP}\" \"${LPFCPP}\"
BACKENDCFLAGS+=\"-D_GRB_WITH_LPF -D_GRB_BACKEND=BSP1D -D_GRB_COORDINATES_BACKEND=reference\" \"-D_GRB_WITH_LPF -D_GRB_BACKEND=BSP1D -D_GRB_BSP1D_BACKEND=reference_omp -D_GRB_COORDINATES_BACKEND=reference_omp\"
BACKENDLFLAGS+=\"${GRB_INSTALL_PATH}/lib/spmd/libgraphblas.a -llpf_hl -lpthread -lm -ldl -lnuma\" \"lib/hybrid/libgraphblas.a -llpf_hl -lpthread -lm -ldl -lnuma\"
BACKENDRUNENV+=\"\" \"\"
BACKENDRUNNER+=\"${LPFRUN}\" \"${LPFRUN}\"

LIBRARIES+=lib/spmd/libgraphblas.a lib/hybrid/libgraphblas.a lib/spmd/libgraphblas.so lib/hybrid/libgraphblas.so lib/spmd/libgraphblas.so.${VERSION} lib/hybrid/libgraphblas.so.${VERSION}

CLEAN_OBJS+=${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.o}
CLEAN_OBJS+=${GRAPHBLAS_HYB_SOURCES:%.cpp=%.hyb.o}
CLEAN_OBJS+=${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.shared.o}
CLEAN_OBJS+=${GRAPHBLAS_HYB_SOURCES:%.cpp=%.hyb.shared.o}

UNITTESTS+=bin/tests/fork_launcher \
bin/tests/manual_hook_hw \
bin/tests/manual_hook_grb_set \
bin/tests/manual_hook_grb_dot \
bin/tests/manual_hook_grb_reduce \
bin/tests/manual_hook_grb_collectives_blas0 \
bin/tests/manual_hook_grb_collectives_blas1 \
bin/tests/manual_hook_grb_collectives_blas1_raw \
bin/tests/automatic_hook_grb_collectives_blas0 \
bin/tests/automatic_launch_bsp1d_dot \
bin/tests/automatic_launch_sparse_vxm \
bin/tests/automatic_launch_vxm \
bin/tests/automatic_launch_mxv \
bin/tests/distribution \
bin/tests/ewiseapply_bsp1d \
bin/tests/ewiseapply_hybrid \
bin/tests/zip_bsp1d \
bin/tests/zip_hybrid \
bin/tests/dot_bsp1d \
bin/tests/dot_hybrid \
bin/tests/copyVector_bsp1d \
bin/tests/copyVector_hybrid \
bin/tests/swapVector_bsp1d \
bin/tests/swapVector_hybrid \
bin/tests/moveVector_bsp1d \
bin/tests/moveVector_hybrid \
bin/tests/moveMatrix_bsp1d \
bin/tests/moveMatrix_hybrid \
bin/tests/stdVector_bsp1d \
bin/tests/stdVector_hybrid \
bin/tests/RBGaussSeidel_bsp1d \
bin/tests/RBGaussSeidel_hybrid \
bin/tests/muladd_bsp1d \
bin/tests/muladd_hybrid \
bin/tests/masked_muladd_bsp1d \
bin/tests/masked_muladd_hybrid \
bin/tests/buildVector_bsp1d \
bin/tests/buildVector_hybrid \
bin/tests/copyAndAssignVectorIterator_bsp1d \
bin/tests/copyAndAssignVectorIterator_hybrid \
bin/tests/set_bsp1d \
bin/tests/set_hybrid \
bin/tests/vectorToMatrix_bsp1d \
bin/tests/vectorToMatrix_hybrid \
bin/tests/clearMatrix_bsp1d \
bin/tests/clearMatrix_hybrid \
bin/tests/argmin_bsp1d \
bin/tests/argmin_hybrid \
bin/tests/argmax_bsp1d \
bin/tests/argmax_hybrid \
bin/tests/matrixIterator_bsp1d \
bin/tests/matrixIterator_hybrid \
bin/tests/hpcg_bsp1d \
bin/tests/hpcg_hybrid \
bin/tests/matrixSet_bsp1d \
bin/tests/matrixSet_hybrid \
bin/tests/eWiseMatrix_bsp1d \
bin/tests/eWiseMatrix_hybrid \
bin/tests/automatic_launch_graphchallenge_nn_single_inference_bsp1d \
bin/tests/automatic_launch_graphchallenge_nn_single_inference_hybrid \
bin/tests/automatic_launch_conjugate_gradient_bsp1d \
bin/tests/automatic_launch_conjugate_gradient_hybrid \
bin/tests/dense_spmv_debug_bsp1d \
bin/tests/dense_spmv_debug_hybrid \
bin/tests/dense_spmv_bsp1d \
bin/tests/dense_spmv_hybrid
#TODO internal issue #9:
#bin/tests/vmxa_bsp1d \
#bin/tests/vmx_bsp1d \
#bin/tests/emptyVector_bsp1d \
#bin/examples/sp_bsp1d \
#bin/tests/sparse_vxm_bsp1d \
#bin/tests/sparse_mxv_bsp1d \
#bin/tests/masked_vxm_bsp1d \
#bin/tests/masked_mxv_bsp1d \
#bin/tests/vxm_bsp1d \
#bin/tests/mxv_bsp1d

KNNTESTS+=bin/tests/automatic_launch_knn_debug \
bin/tests/automatic_launch_knn \
bin/tests/automatic_launch_knn_hybrid \
bin/tests/automatic_hook_knn \
bin/tests/manual_hook_knn

PRTESTS+=bin/tests/automatic_hook_pagerank \
bin/tests/automatic_hook_pagerank_big \
bin/tests/automatic_hook_simple_pagerank \
bin/tests/automatic_launch_simple_pagerank \
bin/tests/automatic_launch_pagerank_debug \
bin/tests/automatic_launch_pagerank \
bin/tests/automatic_launch_pagerank_hybrid \
bin/tests/from_mpi_launch_simple_pagerank \
bin/tests/from_mpi_launch_simple_pagerank_multiple_entry \
bin/tests/from_mpi_launch_simple_pagerank_broadcast \
bin/tests/from_mpi_launch_simple_pagerank_broadcast_multiple_entry \
bin/tests/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry \
bin/tests/hpcg_hybrid

LABELTESTS+=bin/tests/automatic_launch_label \
bin/tests/automatic_launch_label_hybrid

LABELPERFTESTS+=bin/tests/automatic_launch_label_test \
bin/tests/automatic_launch_label_hybrid_test

SCALETESTS+=bin/tests/automatic_launch_scaling \
bin/tests/automatic_launch_scaling_hybrid

SMOKETESTS+=bin/tests/hpcg_bsp1d

bin/tests/fork_launcher: tests/fork_launcher.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} tests/fork_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_hw: tests/manual_launcher.cpp tests/hook/hello_world.cpp lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} tests/hook/hello_world.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_set: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/setvector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/setvector.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_dot: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/dot.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -Wno-maybe-uninitialized tests/hook/dot.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_reduce: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/reduce.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/reduce.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_collectives_blas0: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/collectives_blas0.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/collectives_blas0.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_collectives_blas1: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/collectives_blas1.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/collectives_blas1.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_grb_collectives_blas1_raw: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/collectives_blas1_raw.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/collectives_blas1_raw.cpp tests/hook/grb_launcher.cpp tests/manual_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_hook_grb_collectives_blas0: tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/collectives_blas0.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/hook/collectives_blas0.cpp tests/hook/grb_launcher.cpp tests/auto_launcher.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_bsp1d_dot: tests/launcher/dot.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -Wno-maybe-uninitialized $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_vxm: tests/launcher/vxm.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_mxv: tests/launcher/mxv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_knn_debug: tests/launcher/knn.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -DPRINT_FIRST_TEN tests/launcher/knn.cpp tests/parser.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_knn: tests/launcher/knn.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} tests/launcher/knn.cpp tests/parser.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_knn_hybrid: tests/launcher/knn.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/knn.cpp tests/parser.cpp -o $@ ${HYB_LFLAGS}

bin/tests/automatic_hook_knn: tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp -o $@ ${PAR_LFLAGS}

bin/tests/manual_hook_knn: tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/manual_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/knn.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_hook_pagerank: tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp  ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_hook_pagerank_big: tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp  ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${PERFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -DPR_TEST_DIMENSION=1000000 tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_hook_simple_pagerank: tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp  ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${TESTFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -DSIMPLE_PR_TEST tests/auto_launcher.cpp tests/hook/grb_launcher.cpp tests/hook/pagerank.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_simple_pagerank: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_pagerank_debug: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -DPRINT_FIRST_TEN tests/launcher/simple_pagerank.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_pagerank: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} tests/launcher/simple_pagerank.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_pagerank_hybrid: tests/launcher/simple_pagerank.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/simple_pagerank.cpp -o $@ ${HYB_LFLAGS}

bin/tests/from_mpi_launch_simple_pagerank: tests/launcher/simple_pagerank_from_mpi.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${TESTFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/from_mpi_launch_simple_pagerank_multiple_entry: tests/launcher/simple_pagerank_from_mpi.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${TESTFLAGS} -DMULTIPLE_ENTRY ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/from_mpi_launch_simple_pagerank_broadcast: tests/launcher/simple_pagerank_broadcast.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${TESTFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/from_mpi_launch_simple_pagerank_broadcast_multiple_entry: tests/launcher/simple_pagerank_broadcast.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${TESTFLAGS} -DMULTIPLE_ENTRY ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry: tests/launcher/simple_pagerank_broadcast.cpp ${GRAPHBLAS_INCLDES} lib/spmd/libgraphblas.a | dirtree
	${MPICPP11} ${CFLAGS} ${TESTFLAGS} -DMULTIPLE_ENTRY -DPINNED_OUTPUT ${WFLAGS} ${IFLAGS} ${PARFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_label: tests/launcher/label.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} ${TEST_UTILS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${PARFLAGS} ${PERFLAGS} tests/launcher/label.cpp tests/parser.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_label_hybrid: tests/launcher/label.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} ${TEST_UTILS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/label.cpp tests/parser.cpp -o $@ ${HYB_LFLAGS}

bin/tests/automatic_launch_label_test: tests/launcher/label_test.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} tests/launcher/label_test.cpp tests/parser.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_label_hybrid_test: tests/launcher/label_test.cpp tests/parser.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/label_test.cpp tests/parser.cpp -o $@ ${HYB_LFLAGS}

bin/tests/automatic_launch_scaling: tests/launcher/scaling.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} tests/launcher/scaling.cpp -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_scaling_hybrid: tests/launcher/scaling.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/scaling.cpp -o $@ ${HYB_LFLAGS}

bin/tests/automatic_launch_sparse_vxm: tests/launcher/sparse_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} tests/launcher/sparse_vxm.cpp -o $@ ${HYB_LFLAGS}

bin/tests/distribution: tests/distribution.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/sparse_vxm_bsp1d: tests/sparse_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/sparse_mxv_bsp1d: tests/sparse_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/masked_vxm_bsp1d: tests/masked_vxm.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/masked_mxv_bsp1d: tests/masked_mxv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/vmx_bsp1d: tests/vmx.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} tests/vmx.cpp -o $@ ${PAR_LFLAGS}

bin/tests/vmxa_bsp1d: tests/vmxa.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} tests/vmxa.cpp -o $@ ${PAR_LFLAGS}

bin/tests/emptyVector_bsp1d: tests/emptyVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PARFLAGS} tests/emptyVector.cpp -o $@ ${PAR_LFLAGS}

bin/examples/sp_bsp1d: examples/sp.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} examples/sp.cpp -o $@ ${PAR_LFLAGS}

bin/tests/vxm_bsp1d: tests/launcher/vxm.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PERFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/mxv_bsp1d: tests/launcher/mxv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/ewiseapply_bsp1d: tests/ewiseapply.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/ewiseapply_hybrid: tests/ewiseapply.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/zip_bsp1d: tests/zip.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/zip_hybrid: tests/zip.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/dot_bsp1d: tests/dot_unit.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/dot_hybrid: tests/dot_unit.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} -Wno-maybe-uninitialized "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/copyVector_bsp1d: tests/copyVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/copyVector_hybrid: tests/copyVector.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/swapVector_bsp1d: tests/swapVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/swapVector_hybrid: tests/swapVector.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/moveVector_bsp1d: tests/moveVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/moveVector_hybrid: tests/moveVector.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/moveMatrix_bsp1d: tests/moveMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/moveMatrix_hybrid: tests/moveMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/stdVector_bsp1d: tests/stdVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/stdVector_hybrid: tests/stdVector.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/RBGaussSeidel_bsp1d: tests/RBGaussSeidel.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/RBGaussSeidel_hybrid: tests/RBGaussSeidel.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/muladd_bsp1d: tests/muladd.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/muladd_hybrid: tests/muladd.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/masked_muladd_bsp1d: tests/masked_muladd.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/masked_muladd_hybrid: tests/masked_muladd.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/buildVector_bsp1d: tests/buildVector.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/buildVector_hybrid: tests/buildVector.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/copyAndAssignVectorIterator_bsp1d: tests/copyAndAssignVectorIterator.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/copyAndAssignVectorIterator_hybrid: tests/copyAndAssignVectorIterator.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/set_bsp1d: tests/set.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/set_hybrid: tests/set.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/vectorToMatrix_bsp1d: tests/vectorToMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/vectorToMatrix_hybrid: tests/vectorToMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/clearMatrix_bsp1d: tests/clearMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/clearMatrix_hybrid: tests/clearMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/argmin_bsp1d: tests/argmin.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/argmin_hybrid: tests/argmin.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/argmax_bsp1d: tests/argmax.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/argmax_hybrid: tests/argmax.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/hpcg_bsp1d: tests/launcher/hpcg_test.cpp tests/launcher/hpcg_matrix_building_utils.hpp tests/launcher/hpcg_system_building_utils.hpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a ${LIB_TEST_UTILS} | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${PARFLAGS} $< ${LIB_TEST_UTILS} -o "$@" ${PAR_LFLAGS}

bin/tests/hpcg_hybrid: tests/launcher/hpcg_test.cpp tests/launcher/hpcg_matrix_building_utils.hpp tests/launcher/hpcg_system_building_utils.hpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a ${LIB_TEST_UTILS} | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${ITFLAGS} ${HYBRIDFLAGS} $< ${LIB_TEST_UTILS} -o "$@" ${HYB_LFLAGS}

bin/tests/automatic_launch_conjugate_gradient_bsp1d: tests/launcher/conjugate_gradient.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_conjugate_gradient_hybrid: tests/launcher/conjugate_gradient.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} $< -o $@ ${HYB_LFLAGS}

bin/tests/automatic_launch_graphchallenge_nn_single_inference_bsp1d: tests/launcher/graphchallenge_nn_single_inference.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} $< -o $@ ${PAR_LFLAGS}

bin/tests/automatic_launch_graphchallenge_nn_single_inference_hybrid: tests/launcher/graphchallenge_nn_single_inference.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} $< -o $@ ${HYB_LFLAGS}

bin/tests/matrixIterator_bsp1d: tests/matrixIterator.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/matrixIterator_hybrid: tests/matrixIterator.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${PERFLAGS} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/matrixSet_bsp1d: tests/matrixSet.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/matrixSet_hybrid: tests/matrixSet.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/eWiseMatrix_bsp1d: tests/eWiseMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/eWiseMatrix_hybrid: tests/eWiseMatrix.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/dense_spmv_debug_bsp1d: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/dense_spmv_debug_hybrid: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

bin/tests/dense_spmv_bsp1d: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/spmd/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${PARFLAGS} ${PERFLAGS} "$<" -o "$@" ${PAR_LFLAGS}

bin/tests/dense_spmv_hybrid: tests/dense_spmv.cpp ${GRAPHBLAS_INCLUDES} lib/hybrid/libgraphblas.a | dirtree
	${LPFCPP11} ${CFLAGS} ${WFLAGS} ${IFLAGS} ${HYBRIDFLAGS} ${PERFLAGS} "$<" -o "$@" ${HYB_LFLAGS}

