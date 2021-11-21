
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
.PHONY: install-dirs install all libs tests clean examples veryclean dirtree unittests smoketests perftests bin/tests/pagerank bin/tests/knn bin/tests/label bin/tests/scaling bin/tests/kernels

default: libs

SHELL=/bin/bash

#where to find the dependency sources
MY_DIR=$(CURDIR)/

#version info
MAJORVERSION=0
MINORVERSION=3
BUGVERSION=0
VERSION=${MAJORVERSION}.${MINORVERSION}.${BUGVERSION}

#main compilation flags
#EXTRA_CFLAGS allows injecting extra flags (like -Wfatal-errors or -O3 for tests)
CFLAGS=-fopenmp -g -fPIC ${EXTRA_CFLAGS}

#standard includes
GRAPHBLAS_INCLUDES_BASE=\
include/graphblas.hpp \
include/graphblas/backends.hpp \
include/graphblas/benchmark.hpp \
include/graphblas/blas0.hpp \
include/graphblas/blas1.hpp \
include/graphblas/blas2.hpp \
include/graphblas/blas3.hpp \
include/graphblas/collectives.hpp \
include/graphblas/config.hpp \
include/graphblas/coordinates.hpp \
include/graphblas/descriptors.hpp \
include/graphblas/distribution.hpp \
include/graphblas/exec.hpp \
include/graphblas/identities.hpp \
include/graphblas/init.hpp \
include/graphblas/internalops.hpp \
include/graphblas/io.hpp \
include/graphblas/iomode.hpp \
include/graphblas/matrix.hpp \
include/graphblas/monoid.hpp \
include/graphblas/ops.hpp \
include/graphblas/phase.hpp \
include/graphblas/pinnedvector.hpp \
include/graphblas/properties.hpp \
include/graphblas/rc.hpp \
include/graphblas/semiring.hpp \
include/graphblas/spmd.hpp \
include/graphblas/tags.hpp \
include/graphblas/type_traits.hpp \
include/graphblas/utils.hpp \
include/graphblas/vector.hpp \
include/graphblas/base/alloc.hpp \
include/graphblas/base/vector.hpp \
include/graphblas/base/internalops.hpp \
include/graphblas/base/io.hpp \
include/graphblas/base/coordinates.hpp \
include/graphblas/base/config.hpp \
include/graphblas/base/pinnedvector.hpp \
include/graphblas/base/distribution.hpp \
include/graphblas/base/matrix.hpp \
include/graphblas/base/blas2.hpp \
include/graphblas/base/spmd.hpp \
include/graphblas/base/properties.hpp \
include/graphblas/base/blas3.hpp \
include/graphblas/base/exec.hpp \
include/graphblas/base/init.hpp \
include/graphblas/base/collectives.hpp \
include/graphblas/base/benchmark.hpp \
include/graphblas/utils/alloc.hpp \
include/graphblas/utils/config.hpp \
include/graphblas/utils/hpparser.h \
include/graphblas/utils/parser.hpp \
include/graphblas/utils/ranges.hpp \
include/graphblas/utils/pattern.hpp \
include/graphblas/utils/autodeleter.hpp \
include/graphblas/utils/NonzeroIterator.hpp \
include/graphblas/utils/IndexedVectorMap.hpp \
include/graphblas/utils/MatrixVectorIterator.hpp \
include/graphblas/utils/SynchronizedNonzeroIterator.hpp \
include/graphblas/utils/Timer.hpp \
include/graphblas/utils/TimerResults.hpp \
include/graphblas/utils/ThreadLocalStorage.hpp \
include/graphblas/utils/parser/MatrixFileIterator.hpp \
include/graphblas/utils/parser/MatrixFileProperties.hpp \
include/graphblas/utils/parser/MatrixFileReader.hpp \
include/graphblas/utils/parser/MatrixFileReaderBase.hpp \
include/graphblas/utils/ndim_matrix_builders.hpp

GRAPHBLAS_INCLUDES_ALGOS=\
include/graphblas/algorithms/knn.hpp \
include/graphblas/algorithms/mpv.hpp \
include/graphblas/algorithms/spy.hpp \
include/graphblas/algorithms/hpcg.hpp \
include/graphblas/algorithms/label.hpp \
include/graphblas/algorithms/kmeans.hpp \
include/graphblas/algorithms/hpcg_data.hpp \
include/graphblas/algorithms/simple_pagerank.hpp \
include/graphblas/algorithms/cosine_similarity.hpp \
include/graphblas/algorithms/conjugate_gradient.hpp \
include/graphblas/algorithms/multigrid_v_cycle.hpp \
include/graphblas/algorithms/sparse_nn_single_inference.hpp \
include/graphblas/algorithms/red_black_gauss_seidel.hpp

SMOKETESTS=bin/tests/pagerank \
	   bin/tests/knn

TEST_UTILS_SOURCES=\
tests/utils/argument_parser.cpp \
tests/utils/assertion_engine.cpp

TEST_UTILS_OBJS1=${TEST_UTILS_SOURCES:%.cpp=%.o}
TEST_UTILS_OBJS=${TEST_UTILS_OBJS1:%.c=%.o}

TEST_UTILS_INCLUDES=\
tests/utils/token_handlers.hpp \
tests/utils/assertions.hpp \
tests/utils/parsed_types.hpp \
tests/utils/internal_argument_parser_defs.hpp \
tests/utils/internal_argument_parser.hpp \
tests/utils/argument_parser.hpp\
tests/utils/print_vec_mat.hpp \
tests/utils/assertion_engine.hpp

LIB_TEST_UTILS=lib/libtestutils.a


#include environment-dependent info
include flags.mk
include paths.mk

#check paths
ifndef GRB_INSTALL_PATH
$(error GRB_INSTALL_PATH was not defined)
endif
ifeq ($(GRB_INSTALL_PATH),$(shell pwd))
$(error GRB_INSTALL_PATH cannot be equal to the current directory)
endif

CPP11=${CXX} -std=c++11
C89=${CC} -std=c89
C99=${CC} -std=c99

CLEAN_OBJS=${TEST_UTILS_OBJS}

ifndef NO_REFERENCE
 include reference.mk
endif
ifndef NO_BANSHEE
 include banshee.mk
endif

ifndef NO_LPF
 include bsp.mk
 include bsp1d.mk

lib/spmd/libgraphblas.a: ${GRAPHBLAS_OBJS} ${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.o} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${AR} rcs $@ ${GRAPHBLAS_OBJS} ${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.o}

lib/hybrid/libgraphblas.a: ${GRAPHBLAS_OBJS} ${GRAPHBLAS_HYB_SOURCES:%.cpp=%.hyb.o} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${AR} rcs $@ ${GRAPHBLAS_OBJS} ${GRAPHBLAS_HYB_SOURCES:%.cpp=%.hyb.o}

lib/spmd/libgraphblas.so: lib/spmd/libgraphblas.so.${VERSION}
	ln -s libgraphblas.so.${VERSION} $@ || true

lib/spmd/libgraphblas.so.${VERSION}: ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.shared.o} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${CPP11} -shared ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_BSP_SOURCES:%.cpp=%.bsp.shared.o} -Wl,-soname,libgraphblas.so.${MAJORVERSION} -o $@

lib/hybrid/libgraphblas.so: lib/hybrid/libgraphblas.so.${VERSION}
	ln -s libgraphblas.so.${VERSION} $@ || true

lib/hybrid/libgraphblas.so.${VERSION}: ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_HYB_SOURCES:%.cpp=%.hyb.shared.o} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${CPP11} -shared ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_BSP_SOURCES:%.cpp=%.hyb.shared.o} -Wl,-soname,libgraphblas.so.${MAJORVERSION} -o $@
endif

# all GraphBLAS includes, especially for tests, which need them all
GRAPHBLAS_INCLUDES=${GRAPHBLAS_INCLUDES_BASE} ${GRAPHBLAS_INCLUDES_ALGOS}

lib/sequential/libgraphblas.a: ${GRAPHBLAS_OBJS} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${AR} rcs $@ ${GRAPHBLAS_OBJS}

lib/sequential/libgraphblas.so: lib/sequential/libgraphblas.so.${VERSION} | dirtree
	ln -s libgraphblas.so.${VERSION} $@ || true

lib/sequential/libgraphblas.so.${VERSION}: ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_SEQ_SOURCES:%.cpp=%.shared.o} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${CPP11} -shared ${GRAPHBLAS_SHARED_OBJS} ${GRAPHBLAS_SEQ_SOURCES:%.cpp=%.shared.o} -Wl,-soname,libgraphblas.so.${VERSION} -o $@

${LIB_TEST_UTILS}: ${TEST_UTILS_OBJS} ${TEST_UTILS_INCLUDES} ${GRAPHBLAS_INCLUDES_BASE} | dirtree
	${AR} rcs $@ ${TEST_UTILS_OBJS}

#various compiler flags
WFLAGS=-Wall -Wextra
IFLAGS=-Iinclude/
ITFLAGS=-Itests/
TESTFLAGS=-O2
PERFLAGS=-march=native -mtune=native -O3 -funroll-loops -DNDEBUG

#Environment info
NUMTHREADS:=`grep processor /proc/cpuinfo | tail -1 | cut -d':' -f2- | sed 's/\ //g'`

INSTALL_TARGETS=$(addprefix $(GRB_INSTALL_PATH)/,$(GRAPHBLAS_INCLUDES))
INSTALL_TARGETS+=$(addprefix $(GRB_INSTALL_PATH)/,$(LIBRARIES))

# intercept only backends compilations, which depend only on the base headers
src/graphblas/%.o: src/graphblas/%.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${CPP11} ${PERFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

#compiles objects that go into the static library. Since the number of symbols
#here fit on half a page, we simply use -fPIC instead of tightly controlling
#which symbols get exported. (Aug 2018)
%.shared.o: %.cpp ${GRAPHBLAS_INCLUDES_BASE}
	${CPP11} -fPIC ${PERFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

# intercept building for test utils
tests/utils/%.o: tests/utils/%.cpp ${TEST_UTILS_INCLUDES} ${GRAPHBLAS_INCLUDES}
	${CPP11} ${PERFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

# intercept all remaining compilations, which may depend on all headers
%.o: %.cpp ${GRAPHBLAS_INCLUDES}
	${CPP11} ${PERFLAGS} ${IFLAGS} ${CFLAGS} $< -c -o $@

libs: ${LIBRARIES}

all: examples tests libs

flags.mk:
	@echo 'Error: please execute `./configure --prefix=/path/to/install directory'"'"' first or issue `./configure --help'"'"' for details.' && false

paths.mk:
	@echo 'Error: please execute `./configure --prefix=/path/to/install directory'"'"' first or issue `./configure --help'"'"' for details.' && false

install: $(INSTALL_TARGETS) ${GRB_INSTALL_PATH}/bin/setenv ${GRB_INSTALL_PATH}/bin/grbcxx ${GRB_INSTALL_PATH}/bin/grbrun

install-dirs:
	#make sure GRB_INSTALL_PATH exists or bad things may happen
	mkdir "${GRB_INSTALL_PATH}" || true
	#make sure directory structure exists
	mkdir "${GRB_INSTALL_PATH}/bin" || true
	mkdir "${GRB_INSTALL_PATH}/lib" || true
	mkdir "${GRB_INSTALL_PATH}/lib/sequential" || true
	mkdir "${GRB_INSTALL_PATH}/lib/spmd" || true
	mkdir "${GRB_INSTALL_PATH}/lib/hybrid" || true
	mkdir "${GRB_INSTALL_PATH}/include" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/base" || true
ifndef NO_LPF
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/bsp" || true
endif
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/omp" || true
ifndef NO_LPF
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/bsp1d" || true
endif
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/utils" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/reference" || true
ifndef NO_BANSHEE
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/banshee" || true
endif
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/algorithms" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/utils/parser" || true

${GRB_INSTALL_PATH}/%: % | install-dirs
	cp "$<" "$@"

dirtree:
	mkdir lib || true
	mkdir lib/spmd || true
	mkdir lib/sequential || true
	mkdir lib/hybrid || true
	mkdir bin || true
	mkdir bin/tests || true
	mkdir bin/tests/output || true
	mkdir bin/examples || true

src/graphblas/utils/hpparser.o: src/graphblas/utils/hpparser.c include/graphblas/utils/hpparser.h
	${C89} ${WFLAGS} ${IFLAGS} ${CFLAGS} ${PERFLAGS} -D_GNU_SOURCE -c $< -o $@

src/graphblas/utils/hpparser.shared.o: src/graphblas/utils/hpparser.c include/graphblas/utils/hpparser.h
	${C89} ${WFLAGS} ${IFLAGS} -fPIC ${CFLAGS} ${PERFLAGS} -D_GNU_SOURCE -c $< -o $@

tests/bench_kernels.o: tests/bench_kernels.c tests/bench_kernels.h
	${C99} ${WFLAGS} ${ITFLAGS} ${CFLAGS} ${PERFLAGS} -c $< -o $@

tests/bench_kernels_omp.o: tests/bench_kernels.c tests/bench_kernels.h
	${C99} ${WFLAGS} ${ITFLAGS} ${CFLAGS} ${PERFLAGS} -DBENCH_KERNELS_OPENMP -c $< -o $@

docs: ${GRAPHBLAS_INCLUDES}
	doxygen doxy.conf &> doxygen.log

${GRB_INSTALL_PATH}:
	mkdir "$@" || true

${GRB_INSTALL_PATH}/include: | ${GRB_INSTALL_PATH}
	mkdir "$@" || true

${GRB_INSTALL_PATH}/bin: | ${GRB_INSTALL_PATH}
	mkdir "$@" || true

${GRB_INSTALL_PATH}/bin/grbcxx: src/grbcxx.in flags.mk paths.mk reference.mk banshee.mk bsp.mk bsp1d.mk | ${GRB_INSTALL_PATH}/bin
	echo "#!/bin/bash" > "$@"
	echo "GRB_INSTALL_PATH=${GRB_INSTALL_PATH}" >> "$@"
	echo "INCLUDEDIR=${GRB_INSTALL_PATH}/include/" >> "$@"
	echo "BACKENDS=(${BACKENDS})" >> "$@"
	echo "BACKENDCOMPILERS=(${BACKENDCOMPILER})" >> "$@"
	echo "BACKENDCFLAGS=(${BACKENDCFLAGS})" >> "$@"
	echo "BACKENDLFLAGS=(${BACKENDLFLAGS})" >> "$@"
	echo "COMMONCFLAGS=\"${COMMONCFLAGS}\"" >> "$@"
	echo "VERSION=\"${VERSION}\"" >> "$@"
	cat "$<" >> "$@"
	chmod +x "$@"

${GRB_INSTALL_PATH}/bin/grbrun: src/grbrun.in flags.mk paths.mk reference.mk banshee.mk bsp.mk bsp1d.mk | ${GRB_INSTALL_PATH}/bin
	echo "#!/bin/bash" > "$@"
	echo "BACKENDS=(${BACKENDS})" >> "$@"
	echo "BACKENDRUNENV=(${BACKENDRUNENV})" >> "$@"
	echo "BACKENDRUNNER=(${BACKENDRUNNER})" >> "$@"
	cat "$<" >> "$@"
	chmod +x "$@"

${GRB_INSTALL_PATH}/bin/setenv: flags.mk src/deps.env reference.mk bsp1d.mk banshee.mk ${GRB_INSTALL_PATH}/bin
	echo "DEPDIR=${GRB_INSTALL_PATH}" > ${GRB_INSTALL_PATH}/bin/setenv
	cat src/deps.env >> ${GRB_INSTALL_PATH}/bin/setenv
	chmod a+x ${GRB_INSTALL_PATH}/bin/setenv
ifdef LPFRUN
	echo "export LPFRUN=\"${LPFRUN}\"" >> ${GRB_INSTALL_PATH}/bin/setenv
endif
ifdef MANUALRUN
	echo "export MANUALRUN=\"${MANUALRUN}\"" >> ${GRB_INSTALL_PATH}/bin/setenv
endif
	echo "export BACKENDS=\"${BACKENDS}\"" >> ${GRB_INSTALL_PATH}/bin/setenv

examples: bin/examples/sp_reference | dirtree

tests: bin/tests/knn bin/tests/pagerank bin/tests/label bin/tests/scaling bin/tests/kernels | dirtree ${GRB_INSTALL_PATH}/bin/setenv
	$(MAKE) unittests
ifndef NO_LPF
	$(MAKE) smoketests
endif
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./scaling.sh 100)
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./benchmark.sh KERNEL)
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./benchmark.sh west0497.mtx)

unittests: bin/tests/unit | dirtree ${GRB_INSTALL_PATH}/bin/setenv
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./unittests.sh)

smoketests: ${SMOKETESTS} ${LABELTESTS} ${LABELPERFTESTS} | dirtree ${GRB_INSTALL_PATH}/bin/setenv
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./smoketests.sh)
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./label_test.sh)

perftests: bin/tests/knn bin/tests/pagerank bin/tests/label bin/tests/scaling bin/tests/kernels | dirtree ${GRB_INSTALL_PATH}/bin/setenv
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./scaling.sh)
	
	@echo "*****************************************************************************************"
	@echo "All scaling tests done; see bin/tests/output/scaling."
	@echo " "
	
	(. ${GRB_INSTALL_PATH}/bin/setenv && ./benchmark.sh)
	
	@echo "*****************************************************************************************"
	@echo "All benchmark tests done; see bin/tests/output/benchmarks."
	@echo " "

# -------------------------------------------------------------------
#  tests

bin/tests/unit: ${UNITTESTS} | dirtree

# -------------------------------------------------------------------
#  apps: knn

bin/tests/knn: ${KNNTESTS} | dirtree

# -------------------------------------------------------------------
#  apps: pagerank

bin/tests/pagerank: ${PRTESTS} | dirtree

# -------------------------------------------------------------------
#  apps: label propagation

bin/tests/label: ${LABELTESTS} | dirtree

# -------------------------------------------------------------------
# kernels: level-2 performance scaling

bin/tests/scaling: ${SCALETESTS} | dirtree

# -------------------------------------------------------------------
# kernels: level-1 performance tests

bin/tests/kernels: bin/tests/fma bin/tests/reduce bin/tests/dot bin/tests/dot-openmp bin/tests/reduce-openmp bin/tests/fma-openmp | dirtree

# -------------------------------------------------------------------
# cleaning

clean:
	rm -r bin || true
	rm -r lib || true
	rm -r docs || true
	rm -f tests/bench_kernels.o
	rm -f tests/bench_kernels_omp.o
	rm -f src/tools/txt2bin
	rm -f doxygen.log
	rm -f ${CLEAN_OBJS} || true
	rm -f ${LIB_TEST_UTILS} || true

veryclean: clean
	rm -f paths.mk flags.mk || true

