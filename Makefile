
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
.PHONY: install-dirs install all libs tests clean examples veryclean dirtree unittests smoketests perftests bin/tests/pagerank bin/tests/knn bin/tests/label bin/tests/scaling bin/tests/kernels docs

default: libs

SHELL=/bin/bash

#where to find the dependency sources
MY_DIR=$(CURDIR)/

#version info
MAJORVERSION=0
MINORVERSION=3
BUGVERSION=0
VERSION=${MAJORVERSION}.${MINORVERSION}.${BUGVERSION}

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
include/graphblas/algorithms/spec_part_utils.hpp \
include/graphblas/algorithms/cosine_similarity.hpp \
include/graphblas/algorithms/conjugate_gradient.hpp \
include/graphblas/algorithms/multigrid_v_cycle.hpp \
include/graphblas/algorithms/sparse_nn_single_inference.hpp \
include/graphblas/algorithms/gnn_single_inference.hpp \
include/graphblas/algorithms/multilevel_partition.hpp \
include/graphblas/algorithms/red_black_gauss_seidel.hpp \
include/graphblas/algorithms/spectral_graph_partition.hpp \
include/graphblas/algorithms/ROPTLIB/Grassmann_pLap.hpp \
include/graphblas/algorithms/pLaplacian_spectral_partition.hpp

#include environment-dependent info
include paths.mk

#check paths
ifndef GRB_INSTALL_PATH
$(error GRB_INSTALL_PATH was not defined)
endif
ifeq ($(GRB_INSTALL_PATH),$(shell pwd))
$(error GRB_INSTALL_PATH cannot be equal to the current directory)
endif

include banshee.mk

INSTALL_TARGETS=$(addprefix $(GRB_INSTALL_PATH)/,$(GRAPHBLAS_INCLUDES))
INSTALL_TARGETS+=$(addprefix $(GRB_INSTALL_PATH)/,$(LIBRARIES))

libs: ${LIBRARIES}

all: tests libs

paths.mk:
	@echo 'Error: please execute `./bootstrap.sh --prefix=/path/to/install directory'"'"' first or issue `./bootstrap.sh --help'"'"' for details.' && false

install: $(INSTALL_TARGETS) ${GRB_INSTALL_PATH}/bin/setenv ${GRB_INSTALL_PATH}/bin/grbcxx ${GRB_INSTALL_PATH}/bin/grbrun

install-dirs:
	#make sure GRB_INSTALL_PATH exists or bad things may happen
	mkdir "${GRB_INSTALL_PATH}" || true
	#make sure directory structure exists
	mkdir "${GRB_INSTALL_PATH}/bin" || true
	mkdir "${GRB_INSTALL_PATH}/lib" || true
	mkdir "${GRB_INSTALL_PATH}/include" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/base" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/reference" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/banshee" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/algorithms" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/algorithms/ROPTLIB" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/utils" || true
	mkdir "${GRB_INSTALL_PATH}/include/graphblas/utils/parser" || true

${GRB_INSTALL_PATH}/%: % | install-dirs
	cp "$<" "$@"

dirtree:
	mkdir lib || true
	mkdir bin || true
	mkdir bin/tests || true
	mkdir bin/tests/output || true

docs/code: ${GRAPHBLAS_INCLUDES} docs/doxy.conf
	doxygen docs/doxy.conf &> doxygen.log

docs: docs/code

${GRB_INSTALL_PATH}:
	mkdir "$@" || true

${GRB_INSTALL_PATH}/include: | ${GRB_INSTALL_PATH}
	mkdir "$@" || true

${GRB_INSTALL_PATH}/bin: | ${GRB_INSTALL_PATH}
	mkdir "$@" || true

${GRB_INSTALL_PATH}/bin/grbcxx: src/grbcxx.in paths.mk reference.mk banshee.mk bsp.mk bsp1d.mk | ${GRB_INSTALL_PATH}/bin
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

${GRB_INSTALL_PATH}/bin/grbrun: src/grbrun.in paths.mk reference.mk banshee.mk bsp.mk bsp1d.mk | ${GRB_INSTALL_PATH}/bin
	echo "#!/bin/bash" > "$@"
	echo "BACKENDS=(${BACKENDS})" >> "$@"
	echo "BACKENDRUNENV=(${BACKENDRUNENV})" >> "$@"
	echo "BACKENDRUNNER=(${BACKENDRUNNER})" >> "$@"
	cat "$<" >> "$@"
	chmod +x "$@"

${GRB_INSTALL_PATH}/bin/setenv: src/deps.env reference.mk bsp1d.mk banshee.mk ${GRB_INSTALL_PATH}/bin
	echo "DEPDIR=${GRB_INSTALL_PATH}" > ${GRB_INSTALL_PATH}/bin/setenv
	cat src/deps.env >> ${GRB_INSTALL_PATH}/bin/setenv
	chmod a+x ${GRB_INSTALL_PATH}/bin/setenv
ifdef MANUALRUN
	echo "export MANUALRUN=\"${MANUALRUN}\"" >> ${GRB_INSTALL_PATH}/bin/setenv
endif
	echo "export BACKENDS=\"${BACKENDS}\"" >> ${GRB_INSTALL_PATH}/bin/setenv


bin/tests/unit: ${UNITTESTS} | dirtree

clean:
	rm -r bin || true
	rm -r lib || true
	rm -r docs/code || true
	rm -f doxygen.log
	rm -f ${CLEAN_OBJS} || true

veryclean: clean
	rm -f paths.mk || true
