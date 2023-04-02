#!/bin/bash

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
#set -e

TESTS_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ &> /dev/null && pwd )"
source ${TESTS_ROOT_DIR}/parse_env.sh

for MODE in debug ndebug; do

	echo "****************************************************************************************"
	echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
	echo "----------------------------------------------------------------------------------------"
	echo " "

	echo ">>>      [x]           [ ]       Testing grb::utils::equals over floats and doubles"
	${TEST_BIN_DIR}/equals_${MODE} &> ${TEST_OUT_DIR}/equals_${MODE}.log
	head -1 ${TEST_OUT_DIR}/equals_${MODE}.log
	grep 'Test OK' ${TEST_OUT_DIR}/equals_${MODE}.log || echo "Test FAILED"
	echo " "

	echo ">>>      [x]           [ ]       Testing numerical addition operator over doubles"
	${TEST_BIN_DIR}/add15d_${MODE}

	echo ">>>      [x]           [ ]       Testing numerical addition operator over a mixed field"
	echo "                                 (double, integers, and floats)"
	${TEST_BIN_DIR}/add15m_${MODE}

	echo ">>>      [x]           [ ]       Testing numerical multiplication operator over integers"
	${TEST_BIN_DIR}/mul15i_${MODE}

	echo ">>>      [x]           [ ]       Testing numerical multiplication operator over a mixed"
	echo "                                 field (double, integers, and floats)"
	${TEST_BIN_DIR}/mul15m_${MODE}

	echo ">>>      [x]           [ ]       Tests the built-in parser on the west0497 MatrixMarket file"
	if [ -f ${INPUT_DIR}/west0497.mtx ]; then
		${TEST_BIN_DIR}/parserTest_${MODE} 2> ${TEST_OUT_DIR}/parserTest_${MODE}.err 1> ${TEST_OUT_DIR}/parserTest_${MODE}.out
		head -1 ${TEST_OUT_DIR}/parserTest_${MODE}.out
		grep 'Test OK' ${TEST_OUT_DIR}/parserTest_${MODE}.out || echo "Test FAILED"
	else
		echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
	fi
	echo " "

	echo ">>>      [x]           [ ]       Tests the built-in parser (in graphblas/utils/parser.hpp)"
	echo "                                 versus the parser in tests/parser.cpp on cit-HepTh.txt."
	if [ -f ${INPUT_DIR}/cit-HepTh.txt ]; then
		${TEST_BIN_DIR}/compareParserTest_${MODE} &> ${TEST_OUT_DIR}/compareParserTest_${MODE}
		head -1 ${TEST_OUT_DIR}/compareParserTest_${MODE}
		tail -2 ${TEST_OUT_DIR}/compareParserTest_${MODE}
	else
		echo "Test DISABLED: cit-HepTh was not found. To enable, please provide the dataset within ${INPUT_DIR}/cit-HepTh.txt"
		echo " "
	fi

	echo ">>>      [x]           [ ]       Tests the built-in high-performance parser (in"
	echo "                                 include/graphblas/utils/parser.h &"
	echo "                                  src/graphblas/utils/parser.c) on dwt_59.mtx"
	echo "                                 Parameters: P=1, no hyperthreads (half the available threads),"
	echo "                                 block size = 128k, buffer size = 8M"
	if [ -f ${INPUT_DIR}/dwt_59.mtx ]; then
		echo "Functional test executable: ${TEST_BIN_DIR}/hpparser_${MODE}"
		${TEST_BIN_DIR}/hpparser_${MODE} 1 ${MAX_THREADS} 131072 8388608 ${INPUT_DIR}/dwt_59.mtx 1 &> ${TEST_OUT_DIR}/hpparser_${MODE}
		echo "[ 0, *] nrow =           59, ncol =           59, nnnz =          163
[ 0, *] offb =          564, fsiz =         1494, offe =         1493
[ *, *] ntot =          163" > ${TEST_OUT_DIR}/hpparser.chk
		(diff ${TEST_OUT_DIR}/hpparser_${MODE} ${TEST_OUT_DIR}/hpparser.chk && printf "Test OK.\n\n") || printf "Test FAILED.\n\n"
	else
		echo "Test DISABLED: dwt_59.mtx was not found. To enable, please provide ${INPUT_DIR}/dwt_59.mtx"
		echo " "
	fi

	for BACKEND in ${BACKENDS[@]}; do
		if [ "$BACKEND" = "bsp1d" ]; then
			Ps=( 1 2 16 )
		fi
		if [ "$BACKEND" = "hybrid" ]; then
			Ps=( 2 7 )
		fi
		if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
			if [ -z "${LPFRUN}" ]; then
				echo "LPFRUN is not set!"
				exit 255;
			fi
			if [ -z "${MANUALRUN}" ]; then
				echo "MANUALRUN is not set!"
				exit 255;
			fi
		else
			Ps=( 1 )
		fi
		if [ "$BACKEND" = "reference_omp" ]; then
			Pt=( 1 2 ${MAX_THREADS} )
		elif [ "$BACKEND" = "nonblocking" ]; then
			Pt=( 1 2 ${MAX_THREADS} )
		elif [ "$BACKEND" = "hybrid" ]; then
			MTDS=$((MAX_THREADS/7))
			if [ "$MTDS" -le "2" ]; then
				Pt=( 2 )
			else
				Pt=( 2 $((MAX_THREADS/7)) )
			fi
		else
			Pt=( 1 )
		fi
		for P in ${Ps[@]}; do
			for T in ${Pt[@]}; do

				runner=
				if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
					runner="${LPFRUN} -n ${P}"
				fi
				if [ "${BACKEND}" = "bsp1d" ]; then
					runner="${runner} ${BIND_PROCESSES_TO_HW_THREADS}"
				elif [ "${BACKEND}" = "hybrid" ]; then
					runner="${runner} ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}OMP_NUM_THREADS=${T}"
					runner="${runner} ${BIND_PROCESSES_TO_MULTIPLE_HW_THREADS}${T}"
				elif [ "$BACKEND" = "reference_omp" ]; then
					export OMP_NUM_THREADS=${T}
				fi

				if [ "$BACKEND" = "reference" ] || [ "${BACKEND}" = "reference_omp" ]; then
					echo "#################################################################"
					echo "# Starting unit tests specific to the ${BACKEND} backend"
					echo "#   using ${MODE} mode"
					echo "#   using ${T} threads"
					echo "#################################################################"
					echo " "

					if [ "x${runner}" != "x" ]; then
						echo "#   using runner \`\`$runner''"
					fi

					echo ">>>      [x]           [ ]       Testing grb::eWiseApply on tiny matrices whilst"
					echo "                                 checking the resulting internal data structures"
					$runner ${TEST_BIN_DIR}/eWiseApplyMatrixReference_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/eWiseApplyMatrixReference_${MODE}_${BACKEND}_${T}.log
					head -1 ${TEST_OUT_DIR}/eWiseApplyMatrixReference_${MODE}_${BACKEND}_${T}.log
					grep 'Test OK' ${TEST_OUT_DIR}/eWiseApplyMatrixReference_${MODE}_${BACKEND}_${T}.log || echo "Test FAILED"
					echo " "
				fi

				echo "#################################################################"
				echo "# Starting standardised unit tests for the ${BACKEND} backend"
				echo "#   using ${MODE} mode"
				echo "#   using ${P} user processes"
				echo "#   using ${T} threads"
				if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
					echo "#   using \`\`${LPFRUN}'' for automatic launchers"
				fi
				if [ "x${runner}" != "x" ]; then
					echo "#   using runner \`\`$runner''"
				fi
				echo "#################################################################"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::id on vectors and matrices"
				$runner ${TEST_BIN_DIR}/id_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/id_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/id_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/id_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::capacity, grb::resize, and default"
				echo "                                 and explicit capacities set during container"
				echo "                                 construction"
				$runner ${TEST_BIN_DIR}/capacity_${MODE}_${BACKEND} 5230 &> ${TEST_OUT_DIR}/capacity_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/capacity_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/capacity_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::set on vectors of doubles of size"
				echo "                                 1 000 000."
				$runner ${TEST_BIN_DIR}/set_${MODE}_${BACKEND} 1000000 &> ${TEST_OUT_DIR}/set_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/set_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/set_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing the grb::pinnedVector on fundamental and"
				echo "                                 non-fundamental value types."
				$runner ${TEST_BIN_DIR}/pinnedVector_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/pinnedVector_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/pinnedVector_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/pinnedVector_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseApply using (+,0) on vectors"
				echo "                                 of doubles of size 14."
				$runner ${TEST_BIN_DIR}/ewiseapply_${MODE}_${BACKEND} 14 &> ${TEST_OUT_DIR}/ewiseapply_small_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/ewiseapply_small_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/ewiseapply_small_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseApply using (+,0) on vectors"
				echo "                                 of doubles of size 100."
				$runner ${TEST_BIN_DIR}/ewiseapply_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/ewiseapply_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/ewiseapply_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/ewiseapply_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseApply using (+,0) on vectors"
				echo "                                 of doubles of size 10 000 000."
				$runner ${TEST_BIN_DIR}/ewiseapply_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/ewiseapply_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/ewiseapply_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/ewiseapply_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [x]       Testing grb::foldl and grb::foldr reducing dense"
				echo "                                 vectors into scalars using operators and monoids."
				$runner ${TEST_BIN_DIR}/fold_to_scalar_${MODE}_${BACKEND} ${P} &> ${TEST_OUT_DIR}/fold_to_scalar_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/fold_to_scalar_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/fold_to_scalar_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::dot on two vectors of doubles and"
				echo "                                 ints of size 1874."
				$runner ${TEST_BIN_DIR}/dot_${MODE}_${BACKEND} 1874 &> ${TEST_OUT_DIR}/dot_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/dot_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/dot_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::dot on two vectors of doubles and"
				echo "                                 ints of size 10 000 000."
				$runner ${TEST_BIN_DIR}/dot_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/dot_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/dot_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/dot_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing std::swap on two vectors of doubles of"
				echo "                                 size 100."
				$runner ${TEST_BIN_DIR}/swapVector_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/swapVector_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/swapVector_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/swapVector_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing std::move on two vectors of doubles of"
				echo "                                 size 100."
				$runner ${TEST_BIN_DIR}/moveVector_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/moveVector_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/moveVector_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/moveVector_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing std::move on two vectors of doubles of"
				echo "                                 size 100."
				$runner ${TEST_BIN_DIR}/moveMatrix_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/moveMatrix_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/moveMatrix_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/moveMatrix_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing std::vector of thirteen GraphBLAS"
				echo "                                 vectors of unsigned chars of sizes 100 and 50."
				$runner ${TEST_BIN_DIR}/stdVector_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/stdVector_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/stdVector_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/stdVector_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing std::vector of thirteen GraphBLAS"
				echo "                                 matrices of unsigned chars of various sizes."
				$runner ${TEST_BIN_DIR}/stdMatrix_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/stdMatrix_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/stdMatrix_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/stdMatrix_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::Vector's copy-constructor on a"
				echo "                                 vector of doubles of size 100."
				$runner ${TEST_BIN_DIR}/copyVector_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/copyVector_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/copyVector_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/copyVector_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::Vector's copy-constructor on a"
				echo "                                 vector of doubles of size 10 000 000."
				$runner ${TEST_BIN_DIR}/copyVector_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/copyVector_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/copyVector_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/copyVector_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::Matrix's output-iterators on"
				echo "                                 square matrices of size 15 and 1 000 000."
				$runner ${TEST_BIN_DIR}/matrixIterator_${MODE}_${BACKEND} 1000000 2> ${TEST_OUT_DIR}/matrixIterator_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/matrixIterator_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/matrixIterator_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/matrixIterator_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing double-assignment of ALP/GraphBLAS containers, i.e.,"
				echo "                                 assigning one container another one (a=b), twice in a row."
				$runner ${TEST_BIN_DIR}/doubleAssign_${MODE}_${BACKEND} 1337 &> ${TEST_OUT_DIR}/doubleAssign_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/doubleAssign_${MODE}_${BACKEND}_${P}_${T}.log
				grep -i 'test ok' ${TEST_OUT_DIR}/doubleAssign_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing copy and move constructors and assignment"
				echo "                                 of the const_iterator of grb::Vector< double > of"
				echo "                                 length 10 000 000."
				$runner ${TEST_BIN_DIR}/copyAndAssignVectorIterator_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/copyAndAssignVectorIterator_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/copyAndAssignVectorIterator_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/copyAndAssignVectorIterator_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseMul on a vector of"
				echo "                                 doubles of size 100."
				$runner ${TEST_BIN_DIR}/eWiseMul_${MODE}_${BACKEND} 100 &> ${TEST_OUT_DIR}/eWiseMul_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/eWiseMul_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/eWiseMul_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseMul on a vector of"
				echo "                                 doubles of size 100002."
				$runner ${TEST_BIN_DIR}/eWiseMul_${MODE}_${BACKEND} 100002 &> ${TEST_OUT_DIR}/eWiseMul_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/eWiseMul_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/eWiseMul_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseMulAdd on a vector of"
				echo "                                 doubles of size 7 000 000."
				$runner ${TEST_BIN_DIR}/masked_muladd_${MODE}_${BACKEND} 7000000 &> ${TEST_OUT_DIR}/masked_muladd_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/masked_muladd_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/masked_muladd_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseMulAdd on a vector of"
				echo "                                 doubles of size 10 000 000."
				$runner ${TEST_BIN_DIR}/muladd_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/muladd_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/muladd_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/muladd_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "
				echo ">>>      [x]           [ ]       Testing grb::buildVector and"
				echo "                                 grb::buildVectorUnique"
				$runner ${TEST_BIN_DIR}/buildVector_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/buildVector_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/buildVector_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/buildVector_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::vectorToMatrixConverter"
				$runner ${TEST_BIN_DIR}/vectorToMatrix_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/vectorToMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/vectorToMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/vectorToMatrix_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::clear on a 1M by 1M matrix of"
				echo "                                 doubles"
				$runner ${TEST_BIN_DIR}/clearMatrix_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/clearMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/clearMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/clearMatrix_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing double-masked grb::vxm and grb::mxv"
				echo "                                 on a mock red-black Gauss Seidel on west0497."
				if [ -f ${INPUT_DIR}/west0497.mtx ]; then
					$runner ${TEST_BIN_DIR}/RBGaussSeidel_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/RBGaussSeidel_${MODE}_${BACKEND}_${P}_${T}
					head -1 ${TEST_OUT_DIR}/RBGaussSeidel_${MODE}_${BACKEND}_${P}_${T}
					grep 'Test OK' ${TEST_OUT_DIR}/RBGaussSeidel_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				else
					echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
				fi
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::argmin"
				$runner ${TEST_BIN_DIR}/argmin_${MODE}_${BACKEND} 2> ${TEST_OUT_DIR}/argmin_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/argmin_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/argmin_${MODE}_${BACKEND}_${P}_${T}.log
				grep "Test OK" ${TEST_OUT_DIR}/argmin_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::argmax"
				$runner ${TEST_BIN_DIR}/argmax_${MODE}_${BACKEND} 2> ${TEST_OUT_DIR}/argmax_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/argmax_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/argmax_${MODE}_${BACKEND}_${P}_${T}.log
				grep "Test OK" ${TEST_OUT_DIR}/argmax_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::set (matrices)"
				$runner ${TEST_BIN_DIR}/matrixSet_${MODE}_${BACKEND} 2> ${TEST_OUT_DIR}/matrixSet_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/matrixSet_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/matrixSet_${MODE}_${BACKEND}_${P}_${T}.log
				echo "Test OK" ${TEST_OUT_DIR}/matrixSet_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Tests the \`level-0' grb::collectives"
				echo "Functional test executable: ${TEST_BIN_DIR}/collectives_blas0_${MODE}_${BACKEND}"
				$runner ${TEST_BIN_DIR}/collectives_blas0_${MODE}_${BACKEND} ${P} &> ${TEST_OUT_DIR}/collectives_blas0_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/collectives_blas0_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				if [ -f ${INPUT_DIR}/west0497.mtx ]; then
					echo ">>>      [x]           [ ]       Testing the spmv (y=Ax) on west0497, using dense vectors"
					$runner ${TEST_BIN_DIR}/dense_spmv_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 1 1 1 &> ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_Ax_${P}_${T}.log
					head -1 ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_Ax_${P}_${T}.log
					if grep -q 'Test OK' ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_Ax_${P}_${T}.log; then
						echo 'Test OK'
					else
						echo 'Test FAILED'
					fi
					echo " "

					echo ">>>      [x]           [ ]       Testing the spmv (y=A^Tx) on west0497, using dense vectors"
					$runner ${TEST_BIN_DIR}/dense_spmv_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 2 1 1 &> ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_ATx_${P}_${T}.log
					head -1 ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_ATx_${P}_${T}.log
					if grep -q 'Test OK' ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_ATx_${P}_${T}.log; then
						echo 'Test OK'
					else
						echo 'Test FAILED'
					fi
					echo " "

					echo ">>>      [x]           [ ]       Testing the spmv (y=xA) on west0497, using dense vectors"
					$runner ${TEST_BIN_DIR}/dense_spmv_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 3 1 1 &> ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xA_${P}_${T}.log
					head -1 ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xA_${P}_${T}.log
					if grep -q 'Test OK' ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xA_${P}_${T}.log; then
						echo 'Test OK'
					else
						echo 'Test FAILED'
					fi
					echo " "

					echo ">>>      [x]           [ ]       Testing the spmv (y=xA^T) on west0497, using dense vectors"
					$runner ${TEST_BIN_DIR}/dense_spmv_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 4 1 1 &> ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xAT_${P}_${T}.log
					head -1 ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xAT_${P}_${T}.log
					if grep -q 'Test OK' ${TEST_OUT_DIR}/dense_spmv_${MODE}_${BACKEND}_xAT_${P}_${T}.log; then
						echo 'Test OK'
					else
						echo 'Test FAILED'
					fi
					echo " "
				else
					echo "Test DISABLED: ${INPUT_DIR}/west0497.mtx was not found. To enabled, please provide the dataset."
				fi

				echo ">>>      [x]           [ ]       Testing BLAS1 functions on empty vectors"
				$runner ${TEST_BIN_DIR}/emptyVector_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/emptyVector_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/emptyVector_${MODE}_${BACKEND}_${P}_${T}.log
				grep -i "test ok" ${TEST_OUT_DIR}/emptyVector_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
				echo "                                 semiring over integers on a diagonal 15x15 matrix. Each"
				echo "                                 of the 15 output elements are computed element-by-element"
				echo "                                 using masked operations. The implementation should keep"
				echo "                                 both the mask and the output vector sparse. In this test,"
				echo "                                 the in_place variant is also tested-- there, also the"
				echo "                                 input vector shall be sparse."
				$runner ${TEST_BIN_DIR}/sparse_mxv_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/sparse_mxv_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/sparse_mxv_${MODE}_${BACKEND}_${P}_${T}.log
				grep -i 'test ok' ${TEST_OUT_DIR}/sparse_mxv_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
				echo "                                 semiring over integers on a 10x10 matrix. The input vector"
				echo "                                 is sparse. Each of y=Ax, y=A^Tx, y=xA, and y=xA^T is"
				echo "                                 tested in turn. The implementation should result in a"
				echo "                                 sparse output vector."
				echo "Functional test executable: ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND}"
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 10 1 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_1.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 10 2 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_2.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 10 3 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_3.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 10 4 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_4.log
				(grep -i "Test failed" ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_?.log) ||
					(grep -i "Test OK" ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_10_?.log) ||
					echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
				echo "                                 semiring over integers on a 1000x1000 matrix. The"
				echo "                                 input vector is sparse. Each of y=Ax, y=A^Tx, y=xA,"
				echo "                                 and y=xA^T is tested in turn. The implementation"
				echo "                                 should result in a sparse output vector."
				echo "Functional test executable: ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND}"
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 1000 1 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_1.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 1000 2 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_2.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 1000 3 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_3.log
				$runner ${TEST_BIN_DIR}/sparse_vxm_${MODE}_${BACKEND} 1000 4 1 1 &> ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_4.log
				(grep -i "Test failed" ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_?.log) ||
					(grep -i "Test OK" ${TEST_OUT_DIR}/sparse_vxm_${MODE}_${BACKEND}_${P}_${T}_1000_?.log) ||
					echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing dense vector times matrix using the double (+,*)"
				echo "                                 semiring where matrix elements are doubles and vector"
				echo "                                 elements ints. The input matrix is taken from west0497."
				echo " "
				if [ -f ${INPUT_DIR}/west0497.mtx ]; then
					$runner ${TEST_BIN_DIR}/vxm_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/vxm_${MODE}_${BACKEND}_${P}_${T}.west0497
					head -1 ${TEST_OUT_DIR}/vxm_${MODE}_${BACKEND}_${P}_${T}.west0497
					grep 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_${BACKEND}_${P}_${T}.west0497 || echo "Test FAILED"
				else
					echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
				fi
				echo " "

				echo ">>>      [x]           [ ]       Testing matrix times dense vector using the double (+,*)"
				echo "                                 semiring where matrix elements are doubles and vector"
				echo "                                 elements ints. The input matrix is taken from west0497."
				echo " "
				if [ -f ${INPUT_DIR}/west0497.mtx ]; then
					$runner ${TEST_BIN_DIR}/mxv_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/mxv_${MODE}_${BACKEND}_${P}_${T}.west0497
					head -1 ${TEST_OUT_DIR}/mxv_${MODE}_${BACKEND}_${P}_${T}.west0497
					grep 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_${BACKEND}_${P}_${T}.west0497 || echo "Test FAILED"
				else
					echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
				fi
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::wait on small inputs"
				$runner ${TEST_BIN_DIR}/wait_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/wait_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/wait_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/wait_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::wait on large inputs"
				$runner ${TEST_BIN_DIR}/wait_${MODE}_${BACKEND} 11733 &> ${TEST_OUT_DIR}/wait_large_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/wait_large_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/wait_large_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing building a matrix via input iterators"
				echo "                                 both sequentially and in parallel"
				$runner ${TEST_BIN_DIR}/buildMatrixUnique_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/buildMatrixUnique_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/buildMatrixUnique_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/buildMatrixUnique_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseApply using + on matrices"
				$runner ${TEST_BIN_DIR}/eWiseApply_matrix_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/eWiseApply_matrix_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/eWiseApply_matrix_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/eWiseApply_matrix_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::eWiseLambda (matrices)"
				$runner ${TEST_BIN_DIR}/eWiseMatrix_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/eWiseMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/eWiseMatrix_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/eWiseMatrix_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::zip on two vectors of doubles and"
				echo "                                 ints of size 10 000 000."
				$runner ${TEST_BIN_DIR}/zip_${MODE}_${BACKEND} 10000000 &> ${TEST_OUT_DIR}/zip_large_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/zip_large_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/zip_large_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing copy-constructor of square pattern matrices"
				echo "                                 of size 1003."
				$runner ${TEST_BIN_DIR}/copyVoidMatrices_${MODE}_${BACKEND} 1003 &> ${TEST_OUT_DIR}/copyVoidMatrices_${MODE}_${BACKEND}_${P}_${T}
				head -1 ${TEST_OUT_DIR}/copyVoidMatrices_${MODE}_${BACKEND}_${P}_${T}
				grep 'Test OK' ${TEST_OUT_DIR}/copyVoidMatrices_${MODE}_${BACKEND}_${P}_${T} || echo "Test FAILED"
				echo " "

				if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
					echo "Additional standardised unit tests not yet supported for the ${BACKEND} backend."
					echo
					continue
				fi

				echo ">>>      [x]           [ ]       Testing BLAS3 grb::mxm (unmasked) on simple matrices"
				echo "                                 of size 100 x 100 using the (+,*) semiring over"
				echo "                                 doubles"
				$runner ${TEST_BIN_DIR}/mxm_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/mxm_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/mxm_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/mxm_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing grb::outer on a small matrix"
				$runner ${TEST_BIN_DIR}/outer_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/outer_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/outer_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/outer_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing vector times matrix using the normal (+,*)"
				echo "                                 semiring over integers on a diagonal matrix"
				echo " "
				$runner ${TEST_BIN_DIR}/vmx_${MODE}_${BACKEND} 2> ${TEST_OUT_DIR}/vmx_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/vmx_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/vmx_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/vmx_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing vector times matrix using a (*,+) semiring over"
				echo "                                 doubles on a diagonal matrix"
				echo " "
				$runner ${TEST_BIN_DIR}/vmxa_${MODE}_${BACKEND} 2> ${TEST_OUT_DIR}/vmxa_${MODE}_${BACKEND}_${P}_${T}.err 1> ${TEST_OUT_DIR}/vmxa_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/vmxa_${MODE}_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/vmxa_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing vector times matrix using the number (+,*)"
				echo "                                 semiring over integers on a diagonal 15x15 matrix. Each"
				echo "                                 of the 15 output elements are computed, like the above"
				echo "                                 test, by masked operations. Instead of one element per"
				echo "                                 mask, this mask will have two elements. One element is"
				echo "                                 fixed to 3. All 14 combinations are tested. The in_place"
				echo "                                 specifier is tested as well."
				$runner ${TEST_BIN_DIR}/masked_vxm_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/masked_vxm_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/masked_vxm_${MODE}_${BACKEND}_${P}_${T}.log
				grep -i 'test ok' ${TEST_OUT_DIR}/masked_vxm_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
				echo "                                 semiring over integers on a diagonal 15x15 matrix--"
				echo "                                 apart from mxv instead of vxm, this is the same test"
				echo "                                 as the above."
				$runner ${TEST_BIN_DIR}/masked_mxv_${MODE}_${BACKEND} &> ${TEST_OUT_DIR}/masked_mxv_${MODE}_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/masked_mxv_${MODE}_${BACKEND}_${P}_${T}.log
				grep -i 'test ok' ${TEST_OUT_DIR}/masked_mxv_${MODE}_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
				echo " "

				echo ">>>      [x]           [ ]       Testing the spy algorithm on west0497. While perhaps not a pure unit"
				echo "                                 test, this test naturally hits uncommon border cases and hence is"
				echo "                                 retained within the unit test suite."
				if [ -f ${INPUT_DIR}/west0497.mtx ]; then
					$runner ${TEST_BIN_DIR}/spy_${MODE}_${BACKEND} ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/spy_${MODE}_${BACKEND}_${P}_${T}.log
					head -1 ${TEST_OUT_DIR}/spy_${MODE}_${BACKEND}_${P}_${T}.log
					if grep -q 'Test OK' ${TEST_OUT_DIR}/spy_${MODE}_${BACKEND}_${P}_${T}.log; then
						if grep 'Spy matrix' ${TEST_OUT_DIR}/spy_${MODE}_${BACKEND}_${P}_${T}.log | cut -d' ' -f9 | grep -q 315; then
							echo 'Test OK'
						else
							echo 'Verification FAILED'
							echo 'Test FAILED'
						fi
					else
						echo 'Test FAILED'
					fi
				else
					echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
				fi
				echo " "

				#if [ "$BACKEND" = "reference_omp" ]; then
				#	echo "Additional standardised unit tests not yet supported for the ${BACKEND} backend"
				#	echo
				#	continue
				#fi

				#none here: all unit tests are operational for reference_omp

			done
		done

		if [ "$BACKEND" = "bsp1d" ]; then
			echo "Additional unit tests for the BSP1D backend:"
			echo " "
			echo ">>>      [x]           [ ]       Testing BSP1D distribution for a vector of size 100000"
			echo " "
			${TEST_BIN_DIR}/distribution_bsp1d_${MODE}

			echo ">>>      [x]           [ ]       Testing dense vector times matrix using the double (+,*)"
			echo "                                 semiring where matrix elements are doubles and vector"
			echo "                                 elements ints. The input matrix is taken from west0497."
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
				${LPFRUN} -np 1 ${TEST_BIN_DIR}/vxm_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1
				${LPFRUN} -np 2 ${TEST_BIN_DIR}/vxm_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P2
				${LPFRUN} -np 3 ${TEST_BIN_DIR}/vxm_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P3
				${LPFRUN} -np 4 ${TEST_BIN_DIR}/vxm_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4
				head -1 ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4
				(grep -q 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_reference_1_1.west0497 && grep -q 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1 && grep -q 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P2 && grep -q 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P3 && grep -q 'Test OK' ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4 && printf "Test OK.\n") || printf "Test FAILED.\n"
				cat ${TEST_OUT_DIR}/vxm_${MODE}_reference_1_1.west0497 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n > ${TEST_OUT_DIR}/vxm_${MODE}.west0497.chk
				cat ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1.chk
				cat ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P2 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P2.chk
				cat ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P3 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P3.chk
				cat ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4.chk
				(diff -q ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/vxm_${MODE}.west0497.chk && printf "Verification (1 to serial) OK.\n") || printf "Verification (1 to serial) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P2.chk && printf "Verification (1 to 2) OK.\n") || printf "Verification (1 to 2) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P3.chk && printf "Verification (1 to 3) OK.\n") || printf "Verification (1 to 3) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/vxm_${MODE}_bsp1d.west0497.P4.chk && printf "Verification (1 to 4) OK.\n\n") || printf "Verification (1 to 4) FAILED.\n\n"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing matrix times dense vector using the double (+,*)"
			echo "                                 semiring where matrix elements are doubles and vector"
			echo "                                 elements ints. The input matrix is taken from west0497."
			echo " "
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
				${LPFRUN} -np 1 ${TEST_BIN_DIR}/mxv_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1
				${LPFRUN} -np 2 ${TEST_BIN_DIR}/mxv_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P2
				${LPFRUN} -np 3 ${TEST_BIN_DIR}/mxv_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P3
				${LPFRUN} -np 4 ${TEST_BIN_DIR}/mxv_${MODE}_bsp1d ${INPUT_DIR}/west0497.mtx &> ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4
				head -1 ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4
				(grep -q 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_reference_1_1.west0497 && grep -q 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1 && grep -q 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P2 && grep -q 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P3 && grep -q 'Test OK' ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4 && printf "Test OK.\n") || printf "Test FAILED.\n"
				cat ${TEST_OUT_DIR}/mxv_${MODE}_reference_1_1.west0497 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n > ${TEST_OUT_DIR}/mxv_${MODE}.west0497.chk
				cat ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1.chk
				cat ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P2 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P2.chk
				cat ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P3 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P3.chk
				cat ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4.chk
				(diff -q ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/mxv_${MODE}.west0497.chk && printf "Verification (1 to serial) OK.\n") || printf "Verification (1 to serial) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P2.chk && printf "Verification (1 to 2) OK.\n") || printf "Verification (1 to 2) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P3.chk && printf "Verification (1 to 3) OK.\n") || printf "Verification (1 to 3) FAILED.\n"
				(diff -q ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P1.chk ${TEST_OUT_DIR}/mxv_${MODE}_bsp1d.west0497.P4.chk && printf "Verification (1 to 4) OK.\n\n") || printf "Verification (1 to 4) FAILED.\n\n"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing BSP1D distribution."
			echo " "
			${LPFRUN} -np 1 ${TEST_BIN_DIR}/distribution_${MODE}
		fi
	done

	echo ">>>      [x]           [ ]       Testing threadlocal storage, parallel, double values,"
	echo "                                 including checks for const-correctness."
	echo " "
	${TEST_BIN_DIR}/thread_local_storage_${MODE}

done

echo
echo "*****************************************************************************************"
echo "All unit tests done."
echo " "

