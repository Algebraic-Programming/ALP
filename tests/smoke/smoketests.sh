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

TESTS_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ &> /dev/null && pwd )"
source ${TESTS_ROOT_DIR}/parse_env.sh

if [ -z "${GNN_DATASET_PATH}" ]; then
	echo "Info: GNN_DATASET_PATH was undefined or empty; trying ${INPUT_DIR}/GraphChallengeDataset"
	export GNN_DATASET_PATH=${INPUT_DIR}/GraphChallengeDataset
fi

if [ ! -d "${GNN_DATASET_PATH}" ]; then
	echo "Warning: GNN_DATASET_PATH does not exist. Some tests will not run without input GNN data."
fi


LABELTEST_SIZES=(8 256 4096) # for size 32, the ground-truth number of iterations is 6. This size is
LABELTEST_RESULTS=(4 9 13)   # disabled as there is no reason why this should behave differently
                             # from size 8 (both will map to the same single thread and process).

echo " "
echo " "
echo "****************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "----------------------------------------------------------------------------------------"
echo " "
# run non alp_ backends
for BACKEND in ${BACKENDS[@]}; do
	if [ "${BACKEND:0:4}" == "alp_" ]; then
	    continue
	fi
	
	if [ "$BACKEND" = "bsp1d" ]; then
		if [ -z "${LPFRUN}" ]; then
			echo "LPFRUN is not set!"
			exit 255;
		fi
		if [ -z "${MANUALRUN}" ]; then
			echo "MANUALRUN is not set!"
			exit 255;
		fi
	fi

	if [ "$BACKEND" = "bsp1d" ]; then
		Ps=( 5 )
	fi
	if [ "$BACKEND" = "hybrid" ]; then
		Ps=( 2 )
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
	if [ "$BACKEND" = "reference_omp" ] ; then
		Pt=( ${MAX_THREADS} )
	elif [ "$BACKEND" = "hybrid" ]; then
		MTDS=$((MAX_THREADS/2))
		if [ "$MTDS" -le "2" ]; then
			Pt=( 2 )
		else
			Pt=( ${MTDS} )
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

			echo "#################################################################"
			echo "# Starting standardised smoke tests for the ${BACKEND} backend"
			echo "#   using ${P} user processes"
			echo "#   using ${T} threads"
			if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
				echo "#   using \`\`${LPFRUN}'' for automatic launchers"
				echo "#   using \`\`${MANUALRUN}'' for manual launchers"
			fi
			if [ "x${runner}" != "x" ]; then
				echo "#   using runner \`\`$runner''"
			fi
			echo "#################################################################"
			echo " "

			echo ">>>      [x]           [ ]       Tests k-nearest-neighbourhood (k-NN) calculation through"
			echo "                                 breadth-first search on a tiny graph."
			bash -c "$runner ${TEST_BIN_DIR}/small_knn_${BACKEND} ${P} &> ${TEST_OUT_DIR}/small_knn_${BACKEND}_${P}_${T}.log"
			head -1 ${TEST_OUT_DIR}/small_knn_${BACKEND}_${P}_${T}.log
			grep 'Test OK' ${TEST_OUT_DIR}/small_knn_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [x]       Tests an automatically launching version of the k-NN on"
			echo "                                 the facebook_combined dataset for k=4 in serial mode."
			echo "                                 Uses file IO in direct mode."
			if [ -f ${INPUT_DIR}/facebook_combined.txt ]; then
				$runner ${TEST_BIN_DIR}/knn_${BACKEND} 4 ${INPUT_DIR}/facebook_combined.txt direct 1 1 &> ${TEST_OUT_DIR}/knn_${BACKEND}_${P}_${T}_facebook.log
				head -1 ${TEST_OUT_DIR}/knn_${BACKEND}_${P}_${T}_facebook.log
				if grep -q "Test OK" ${TEST_OUT_DIR}/knn_${BACKEND}_${P}_${T}_facebook.log; then
					(grep -q "Neighbourhood size is 499" ${TEST_OUT_DIR}/knn_${BACKEND}_${P}_${T}_facebook.log && printf "Test OK\n\n") || (printf "Test FAILED (verification error)\n")
				else
					printf "Test FAILED\n"
				fi
			else
				echo "Test DISABLED; dataset not found. Provide facebook_combined.txt in the ${INPUT_DIR} directory to enable."
			fi
			echo " "

			echo ">>>      [x]           [ ]       Tests HPCG on a small matrix"
			bash -c "$runner ${TEST_BIN_DIR}/hpcg_${BACKEND} &> ${TEST_OUT_DIR}/hpcg_${BACKEND}_${P}_${T}.log"
			head -1 ${TEST_OUT_DIR}/hpcg_${BACKEND}_${P}_${T}.log
			grep 'Test OK' ${TEST_OUT_DIR}/hpcg_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Tests an automatically launching version of the simple pagerank"
			echo "                                 algorithm for a small 10 x 10 problem. Verifies against known"
			echo "                                 output."
			echo "Functional test executable: ${TEST_BIN_DIR}/small_pagerank_${BACKEND}"
			bash -c "$runner ${TEST_BIN_DIR}/small_pagerank_${BACKEND} ${P} &> ${TEST_OUT_DIR}/small_pagerank_${BACKEND}_${P}_${T}.log"
			if grep -q 'Test OK' ${TEST_OUT_DIR}/small_pagerank_${BACKEND}_${P}_${T}.log; then
				(grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' ${TEST_OUT_DIR}/small_pagerank_${BACKEND}_${P}_${T}.log && printf "Test OK\n\n") || printf "Test FAILED (verification error)\n\n"
			else
				printf "Test FAILED\n\n"
			fi

			echo ">>>      [x]           [ ]       Testing the pagerank algorithm for the 497 by 497 matrix"
			echo "                                 west0497.mtx. This test verifies against a ground-truth"
			echo "                                 PageRank vector for this dataset. The test employs the"
			echo "                                 grb::Launcher in automatic mode with statically sized IO,"
			echo "                                 and uses sequential file IO in direct mode."
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
			$runner ${TEST_BIN_DIR}/simple_pagerank_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 1 1 verification ${OUTPUT_VERIFICATION_DIR}/pagerank_out_west0497_ref &> ${TEST_OUT_DIR}/simple_pagerank_${BACKEND}_west0497_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/simple_pagerank_${BACKEND}_west0497_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/simple_pagerank_${BACKEND}_west0497_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on the SNAP dataset"
			echo "                                 facebook_combined. The launcher is used in automatic"
			echo "                                 mode and the IO mode is sequential in direct mode."
			echo "                                 Launcher::exec is used with statically sized input and"
			echo "                                 statically sized output."
			echo "Functional test executable: ${TEST_BIN_DIR}/simple_pagerank_${BACKEND}"
			if [ -f ${INPUT_DIR}/facebook_combined.txt ]; then
				$runner ${TEST_BIN_DIR}/simple_pagerank_${BACKEND} ${INPUT_DIR}/facebook_combined.txt direct 1 1 &> ${TEST_OUT_DIR}/simple_pagerank_${BACKEND}_facebook_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/simple_pagerank_${BACKEND}_facebook_${P}_${T}.log || printf 'Test FAILED.\n'
			else
				echo "Test DISABLED; dataset not found. Provide facebook_combined.txt in the ./datasets/ directory to enable."
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing the conjugate gradient algorithm for the input"
			echo "                                 matrix (17361x17361) taken from gyro_m.mtx. This test"
			echo "                                 verifies against a ground-truth solution vector. The test"
			echo "                                 employs the grb::Launcher in automatic mode. It uses"
			echo "                                 direct-mode file IO."
			if [ -f ${INPUT_DIR}/gyro_m.mtx ]; then
				$runner ${TEST_BIN_DIR}/conjugate_gradient_${BACKEND} ${INPUT_DIR}/gyro_m.mtx direct 1 1 verification ${OUTPUT_VERIFICATION_DIR}/conjugate_gradient_out_gyro_m_ref &> ${TEST_OUT_DIR}/conjugate_gradient_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/conjugate_gradient_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/conjugate_gradient_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: gyro_m.mtx was not found. To enable, please provide ${INPUT_DIR}/gyro_m.mtx"
			fi
			echo " "

			TESTNAME=rndHermit256
			if [ -f ${TEST_DATA_DIR}/${TESTNAME}.mtx ]; then
				n=$(grep -v '^%' ${TEST_DATA_DIR}/${TESTNAME}.mtx | head -1 | awk '{print $1}' )
				m=$(grep -v '^%' ${TEST_DATA_DIR}/${TESTNAME}.mtx | head -1 | awk '{print $2}' )
				echo ">>>      [x]           [ ]       Testing the conjugate gradient complex  algorithm for the input"
				echo "                                 matrix (${n}x${m}) taken from ${TESTNAME}.mtx. This test"
				echo "                                 verifies against a ground-truth solution vector. The test"
				echo "                                 employs the grb::Launcher in automatic mode. It uses"
				echo "                                 direct-mode file IO."
				$runner ${TEST_BIN_DIR}/conjugate_gradient_complex_${BACKEND} ${TEST_DATA_DIR}/${TESTNAME}.mtx direct 1 1 verification ${OUTPUT_VERIFICATION_DIR}/complex_conjugate_conjugate_gradient_out_${TESTNAME}_ref &> ${TEST_OUT_DIR}/conjugate_gradient_complex_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/conjugate_gradient_complex_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/conjugate_gradient_complex_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: ${TESTNAME}.mtx was not found. To enable, please provide ${TEST_DATA_DIR}/${TESTNAME}.mtx"
			fi
			echo " "
			
			echo ">>>      [x]           [ ]       Testing the BiCGstab algorithm for the 17361 x 17361 input"
			echo "                                 matrix gyro_m.mtx. This test verifies against a ground-"
			echo "                                 truth solution vector, the same as used for the earlier"
			echo "                                 conjugate gradient test. Likewise to that one, this test"
			echo "                                 employs the grb::Launcher in automatic mode. It uses"
			echo "                                 direct-mode file IO."
			if [ -f ${INPUT_DIR}/gyro_m.mtx ]; then
				$runner ${TEST_BIN_DIR}/bicgstab_${BACKEND} ${INPUT_DIR}/gyro_m.mtx direct 1 1 verification ${OUTPUT_VERIFICATION_DIR}/conjugate_gradient_out_gyro_m_ref &> ${TEST_OUT_DIR}/bicgstab_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/bicgstab_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/bicgstab_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: gyro_m.mtx was not found. To enable, please provide ${INPUT_DIR}/gyro_m.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing the Sparse Neural Network algorithm for the GraphChallenge"
			echo "                                 dataset (neurons=1024, layers=120, offset=294) taken from"
			echo "                                 ${GNN_DATASET_PATH} and using thresholding 32."
			if [ -d ${GNN_DATASET_PATH} ]; then
				$runner ${TEST_BIN_DIR}/graphchallenge_nn_single_inference_${BACKEND} ${GNN_DATASET_PATH} 1024 120 294 1 32 indirect 1 1 verification ${OUTPUT_VERIFICATION_DIR}/graphchallenge_nn_out_1024_120_294_32_threshold_ref &> ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: ${GNN_DATASET_PATH} was not found. To enable, please provide the dataset."
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing the Sparse Neural Network algorithm for the GraphChallenge"
			echo "                                 dataset (neurons=1024, layers=120, offset=294) taken from"
			echo "                                 ${GNN_DATASET_PATH} and without using thresholding."
			if [ -d ${GNN_DATASET_PATH} ]; then
				$runner ${TEST_BIN_DIR}/graphchallenge_nn_single_inference_${BACKEND} ${GNN_DATASET_PATH} 1024 120 294 0 0 indirect 1 1 verification ${OUTPUT_VERIFICATION_DIR}/graphchallenge_nn_out_1024_120_294_no_threshold_ref &> ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log
				grep 'Test OK' ${TEST_OUT_DIR}/graphchallenge_nn_single_inference_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			else
				echo "Test DISABLED: ${GNN_DATASET_PATH} was not found. To enable, please provide the dataset."
			fi
			echo " "

			for ((i=0;i<${#LABELTEST_SIZES[@]};++i));
			do
				LABELTEST_SIZE=${LABELTEST_SIZES[i]}
				LABELTEST_EXPECTED_RESULT=${LABELTEST_RESULTS[i]}
				echo ">>>      [x]           [ ]       Testing label propagation using a a generated dataset"
				echo "                                 of size ${LABELTEST_SIZE} using the ${BACKEND} backend."
				echo "                                 This test verifies the number of iterations required"
				echo "                                 to convergence against the ground-truth value of ${LABELTEST_EXPECTED_RESULT}"
				$runner ${TEST_BIN_DIR}/labeltest_${BACKEND} ${LABELTEST_SIZE} &> ${TEST_OUT_DIR}/labeltest_${BACKEND}_${LABELTEST_SIZE}.log
				head -1 ${TEST_OUT_DIR}/labeltest_${BACKEND}_${LABELTEST_SIZE}.log
				(grep -q "converged in ${LABELTEST_EXPECTED_RESULT} iterations" ${TEST_OUT_DIR}/labeltest_${BACKEND}_${LABELTEST_SIZE}.log && grep -i 'test ok' ${TEST_OUT_DIR}/labeltest_${BACKEND}_${LABELTEST_SIZE}.log) || echo "Test FAILED"
				echo " "
			done

			echo ">>>      [x]           [ ]       Testing the Pregel PageRank-like algorithm using a global"
			echo "                                 stopping criterion. Verifies via a simple regression test in"
			echo "                                 number of rounds required."
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
				$runner ${TEST_BIN_DIR}/pregel_pagerank_global_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 1 1 &> ${TEST_OUT_DIR}/pregel_pagerank_global_west0497_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/pregel_pagerank_global_west0497_${BACKEND}_${P}_${T}.log
				if ! grep -q 'Test OK' ${TEST_OUT_DIR}/pregel_pagerank_global_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Test FAILED"
				elif ! grep -q '56 iterations to converge' ${TEST_OUT_DIR}/pregel_pagerank_global_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Verification FAILED"
					echo "Test FAILED"
				else
					echo "Test OK"
				fi
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing the Pregel PageRank-like algorithm using a vertex-local"
			echo "                                 stopping criterion. Verifies via a simple regression test in"
			echo "                                 number of rounds required."
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
				$runner ${TEST_BIN_DIR}/pregel_pagerank_local_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 1 1 &> ${TEST_OUT_DIR}/pregel_pagerank_local_west0497_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/pregel_pagerank_local_west0497_${BACKEND}_${P}_${T}.log
				if ! grep -q 'Test OK' ${TEST_OUT_DIR}/pregel_pagerank_local_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Test FAILED"
				elif ! grep -q '47 iterations to converge' ${TEST_OUT_DIR}/pregel_pagerank_local_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Verification FAILED"
					echo "Test FAILED"
				else
					echo "Test OK"
				fi
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing the Pregel connected components algorithm. Verifies"
			echo "                                 using a simple regression test in number of rounds required."
			if [ -f ${INPUT_DIR}/west0497.mtx ]; then
				$runner ${TEST_BIN_DIR}/pregel_connected_components_${BACKEND} ${INPUT_DIR}/west0497.mtx direct 1 1 &> ${TEST_OUT_DIR}/pregel_connected_components_west0497_${BACKEND}_${P}_${T}.log
				head -1 ${TEST_OUT_DIR}/pregel_connected_components_west0497_${BACKEND}_${P}_${T}.log
				if ! grep -q 'Test OK' ${TEST_OUT_DIR}/pregel_connected_components_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Test FAILED"
				elif ! grep -q '11 iterations to converge' ${TEST_OUT_DIR}/pregel_connected_components_west0497_${BACKEND}_${P}_${T}.log; then
					echo "Verification FAILED"
					echo "Test FAILED"
				else
					echo "Test OK"
				fi
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide ${INPUT_DIR}/west0497.mtx"
			fi
			echo " "

			if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
				echo "Additional standardised smoke tests not yet supported for the ${BACKEND} backend"
				echo
				continue
			fi

			echo ">>>      [x]           [ ]       Testing the k-means algorithm"
			$runner ${TEST_BIN_DIR}/kmeans_${BACKEND} &> ${TEST_OUT_DIR}/kmeans_${BACKEND}_${P}_${T}.log
			head -1 ${TEST_OUT_DIR}/kmeans_${BACKEND}_${P}_${T}.log
			tail -1 ${TEST_OUT_DIR}/kmeans_${BACKEND}_${P}_${T}.log
			echo " "

		done
	done

	if [ "$BACKEND" = "bsp1d" ]; then
		echo "Non-standard BSP1D-specific smoke tests:"
		echo " "

		echo ">>>      [x]           [ ]       Tests a manual call to bsp_hook via LPF. This is a smoke"
		echo "                                 test that makes sure the manual launcher is operational"
		echo "                                 via a simple ``hello world' test."
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_hw. Script hardcodes test for four"
		echo "separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_hw localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_hw.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_hw localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_hw.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_hw localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_hw.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_hw localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_hw.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_hw.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_hw.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_hw.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_hw.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::set"
		echo "                                 function over an array of doubles of 100 elements"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_set. Script hardcodes test for"
		echo "three separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_set localhost 0 3 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_set.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_set localhost 1 3 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_set.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_set localhost 2 3 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_set.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_set.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_set.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_set.0 && printf "Test OK.\n\n" ) || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::reduce"
		echo "                                 function over an array of doubles"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_reduce. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_reduce localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_reduce.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_reduce localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_reduce.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_reduce localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_reduce.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_reduce localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_reduce.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_reduce.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_reduce.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_reduce.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_reduce.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::set"
		echo "                                 function over an array of ints of 100 000 elements"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_dot. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_dot localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_dot.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_dot localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_dot.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_dot localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_dot.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_dot localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_dot.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_dot.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_dot.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_dot.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_dot.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas0 grb::collectives"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas0. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas0 localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas0 localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas0 localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas0 localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas0.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas1 grb::collectives"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1 localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1 localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1 localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1 localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas1 grb::collectives"
		echo " "
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1_raw. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1_raw localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1_raw localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1_raw localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_grb_collectives_blas1_raw localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_grb_collectives_blas1_raw.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Tests manually hooked k-nearest-neighbourhood"
		echo "                                 calculation on a tiny graph, using 4 processes"
		echo "Functional test executable: ${TEST_BIN_DIR}/manual_hook_small_knn. Script hardcodes test for four"
		echo "separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_small_knn localhost 0 4 77770 &> ${TEST_OUT_DIR}/manual_hook_small_knn.0 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_small_knn localhost 3 4 77770 &> ${TEST_OUT_DIR}/manual_hook_small_knn.3 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_small_knn localhost 1 4 77770 &> ${TEST_OUT_DIR}/manual_hook_small_knn.1 & \
			${MANUALRUN} ${TEST_BIN_DIR}/manual_hook_small_knn localhost 2 4 77770 &> ${TEST_OUT_DIR}/manual_hook_small_knn.2 & \
			wait"
		(grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_small_knn.1 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_small_knn.2 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_small_knn.3 && grep -q 'Test OK' ${TEST_OUT_DIR}/manual_hook_small_knn.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 4, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with statically sized input and statically sized"
		echo "                                 output."
		echo "Functional test executable: ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank"
		bash -c "(set -o pipefail && ${LPFRUN} -np 4 ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank &> ${TEST_OUT_DIR}/from_mpi_launch_simple_pagerank && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 5, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with statically sized input and statically sized"
		echo "                                 output. The entire test is repeated three times, to"
		echo "                                 test re-entrance capabilities of the 1) Launcher"
		echo "                                 constructor, 2) Launcher destructor, and 3) exec"
		echo "                                 function."
		echo "Functional test executable: ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 5 ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_multiple_entry &> ${TEST_OUT_DIR}/from_mpi_launch_simple_pagerank_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 3, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with variably sized input and statically sized"
		echo "                                 output containing a PinnedVector instance. The input"
		echo "                                 at PID 0 is broadcasted to all other processes. The"
		echo "                                 entire test is repeated three times, to test re-"
		echo "                                 entrance capabilities of the 1) Launcher constructor,"
		echo "                                 2) Launcher destructor, and 3) exec function."
		echo "Functional test executable: ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 3 ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry &> ${TEST_OUT_DIR}/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 7, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with variably sized input and statically sized"
		echo "                                 output. The input at PID 0 is broadcasted to all other"
		echo "                                 processes. The entire test is repeated three times, to"
		echo "                                 test re-entrance capabilities of the 1) Launcher"
		echo "                                 constructor, 2) Launcher destructor, and 3) exec"
		echo "                                 function."
		echo "Functional test executable: ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 7 ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast_multiple_entry &> ${TEST_OUT_DIR}/from_mpi_launch_simple_pagerank_broadcast_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 6, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with variably sized input and statically sized"
		echo "                                 output."
		echo "Functional test executable: ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast"
		bash -c "(set -o pipefail && ${LPFRUN} -np 6 ${TEST_BIN_DIR}/from_mpi_launch_simple_pagerank_broadcast &> ${TEST_OUT_DIR}/from_mpi_launch_simple_pagerank_broadcast && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

	fi
done

for BACKEND in ${BACKENDS[@]}; do
	if [ "${BACKEND:0:4}" != "alp_" ]; then
		continue
	fi

	runner=
	echo "#################################################################"
	echo "# Starting standardised smoke tests for the ${BACKEND} backend"
	if [ "x${runner}" != "x" ]; then
	    echo "#   using runner \`\`$runner''"
	fi
	echo "#################################################################"
	echo " "

	NTEST_CHOLESKY=30
	echo ">>>      [x]           [ ]       Tests Cholesky decomposition for a random"
	echo "                                 symmetric matrix (${NTEST_CHOLESKY}x${NTEST_CHOLESKY})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_cholesky_${BACKEND}  -n ${NTEST_CHOLESKY} &> ${TEST_OUT_DIR}/alp_cholesky_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_cholesky_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_cholesky_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_GEMM=100
	echo ">>>      [x]           [ ]       Tests Gemm on matrix (${NTEST_GEMM}x${NTEST_GEMM}x${NTEST_GEMM})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_gemm_${BACKEND} ${NTEST_GEMM} &> ${TEST_OUT_DIR}/alp_gemm_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_gemm_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_gemm_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_HOUSEHOLDER=100
	echo ">>>      [x]           [ ]       Tests dsytrd (Householder tridiagonalisaiton) on"
	echo ">>>                               a real symmetric matrix (${NTEST_HOUSEHOLDER}x${NTEST_HOUSEHOLDER})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_zhetrd_${BACKEND} ${NTEST_HOUSEHOLDER} &> ${TEST_OUT_DIR}/alp_zhetrd_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_zhetrd_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_zhetrd_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_HOUSEHOLDER_COMPLEX=100
	echo ">>>      [x]           [ ]       Tests zhetrd (Householder tridiagonalisaiton) on"
	echo ">>>                               a complex hermitian matrix (${NTEST_HOUSEHOLDER_COMPLEX}x${NTEST_HOUSEHOLDER_COMPLEX})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_zhetrd_complex_${BACKEND} ${NTEST_HOUSEHOLDER_COMPLEX} &> ${TEST_OUT_DIR}/alp_zhetrd_complex_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_zhetrd_complex_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_zhetrd_complex_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_HOUSEHOLDER=20
	echo ">>>      [x]           [ ]       Tests dgeqrf (Householder QR decomposition) on"
	echo ">>>                               a real symmetric matrix (${NTEST_HOUSEHOLDER} x 2x${NTEST_HOUSEHOLDER})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_zgeqrf_${BACKEND} ${NTEST_HOUSEHOLDER} &> ${TEST_OUT_DIR}/alp_zgeqrf_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_zgeqrf_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_zgeqrf_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_HOUSEHOLDER_COMPLEX=20
	echo ">>>      [x]           [ ]       Tests zgeqrf (Householder QR decomposition) on"
	echo ">>>                               a complex hermitian matrix (${NTEST_HOUSEHOLDER_COMPLEX} x 2x${NTEST_HOUSEHOLDER_COMPLEX})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_zgeqrf_complex_${BACKEND} ${NTEST_HOUSEHOLDER_COMPLEX} &> ${TEST_OUT_DIR}/alp_zgeqrf_complex_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_zgeqrf_complex_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_zgeqrf_complex_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_DIVCON=100
	echo ">>>      [x]           [ ]       Tests dstedc (Divide and conquer tridiagonal eigensolver) on"
	echo ">>>                               a tridiagonal real symmetric matrix (${NTEST_DIVCON}x${NTEST_DIVCON})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_dstedc_${BACKEND} ${NTEST_DIVCON} &> ${TEST_OUT_DIR}/alp_dstedc_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_dstedc_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_dstedc_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_BACKSUB=100
	echo ">>>      [x]           [ ]       Tests dtrsv and dtrsm (Triangular linear system solve using backsubstitution ) on"
	echo ">>>                               a upper tridiagonal real matrix (${NTEST_BACKSUB}x${NTEST_BACKSUB})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_backsubstitution_${BACKEND} ${NTEST_BACKSUB} &> ${TEST_OUT_DIR}/alp_backsubstitution_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_backsubstitution_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_backsubstitution_${BACKEND}.log || echo "Test FAILED"
	echo " "

	NTEST_BACKSUB=100
	echo ">>>      [x]           [ ]       Tests ztrsv and ztrsm (Triangular linear system solve using backsubstitution ) on"
	echo ">>>                               a upper tridiagonal complex matrix (${NTEST_BACKSUB}x${NTEST_BACKSUB})."
	bash -c "$runner ${TEST_BIN_DIR}/alp_backsubstitution_complex_${BACKEND} ${NTEST_BACKSUB} &> ${TEST_OUT_DIR}/alp_backsubstitution_complex_${BACKEND}.log"
	head -1 ${TEST_OUT_DIR}/alp_backsubstitution_complex_${BACKEND}.log
	grep 'Test OK' ${TEST_OUT_DIR}/alp_backsubstitution_complex_${BACKEND}.log || echo "Test FAILED"
	echo " "
	
done

echo "*****************************************************************************************"
echo "All smoke tests done."
echo " "


