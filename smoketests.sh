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

if [ -z "${BACKENDS}" ]; then
	echo "BACKENDS is not set!"
	exit 255;
fi
echo "Info: BACKENDS is set to ${BACKENDS}"

# Remove the below check once non-LPF smoketests are added
if [ -z "${LPFRUN}" ]; then
	echo "All current smoketests require LPF but the LPFRUN environment variable was not set -- skipping tests."
	exit 0;
fi

lpf_tests_ran=false

mkdir bin/tests/output || true

echo " "
echo " "
echo "****************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "----------------------------------------------------------------------------------------"
for BACKEND in ${BACKENDS[@]}; do
	if [ "$BACKEND" = "bsp1d" ]; then
		if [ "$lpf_tests_ran" = false ]; then
			if [ -z "${LPFRUN}" ]; then
				echo "LPFRUN is not set!"
				exit 255;
			fi
			if [ -z "${MANUALRUN}" ]; then
				echo "MANUALRUN is not set!"
				exit 255;
			fi
			echo "LPF-specific smoke tests:"
			echo "   - LPFRUN is set to ${LPFRUN}"
			echo "   - MANUALRUN is set to ${MANUALRUN}"
			echo " "
			echo ">>>      [x]           [ ]       Tests sequential k-nearest-neighbourhood calculation on"
			echo "                                 a tiny graph. Launched using lpf."
			bash -c "${LPFRUN} -np 1 bin/tests/sequential_hook_knn &> bin/tests/output/sequential_hook_knn"
			head -1 bin/tests/output/sequential_hook_knn
			grep 'Test OK' bin/tests/output/sequential_hook_knn
			echo " "

			lpf_tests_ran=true
		fi
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
		echo "BSP1D-specific smoke tests:"
		echo "   - LPFRUN is set to ${LPFRUN}"
		echo "   - MANUALRUN is set to ${MANUALRUN}"
		echo " "

		echo ">>>      [x]           [ ]       Tests parallel k-nearest-neighbourhood calculation on"
		echo "                                 a tiny graph. Uses 3 processes."
		bash -c "${LPFRUN} -np 3 bin/tests/automatic_hook_knn 3 &> bin/tests/output/automatic_hook_knn"
		head -1 bin/tests/output/automatic_hook_knn
		grep 'Test OK' bin/tests/output/automatic_hook_knn
		echo " "

		echo ">>>      [x]           [ ]       Tests manually hooked k-nearest-neighbourhood"
		echo "                                 calculation on a tiny graph, using 4 processes"
		echo "Functional test executable: bin/tests/manual_hook_knn. Script hardcodes test for four"
		echo "separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_knn localhost 0 4 77770 &> bin/tests/output/manual_hook_knn.0 & \
			${MANUALRUN} bin/tests/manual_hook_knn localhost 3 4 77770 &> bin/tests/output/manual_hook_knn.3 & \
			${MANUALRUN} bin/tests/manual_hook_knn localhost 1 4 77770 &> bin/tests/output/manual_hook_knn.1 & \
			${MANUALRUN} bin/tests/manual_hook_knn localhost 2 4 77770 &> bin/tests/output/manual_hook_knn.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_knn.1 && grep -q 'Test OK' bin/tests/output/manual_hook_knn.2 && grep -q 'Test OK' bin/tests/output/manual_hook_knn.3 && grep -q 'Test OK' bin/tests/output/manual_hook_knn.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")

		echo ">>>      [x]           [ ]       Tests an automatically launching version of the simple pagerank"
		echo "                                 algorithm for 1, 2, and 4 processes. Verifies against known output."
		echo " "
		echo "Functional test executable: bin/tests/automatic_hook_simple_pagerank <np>."
		bash -c "${LPFRUN} -np 1 bin/tests/automatic_hook_simple_pagerank 1 &> bin/tests/output/automatic_hook_simple_pagerank_p1"
		bash -c "${LPFRUN} -np 2 bin/tests/automatic_hook_simple_pagerank 2 &> bin/tests/output/automatic_hook_simple_pagerank_p2"
		bash -c "${LPFRUN} -np 4 bin/tests/automatic_hook_simple_pagerank 4 &> bin/tests/output/automatic_hook_simple_pagerank_p4"
		(grep -q 'Test OK' bin/tests/output/automatic_hook_simple_pagerank_p1 && grep -q 'Test OK' bin/tests/output/automatic_hook_simple_pagerank_p2 && grep -q 'Test OK' bin/tests/output/automatic_hook_simple_pagerank_p4) || (printf "Test FAILED.\n\n")
		(grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' bin/tests/output/automatic_hook_simple_pagerank_p1) || (printf "Verification at P=1 FAILED.\n\n")
		(grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' bin/tests/output/automatic_hook_simple_pagerank_p2) || (printf "Verification at P=2 FAILED.\n\n")
		(grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' bin/tests/output/automatic_hook_simple_pagerank_p4) || (printf "Verification at P=4 FAILED.\n\n")
		grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' bin/tests/output/automatic_hook_simple_pagerank_p? && printf "Test OK.\n\n"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 4, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with statically sized input and statically sized"
		echo "                                 output."
		echo "Functional test executable: bin/tests/from_mpi_launch_simple_pagerank"
		bash -c "(set -o pipefail && ${LPFRUN} -np 4 bin/tests/from_mpi_launch_simple_pagerank &> bin/tests/output/from_mpi_launch_simple_pagerank && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on the SNAP dataset"
		echo "                                 facebook_combined. The launcher is used in automatic"
		echo "                                 mode, IO is sequential, number of processes is 3, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with statically sized input and statically sized"
		echo "                                 output."
		echo "Functional test executable: bin/tests/automatic_launch_simple_pagerank"
		if [ -f datasets/facebook_combined.txt ]; then
			${LPFRUN} -np 3 bin/tests/automatic_launch_simple_pagerank datasets/facebook_combined.txt direct 1 1 &> bin/tests/output/automatic_launch_simple_pagerank
			grep -A2 'Test OK' bin/tests/output/automatic_launch_simple_pagerank || printf 'Test FAILED.\n\n'
		else
			echo "Test DISABLED; dataset not found. Provide facebook_combined.txt in the ./datasets/ directory to enable."
		fi

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
		echo "Functional test executable: bin/tests/from_mpi_launch_simple_pagerank_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 5 bin/tests/from_mpi_launch_simple_pagerank_multiple_entry &> bin/tests/output/from_mpi_launch_simple_pagerank_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

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
		echo "Functional test executable: bin/tests/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 3 bin/tests/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry &> bin/tests/output/from_mpi_launch_simple_pagerank_broadcast_pinning_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

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
		echo "Functional test executable: bin/tests/from_mpi_launch_simple_pagerank_broadcast_multiple_entry"
		bash -c "(set -o pipefail && ${LPFRUN} -np 7 bin/tests/from_mpi_launch_simple_pagerank_broadcast_multiple_entry &> bin/tests/output/from_mpi_launch_simple_pagerank_broadcast_multiple_entry && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [ ]       Tests grb::Launcher on a PageRank on a 1M x 1M matrix"
		echo "                                 with 1M+1 nonzeroes. The matrix corresponds to a cycle"
		echo "                                 path through all 1M vertices, plus one edge from vertex"
		echo "                                 1M-3 to vertex 1M-1. The launcher is used in FROM_MPI"
		echo "                                 mode, IO is sequential, number of processes is 6, and"
		echo "                                 the backend implementation is BSP1D. Launcher::exec is"
		echo "                                 used with variably sized input and statically sized"
		echo "                                 output."
		echo "Functional test executable: bin/tests/from_mpi_launch_simple_pagerank_broadcast"
		bash -c "(set -o pipefail && ${LPFRUN} -np 6 bin/tests/from_mpi_launch_simple_pagerank_broadcast &> bin/tests/output/from_mpi_launch_simple_pagerank_broadcast && printf 'Test OK.\n\n') || (printf 'Test FAILED.\n\n')"

		echo ">>>      [x]           [x]       Tests an automatically launching version of the k-NN on"
		echo "                                 the facebook_combined dataset for k=4 in serial mode."
		if [ -f datasets/facebook_combined.txt ]; then
			${LPFRUN} -np 1 bin/tests/automatic_launch_knn_debug 4 datasets/facebook_combined.txt direct &> bin/tests/output/automatic_launch_knn_debug
			head -1 bin/tests/output/automatic_launch_knn_debug
			(grep -q "Test OK." bin/tests/output/automatic_launch_knn_debug) || (printf "Test FAILED.\n\n")
			(grep -q "Neighbourhood size is 499." bin/tests/output/automatic_launch_knn_debug && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
		else
			echo "Test DISABLED; dataset not found. Provide facebook_combined.txt in the ./datasets/ directory to enable."
		fi

		echo ">>>     [x]            [ ]      Tests an automatically launching version of the simple pagerank"
		echo "                                algorithm using OpenMP. Verifies against known output."
		echo "Functional test executable: bin/tests/automatic_hook_simple_pagerank-openmp"
		bash -c "${LPFRUN} -np 1 bin/tests/automatic_hook_simple_pagerank-openmp &> bin/tests/output/automatic_hook_simple_pagerank-openmp"
		(grep -q 'Test OK' bin/tests/output/automatic_hook_simple_pagerank-openmp) || (printf "Test FAILED.\n\n")
		(grep -q 'Pagerank vector local to PID 0 on exit is ( 0.106896 0.105862 0.104983 0.104235 0.1036 0.10306 0.102601 0.102211 0.0584396 0.108113 )' bin/tests/output/automatic_hook_simple_pagerank-openmp) || (printf "Verification FAILED.\n\n")
		printf "Test OK.\n\n"

		echo ">>>      [x]           [ ]       Tests HPCG on a small matrix"
		bash -c "${LPFRUN} -np 1 bin/tests/hpcg_bsp1d &> bin/tests/output/hpcg_bsp1d"
		head -1 bin/tests/output/hpcg_bsp1d
		grep 'Test OK' bin/tests/output/hpcg_bsp1d
		echo " "
	fi
done

echo "*****************************************************************************************"
echo "All smoke tests done."
echo " "

