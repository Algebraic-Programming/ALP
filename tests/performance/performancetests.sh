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

#Usage: $0 (DATASET) (EXPERIMENT)
#Note that all arguments are optional. They select a subset of performance tests only.
#EXPERIMENT can be one of PAGERANK, KNN, LABEL, KERNEL, or SCALING.
#DATASET can be one of facebook_combined cit-HepTh com-amazon.ungraph com-youtube.ungraph cit-Patents com-orkut.ungraph

#Example (run everything): $0
#Example (run kernels only): $0 KERNEL
#Example (run scaling tests only): $0 SCALING
#Example (run pagerank experiment on facebook_combined): $0 facebook_combined PAGERANK
#Example (run k-NN experiment on facebook_combined): $0 facebook_combined KNN
#Example (run all non-kernel experiments on given dataset): $0 facebook_combined

TESTS_ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/../ &> /dev/null && pwd )"
source ${TESTS_ROOT_DIR}/parse_env.sh

DATASETTORUN=$1
EXPTYPE=$2
echo "Info: script called with the following arguments: ${DATASETTORUN} ${EXPTYPE}"

#number of sockets of machine
if [[ -z ${NUM_SOCKETS} ]]; then
	NUM_SOCKETS=`grep -i "physical id" /proc/cpuinfo | sort -u | wc -l`
	echo "Info: number of sockets detected is ${NUM_SOCKETS}"
fi
if [ "${NUM_SOCKETS}" -eq "0" ]; then
	echo "Warning: failed to auto-detect the number of sockets, assuming 1;"
	echo "         if incorrect, please set NUM_SOCKETS manually."
	NUM_SOCKETS=1
fi
echo "Info: selected number of sockets is ${NUM_SOCKETS}"

if [[ -z ${MAX_PROCESSES} ]]; then
	echo "Info: MAX_PROCESSES was not set. Will set it equal to the number of sockets."
	MAX_PROCESSES=${NUM_SOCKETS}
fi
echo "Info: maximum number of processes is ${MAX_PROCESSES}"

if [[ "${DATASETTORUN}" == "KERNEL" ]]; then
	EXPTYPE=${DATASETTORUN}
	unset DATASETTORUN
elif [[ "${DATASETTORUN}" == "SCALING" ]]; then
	EXPTYPE=${DATASETTORUN}
	unset DATASETTORUN
else
	echo "Info: selected dataset ${DATASETTORUN}"
fi

if [ "x${EXPTYPE}" != "x" ]; then
	echo "Info: selected experiment: ${EXPTYPE}"
fi

DATASETS=(west0497.mtx facebook_combined.txt cit-HepTh.txt com-amazon.ungraph.txt com-youtube.ungraph.txt cit-Patents.txt com-orkut.ungraph.txt)
DATASET_MODES=(direct direct indirect indirect indirect indirect indirect)
DATASET_SIZES=(497 4039 27770 334863 1134890 3774768 3072441)
KNN4SOLS=(59 421 1138 1 32 1 609122)
KNN6SOLS=(238 526 4189 1 181 1 1268035)

#the following datasets are used for benchmarking SpMV, SpMSpV, and SpMSpM
MULTIPLICATION_DATASETS=(west0497.mtx fidap037.mtx cavity17.mtx s3rmt3m3.mtx bloweybq.mtx bcsstk17.mtx Pres_Poisson.mtx gyro_m.mtx memplus.mtx lhr34.mtx bcsstk32.mtx vanbody.mtx s3dkt3m2.mtx G2_circuit.mtx Stanford.mtx coPapersCiteseer.mtx bundle_adj.mtx Stanford_Berkeley.mtx apache2.mtx Emilia_923.mtx ldoor.mtx ecology2.mtx Serena.mtx cage14.mtx G3_circuit.mtx wikipedia-20051105.mtx wikipedia-20061104.mtx Freescale1.mtx wikipedia-20070206.mtx Queen_4147.mtx cage15.mtx adaptive.mtx rgg_n_2_24_s0.mtx uk-2002.mtx road_usa.mtx MOLIERE_2016.mtx europe_osm.mtx twitter.mtx com-Friendster.mtx)

#the following datasets are used for benchmarking SpMV, SpMSpV, and SpMSpM
MULTIPLICATION_DATASETS=(west0497.mtx fidap037.mtx cavity17.mtx s3rmt3m3.mtx bloweybq.mtx bcsstk17.mtx Pres_Poisson.mtx gyro_m.mtx memplus.mtx lhr34.mtx bcsstk32.mtx vanbody.mtx s3dkt3m2.mtx G2_circuit.mtx Stanford.mtx coPapersCiteseer.mtx bundle_adj.mtx Stanford_Berkeley.mtx apache2.mtx Emilia_923.mtx ldoor.mtx ecology2.mtx Serena.mtx cage14.mtx G3_circuit.mtx wikipedia-20051105.mtx wikipedia-20061104.mtx Freescale1.mtx wikipedia-20070206.mtx Queen_4147.mtx cage15.mtx adaptive.mtx rgg_n_2_24_s0.mtx uk-2002.mtx road_usa.mtx MOLIERE_2016.mtx europe_osm.mtx twitter.mtx com-Friendster.mtx)

#which command to use to run a GraphBLAS program
LPF=yes
if [ -z "${LPFRUN}" ]; then
	echo "LPFRUN is not set; corresponding performance tests will be disabled"
	LPF=no
else
	if [ -z "${LPFRUN_PASSTHROUGH}" ]; then
		echo "Error: LPFRUN was set, but LPFRUN_PASSTHROUGH was not"
		exit 255;
	fi
	if [ -z "${MPI_PASS_ENV}" ]; then
		echo "Error: LPFRUN was set, but MPI_PASS_ENV was not"
		exit 255;
	fi
fi

#binding arguments to underlying MPI layer when spawning a number of processes less than or equal to the number of sockets
if [ -z "${MPI_BINDING_ARGS}" ]; then
	#assume MPICH-style syntax
	MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}-bind-to ${LPFRUN_PASSTHROUGH}socket"
	#NOTE: OpenMPI
	#MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}--map-by ${LPFRUN_PASSTHROUGH}socket ${LPFRUN_PASSTHROUGH}--bind-to ${LPFRUN_PASSTHROUGH}socket"
	#NOTE: Intel MPI
	#MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN=1 ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_DOMAIN=socket ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_ORDER=spread"
	#NOTE: IBM Platform MPI
	#MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}-affcycle=socket ${LPFRUN_PASSTHROUGH}-affwidth=socket"
fi
echo "Info: Using MPI_BINDING_ARGS \`\`${MPI_BINDING_ARGS}''"

#binding arguments to underlying MPI layer when spawning multiple processes per socket
if [ -z "${MPI_OVERBINDING_ARGS}" ]; then
	#Assume equal to MPI_BINDING_ARGS
	MPI_OVERBINDING_ARGS=${MPI_BINDING_ARGS}
	#NOTE: Intel MPI (tested)
	#MPI_OVERBINDING_ARGS="${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN=1 ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_ORDER=spread"
fi
printf "Info: Using MPI_OVERBINDING_ARGS \`\`${MPI_OVERBINDING_ARGS}''. "
printf "The use of these bindings over MPI_BINDING_ARGS is triggered manually by defining USE_MPI_OVERBINDING, "
if [ -z ${USE_MPI_OVERBINDING+x} ]; then
	printf "which is currently NOT defined.\n"
else
	printf "which IS currently defined.\n"
fi

if [ ! -z "${DATASETTORUN}" ]; then
	echo "Info: dataset called is ${DATASETTORUN}"
fi
if [ ! -z ${EXPTYPE} ]; then
	echo "Info: experiment requested is ${EXPTYPE}"
fi

if [ -f "${TEST_OUT_DIR}/benchmarks" ]; then
	echo "Warning: old benchmark summaries are deleted"
	rm -f ${TEST_OUT_DIR}/benchmarks || true
fi

if [ -f "${TEST_OUT_DIR}/scaling" ]; then
	echo "Warning: old scaling summaries are deleted"
	rm -f ${TEST_OUT_DIR}/scaling || true
fi

echo " "
echo "*****************************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "-----------------------------------------------------------------------------------------------------"
echo " "

# kernel performance tests

if [[ -z $DATASETTORUN && ( -z "$EXPTYPE" || "$EXPTYPE" == "KERNEL" ) ]]; then

	echo ">>>      [ ]           [x]       Testing semiring axpy versus hardcoded axpy over"
	echo "                                 10 000 000 doubles"
	echo " "
	${TEST_BIN_DIR}/fma &> ${TEST_OUT_DIR}/fma 10000000 0
	head -1 ${TEST_OUT_DIR}/fma
	tail -2 ${TEST_OUT_DIR}/fma
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/fma | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks

	echo ">>>      [ ]           [x]       Testing monoid reduce versus hardcoded reduce over"
	echo "                                 10 000 000 doubles"
	echo " "
	${TEST_BIN_DIR}/reduce &> ${TEST_OUT_DIR}/reduce 10000000 0
	head -1 ${TEST_OUT_DIR}/reduce
	tail -2 ${TEST_OUT_DIR}/reduce
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/reduce | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks

	echo ">>>      [ ]           [x]       Testing semiring dot product versus its hardcoded variant"
	echo "                                 over 10 000 000 doubles"
	echo " "
	${TEST_BIN_DIR}/dot &> ${TEST_OUT_DIR}/dot 10000000 0
	head -1 ${TEST_OUT_DIR}/dot
	tail -2 ${TEST_OUT_DIR}/dot
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/dot | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks

	echo ">>>      [ ]           [x]       Testing semiring axpy versus hardcoded axpy over"
	echo "                                 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	${TEST_BIN_DIR}/fma-openmp &> ${TEST_OUT_DIR}/fma-openmp 10000000 0
	head -1 ${TEST_OUT_DIR}/fma-openmp
	tail -2 ${TEST_OUT_DIR}/fma-openmp
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/fma-openmp | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks

	echo ">>>      [ ]           [x]       Testing monoid reduce versus hardcoded reduce over"
	echo "                                 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	${TEST_BIN_DIR}/reduce-openmp &> ${TEST_OUT_DIR}/reduce-openmp 10000000 0
	head -1 ${TEST_OUT_DIR}/reduce-openmp
	tail -2 ${TEST_OUT_DIR}/reduce-openmp
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/reduce-openmp | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks


	echo ">>>      [ ]           [x]       Testing semiring dot product versus its hardcoded variant"
	echo "                                 over 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	${TEST_BIN_DIR}/dot-openmp &> ${TEST_OUT_DIR}/dot-openmp 10000000 0
	head -1 ${TEST_OUT_DIR}/dot-openmp
	tail -2 ${TEST_OUT_DIR}/dot-openmp
	egrep 'label|Overall timings|0,' ${TEST_OUT_DIR}/dot-openmp | grep -v Outer >> ${TEST_OUT_DIR}/benchmarks

fi

# start definition of helper functions for remainder performance tests

function runScalingTest()
{
	local runner=$1
	local backend=$2
	if [ "${backend}" = "hyperdags" ]; then
		local DATASETS=( 1000 )
	else
		local DATASETS=( 1000 1000000 10000000 )
	fi
	local TESTS=(1 2 3 4)

	for ((d=0;d<${#DATASETS[@]};++d));
	do
		local DATASET=${DATASETS[d]}
		for ((t=0;t<${#TESTS[@]};++t));
		do
			local TEST=${TESTS[t]}

			echo ">>>      [ ]           [x]       Benchmark level-2 kernel ${TEST} on matrices of size ${DATASET}"
		        echo "                                 to gauge the weak scaling behaviour of the ${backend} backend."
			echo
			$runner ${TEST_BIN_DIR}/scaling_${backend} ${DATASET} ${TEST} 0 &> ${TEST_OUT_DIR}/scaling_${backend}_${DATASET}_${TEST}.log
			head -1 ${TEST_OUT_DIR}/scaling_${backend}_${DATASET}_${TEST}.log
			echo "$backend scaling on size $DATASET and test ${TEST}" >> ${TEST_OUT_DIR}/scaling
			grep -A4 'Overall timings' ${TEST_OUT_DIR}/scaling_${backend}_${DATASET}_${TEST}.log >> ${TEST_OUT_DIR}/scaling
			tail -2 ${TEST_OUT_DIR}/scaling_${backend}_${DATASET}_${TEST}.log | tee -a ${TEST_OUT_DIR}/scaling
		done
	done
}

function runKNNBenchMarkTests()
{
	local runner=$1
	local backend=$2
	local kValue=$3
	local dataSet=$4
	local parseMode=$5
	local parseSize=$6
	local nbhSize=$7

	echo ">>>      [x]           [x]       Testing k-NN using ${dataSet} dataset, k=$kValue,"
	echo "                                 $backend backend. Also verifies the neighbourhood"
	echo "                                 size with a ground truth value."
	echo
	$runner ${TEST_BIN_DIR}/driver_knn_${backend} $kValue ${INPUT_DIR}/${dataSet} ${parseMode} &> ${TEST_OUT_DIR}/driver_${kValue}nn_${backend}_${dataSet}.log
	head -1 ${TEST_OUT_DIR}/driver_${kValue}nn_${backend}_${dataSet}.log
	if grep -q "Neighbourhood size is ${nbhSize}" ${TEST_OUT_DIR}/driver_${kValue}nn_${backend}_${dataSet}.log; then
		printf "Test OK\n\n"
		echo "$backend k-hop computation for k=$kValue using the ${dataSet} dataset" >> ${TEST_OUT_DIR}/benchmarks
		egrep 'Avg|Std' ${TEST_OUT_DIR}/driver_${kValue}nn_${backend}_${dataSet}.log >> ${TEST_OUT_DIR}/benchmarks
		echo >> ${TEST_OUT_DIR}/benchmarks
	else
		printf "Test FAILED\n\n"
	fi
}

runOtherBenchMarkTests()
{
	local runner=$1
	local backend=$2
	local dataSet=$3
	local parseMode=$4
	local parseSize=$5
	local alg=$6


	echo ">>>      [ ]           [x]       Testing $alg using ${dataSet} dataset, $backend backend."
	echo
	$runner ${TEST_BIN_DIR}/driver_${alg}_${backend} ${INPUT_DIR}/${dataSet} ${parseMode} &> ${TEST_OUT_DIR}/driver_${alg}_${backend}_${dataSet}
	head -1 ${TEST_OUT_DIR}/driver_${alg}_${backend}_${dataSet}
	if grep -q "Test OK" ${TEST_OUT_DIR}/driver_${alg}_${backend}_${dataSet}; then
		printf "Test OK\n\n"
	else
		printf "Test FAILED\n\n"
	fi
	echo "$backend $alg using the ${dataSet} dataset" >> ${TEST_OUT_DIR}/benchmarks
	egrep 'Avg|Std' ${TEST_OUT_DIR}/driver_${alg}_${backend}_${dataSet} >> ${TEST_OUT_DIR}/benchmarks
	echo >> ${TEST_OUT_DIR}/benchmarks
}

runMultiplicationKernels()
{
	local runner=$1
	local backend=$2
	local dataSet=$3
	local parseMode=$4
	local i=$5

	# the check for the matrices existence is assumed to have already passed

	if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "SPMV" ]; then

		# ---------------------------------------------------------------------
		# spmv
		echo ">>>      [ ]           [x]       Testing spmv using ${dataSet} dataset, $backend backend."
		echo
		$runner ${TEST_BIN_DIR}/driver_spmv_${backend} ${INPUT_DIR}/${dataSet} ${parseMode} &> ${TEST_OUT_DIR}/driver_spmv_${backend}_${dataSet}
		head -1 ${TEST_OUT_DIR}/driver_spmv_${backend}_${dataSet}
		if grep -q "Test OK" ${TEST_OUT_DIR}/driver_spmv_${backend}_${dataSet}; then
			printf "Test OK\n\n"
		else
			printf "Test FAILED\n\n"
		fi
		echo "$backend spmv using the ${dataSet} dataset" >> ${TEST_OUT_DIR}/benchmarks
		egrep 'Avg|Std' ${TEST_OUT_DIR}/driver_spmv_${backend}_${dataSet} >> ${TEST_OUT_DIR}/benchmarks
		echo >> ${TEST_OUT_DIR}/benchmarks

	fi

	if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "SPMSPV" ]; then

		# ---------------------------------------------------------------------
		# spmspv
		echo ">>>      [ ]           [x]       Testing spmspv using ${dataSet} dataset, $backend backend."
		echo
		$runner ${TEST_BIN_DIR}/driver_spmspv_${backend} ${INPUT_DIR}/${dataSet} ${parseMode} &> ${TEST_OUT_DIR}/driver_spmspv_${backend}_${dataSet}
		head -1 ${TEST_OUT_DIR}/driver_spmspv_${backend}_${dataSet}
		if grep -q "Test OK" ${TEST_OUT_DIR}/driver_spmspv_${backend}_${dataSet}; then
			printf "Test OK\n\n"
		else
			printf "Test FAILED\n\n"
		fi
		echo "$backend spmspv using the ${dataSet} dataset" >> ${TEST_OUT_DIR}/benchmarks
		egrep 'Avg|Std' ${TEST_OUT_DIR}/driver_spmspv_${backend}_${dataSet} >> ${TEST_OUT_DIR}/benchmarks
		echo >> ${TEST_OUT_DIR}/benchmarks

	fi

	if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "SPMSPM" ]; then

		# ---------------------------------------------------------------------
		# spmspm
		echo ">>>      [ ]           [x]       Testing spmspm using ${dataSet} dataset, $backend backend."
		echo
		if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
			echo "Test DISABLED: no sparse level-3 operations recommended for 1D distributions."
			echo " "
		elif [ "$i" -gt "14" ]; then
			echo "Tests DISABLED: by default, long-running sparse matrix--sparse matrix multiplications are disabled (skipping dataset ${dataSet})."
			echo " "
		else
			$runner ${TEST_BIN_DIR}/driver_spmspm_${backend} ${INPUT_DIR}/${dataSet} ${INPUT_DIR}/${dataSet} ${parseMode} &> ${TEST_OUT_DIR}/driver_spmspm_${backend}_${dataSet}
			head -1 ${TEST_OUT_DIR}/driver_spmspm_${backend}_${dataSet}
			if grep -q "Test OK" ${TEST_OUT_DIR}/driver_spmspm_${backend}_${dataSet}; then
				printf "Test OK\n\n"
			else
				printf "Test FAILED\n\n"
			fi
			echo "$backend spmspm using the ${dataSet} dataset" >> ${TEST_OUT_DIR}/benchmarks
			egrep 'Avg|Std' ${TEST_OUT_DIR}/driver_spmspm_${backend}_${dataSet} >> ${TEST_OUT_DIR}/benchmarks
			echo >> ${TEST_OUT_DIR}/benchmarks
		fi
	fi
}

# end helper functions

if [ -z "$EXPTYPE" ] || ! [ "$EXPTYPE" == "KERNEL" ]; then

	for BACKEND in ${BACKENDS[@]}; do

		if [ "$BACKEND" = "hybrid" ]; then
			P=${MAX_PROCESSES}
		fi
		if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
			if [ -z "${LPFRUN}" ]; then
				echo "LPFRUN is not set!"
				exit 255;
			fi
		else
			P=1 # note, also for BSP1D to check for performance loss vs. reference
			    # BSP1D otherwise is never used for a performance test; hybrid(1D)
			    # should be used instead.
		fi
		if [ "$BACKEND" = "reference_omp" ] ; then
			T=${MAX_THREADS}
		elif [ "$BACKEND" = "hybrid" ]; then
			T=$((MAX_THREADS/NUM_SOCKETS))
			echo "Warning: assuming each socket will run its own user process."
			echo "         MAX_PROCESSES = ${MAX_PROCESSES}, NUM_SOCKETS = ${NUM_SOCKETS}"
			echo "         MAX_THREADS = ${MAX_THREADS}, P = ${P}, T = ${T}"
		else
			T=1
		fi
		if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
			runner="${LPFRUN} -n ${P}"
		else
			runner=
		fi
		if [ "$BACKEND" = "hybrid" ]; then
			if [ -z ${USE_MPI_OVERBINDING} ]; then
				runner="${runner} ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}OMP_NUM_THREADS=${T} ${MPI_OVERBINDING_ARGS}"
			else
				runner="${runner} ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}OMP_NUM_THREADS=${T} ${MPI_BINDING_ARGS}"
			fi
		fi
		if [ "$BACKEND" = "reference_omp" ]; then
			export OMP_NUM_THREADS=${T}
		fi

		echo "#######################################################################"
		echo "# Starting standardised performance tests for the ${BACKEND} backend"
		echo "#   using ${P} user processes"
		echo "#   using ${T} threads"
		if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
			echo "#   using \`\`${LPFRUN}'' for automatic launchers"
		fi
		if [ "x${runner}" != "x" ]; then
			echo "#   using runner \`\`$runner''"
		fi
		echo "#######################################################################"
		echo " "

		# scaling performance tests
		if [[ -z $DATASETTORUN && ( -z "$EXPTYPE" || "$EXPTYPE" == "SCALING" ) ]]; then
			runScalingTest "$runner" "${BACKEND}"
		fi

		for ((i=0;i<${#DATASETS[@]};++i));
		do
			if [ "$BACKEND" = "hyperdags" ] && [ "$i" -gt "0" ]; then
				echo "Info: hyperdags performance tests run only on the smallest dataset"
				echo " "
				break
			fi
			if [ ! -z "$DATASETTORUN" ] && [ "$DATASETTORUN" != "${DATASETS[i]}" ]; then
				continue
			fi

			# initialise parameters
			DATASET=${DATASETS[i]}
			PARSE_MODE=${DATASET_MODES[i]}
			PARSE_SIZE=${DATASET_SIZES[i]}
			KNN4SOL=${KNN4SOLS[i]}
			KNN6SOL=${KNN6SOLS[i]}

			# test for file
			if [ ! -f ${INPUT_DIR}/${DATASET} ]; then
				echo ">>>      [x]           [x]       Test algorithms using ${DATASET} dataset, ${BACKEND} backend."
				echo "Tests DISABLED: dataset/${DATASET} not found. Provide the dataset to enable performance tests with it."
				echo " "
				continue
			fi

			if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "KNN" ]; then

				# ---------------------------------------------------------------------
				# k-NN k=4
				runKNNBenchMarkTests "$runner" "$BACKEND" 4 "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "$KNN4SOL"

				if [ "$BACKEND" = "hyperdags" ]; then
					echo "Info: 6-NN is skipped for the hyperdags backend"
					echo " "
				else
					# ---------------------------------------------------------------------
					# k-NN k=6
					runKNNBenchMarkTests "$runner" "$BACKEND" 6 "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "$KNN6SOL"
				fi
			fi
			if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "LABEL" ]; then

				# ---------------------------------------------------------------------
				# label propagation
				runOtherBenchMarkTests "$runner" "$BACKEND" "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "label"

			fi
			if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "PAGERANK" ]; then

				# ---------------------------------------------------------------------
				# pagerank
				runOtherBenchMarkTests "$runner" "$BACKEND" "$DATASET" "$PARSE_MODE" 0 "simple_pagerank"

			fi
		done

		for ((i=0;i<${#MULTIPLICATION_DATASETS[@]};++i));
		do
			if [ ! -z "$DATASETTORUN" ] && [ "$DATASETTORUN" != "${MULTIPLICATION_DATASETS[i]}" ]; then
				continue
			fi

			if [ "$BACKEND" = "hyperdags" ] && [ "$i" -gt "0" ]; then
				echo "Info: hyperdags performance tests run only on the smallest dataset"
				echo " "
				break
			fi

			# initialise parameters
			DATASET=${MULTIPLICATION_DATASETS[i]}
			PARSE_MODE=direct
			
			# test for file
			if [ ! -f ${INPUT_DIR}/${DATASET} ]; then
				echo ">>>      [ ]           [x]       Test multiplication kernels using ${DATASET} dataset,"
				echo "                                 ${BACKEND} backend."
				echo "Tests DISABLED: dataset/${DATASET} not found. Provide the dataset to enable performance tests with it."
				echo " "
				continue
			fi

			runMultiplicationKernels "$runner" "$BACKEND" "$DATASET" "$PARSE_MODE" "$i"

		done

	done

fi

echo "*****************************************************************************************"
echo "All benchmark tests done; see ${TEST_OUT_DIR}/benchmarks."
if [[ -z $DATASETTORUN && ( -z "$EXPTYPE" || "$EXPTYPE" == "SCALING" ) ]]; then
	echo "All scaling tests done; see ${TEST_OUT_DIR}/scaling."
fi
echo " "

