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
#EXPERIMENT can be one of PAGERANK, KNN, LABEL, or KERNEL.
#DATASET can be one of facebook_combined cit-HepTh com-amazon.ungraph com-youtube.ungraph cit-Patents com-orkut.ungraph

#Example (run kernels only): $0 KERNEL
#Example (run pagerank experiment on facebook_combined): $0 facebook_combined PAGERANK
#Example (run k-NN experiment on facebook_combined): $0 facebook_combined KNN
#Example (run all non-kernel experiments on given dataset): $0 facebook_combined
#Example (run everything): $0


DATASETTORUN=$1
EXPTYPE=$2

if [[ "${DATASETTORUN}" == "KERNEL" ]]; then
	EXPTYPE=${DATASETTORUN}
	unset DATASETTORUN
fi

DATASETS=(west0497.mtx facebook_combined.txt cit-HepTh.txt com-amazon.ungraph.txt com-youtube.ungraph.txt cit-Patents.txt com-orkut.ungraph.txt)
DATASET_MODES=(direct direct indirect indirect indirect indirect indirect)
DATASET_SIZES=(497 4039 27770 334863 1134890 3774768 3072441)
KNN4SOL=(118 499 2805 1 64 1 1048907)
KNN6SOL=(357 547 5176 1 246 1 1453447)

#which command to use to run a GraphBLAS program
LPF=yes
if [ -z "${LPFRUN}" ]; then
	echo "LPFRUN is not set; corresponding performance tests will be disabled"
	LPF=no
else
	if [ -z "${LPFRUN_PASSTHROUGH}" ]; then
		LPFRUN_PASSTHROUGH="-mpirun,"
		echo "Warning: LPFRUN_PASSTHROUGH was not set. I assumed the following: -mpirun,"
	fi
	echo "Info: Using the command \`\`${LPFRUN} -np <P> ${LPFRUN_PASSTHROUGH}<MPI arg 1> ${LPFRUN_PASSTHROUGH}<MPI arg 2> ...'' to launch GraphBLAS programs."
fi


#binding arguments to underlying MPI layer when spawning a number of processes less than or equal to the number of sockets
if [ -z "${MPI_BINDING_ARGS}" ]; then
	#assume MPICH-style syntax
	MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}-bind-to ${LPFRUN_PASSTHROUGH}socket"
	#NOTE: OpenMPI (untested!):
	#MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}--bind-to ${LPFRUN_PASSTHROUGH}socket"
	#NOTE: Intel MPI (tested)
	#MPI_BINDING_ARGS="${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN=1 ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_DOMAIN=socket ${LPFRUN_PASSTHROUGH}-genv ${LPFRUN_PASSTHROUGH}I_MPI_PIN_ORDER=spread"
	#NOTE: IBM Platform MPI (tested)
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
echo "Info: Using MPI_OVERBINDING_ARGS \`\`${MPI_OVERBINDING_ARGS}''"

if [ -z "${MPI_PASS_ENV}" ]; then
	#assume MPICH / Intel MPI syntax
	MPI_PASS_ENV=${LPFRUN_PASSTHROUGH}-genv
	#NOTE: IBM Platform MPI (tested)
	#MPI_PASS_ENV=${LPFRUN_PASSTHROUGH}-e
fi
echo "Info: Using the \`\`${MPI_PASS_ENV}'' switch to pass environment variables to the underlying MPI layer"
if [ ! -z "${DATASETTORUN}" ]; then
	echo "Info: dataset called is ${DATASETTORUN}"
fi
if [ ! -z ${EXPTYPE} ]; then
	echo "Info: experiment requested is ${EXPTYPE}"
fi

#number of sockets of machine
NUM_SOCKETS=`grep -i "physical id" /proc/cpuinfo | sort -u | wc -l`
echo "Info: Number of sockets detected is ${NUM_SOCKETS}"

rm -f bin/tests/output/benchmarks || true

echo "*****************************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "-----------------------------------------------------------------------------------------------------"


if [[ -z $DATASETTORUN && ( -z "$EXPTYPE" || "$EXPTYPE" == "KERNEL" ) ]]; then

	echo ">>>      [x]           [x]       Testing semiring axpy versus hardcoded axpy over"
	echo "                                 10 000 000 doubles"
	echo " "
	bin/tests/fma &> bin/tests/output/fma 10000000 0
	head -1 bin/tests/output/fma
	tail -2 bin/tests/output/fma
	egrep 'label|Overall timings|0,' bin/tests/output/fma | grep -v Outer >> bin/tests/output/benchmarks

	echo ">>>      [x]           [x]       Testing monoid reduce versus hardcoded reduce over"
	echo "                                 10 000 000 doubles"
	echo " "
	bin/tests/reduce &> bin/tests/output/reduce 10000000 0
	head -1 bin/tests/output/reduce
	tail -2 bin/tests/output/reduce
	egrep 'label|Overall timings|0,' bin/tests/output/reduce | grep -v Outer >> bin/tests/output/benchmarks

	echo ">>>      [x]           [x]       Testing semiring dot product versus its hardcoded variant"
	echo "                                 over 10 000 000 doubles"
	echo " "
	bin/tests/dot &> bin/tests/output/dot 10000000 0
	head -1 bin/tests/output/dot
	tail -2 bin/tests/output/dot
	egrep 'label|Overall timings|0,' bin/tests/output/dot | grep -v Outer >> bin/tests/output/benchmarks

	echo ">>>      [x]           [x]       Testing semiring axpy versus hardcoded axpy over"
	echo "                                 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	bin/tests/fma-openmp &> bin/tests/output/fma-openmp 10000000 0
	head -1 bin/tests/output/fma-openmp
	tail -2 bin/tests/output/fma-openmp
	egrep 'label|Overall timings|0,' bin/tests/output/fma-openmp | grep -v Outer >> bin/tests/output/benchmarks

	echo ">>>      [x]           [x]       Testing monoid reduce versus hardcoded reduce over"
	echo "                                 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	bin/tests/reduce-openmp &> bin/tests/output/reduce-openmp 10000000 0
	head -1 bin/tests/output/reduce-openmp
	tail -2 bin/tests/output/reduce-openmp
	egrep 'label|Overall timings|0,' bin/tests/output/reduce-openmp | grep -v Outer >> bin/tests/output/benchmarks


	echo ">>>      [x]           [x]       Testing semiring dot product versus its hardcoded variant"
	echo "                                 over 10 000 000 doubles, using the OpenMP reference backend"
	echo " "
	bin/tests/dot-openmp &> bin/tests/output/dot-openmp 10000000 0
	head -1 bin/tests/output/dot-openmp
	tail -2 bin/tests/output/dot-openmp
	egrep 'label|Overall timings|0,' bin/tests/output/dot-openmp | grep -v Outer >> bin/tests/output/benchmarks

fi

function runKNNBenchMarkTests()
{
	local kValue=$1
	local dataSet=$2
	local parseMode=$3
	local parseSize=$4
	local nbhSize=$5


	echo ">>>      [x]           [x]       Testing k-NN using ${dataSet} dataset, k=$kValue, serial."
	echo
	bin/tests/automatic_launch_knn_serial $kValue datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_knn${kValue}_serial_${dataSet}
	head -1 bin/tests/output/automatic_launch_knn${kValue}_serial_${dataSet}
	if grep -q "Neighbourhood size is ${nbhSize}." bin/tests/output/automatic_launch_knn${kValue}_serial_${dataSet}; then
		printf "Test OK.\n\n"
	else
		printf "Test FAILED.\n\n"
		# exit
	fi
	echo "serial k-hop computation for k=$kValue using the ${dataSet} dataset" >> bin/tests/output/benchmarks
	egrep 'Avg|Std' bin/tests/output/automatic_launch_knn${kValue}_serial_${dataSet} >> bin/tests/output/benchmarks
	echo >> bin/tests/output/benchmarks

	if [ "${LPF}" == "yes" ]; then
		BSP1D=(1 2 4)
		for ((bsp=0;bsp<${#BSP1D[@]};++bsp));
		do
			# number of processes
			P=${BSP1D[bsp]}

			echo ">>>      [x]           [x]       Testing k-NN using ${dataSet} dataset, k=$kValue, BSP1D P=${P}."
			echo
			${LPFRUN} -np ${P} bin/tests/automatic_launch_knn $kValue datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_knn${kValue}_${dataSet}_${P}
			head -1 bin/tests/output/automatic_launch_knn${kValue}_${dataSet}_${P}
			if grep -q "Neighbourhood size is ${nbhSize}." bin/tests/output/automatic_launch_knn${kValue}_${dataSet}_${P}; then
				printf "Test OK.\n\n"
			else
				printf "Test FAILED.\n\n"
				# exit
			fi
			echo "BSP1D P=${P} k-hop computation for k=$kValue using the ${dataSet} dataset" >> bin/tests/output/benchmarks
			egrep 'Avg|Std' bin/tests/output/automatic_launch_knn${kValue}_${dataSet}_${P} >> bin/tests/output/benchmarks
			echo >> bin/tests/output/benchmarks
		done
	fi

	echo ">>>      [x]           [x]       Testing k-NN using ${dataSet} dataset, k=$kValue, OpenMP."
	echo
	bin/tests/automatic_launch_knn_openmp $kValue datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_knn${kValue}_openmp_${dataSet}
	head -1 bin/tests/output/automatic_launch_knn${kValue}_openmp_${dataSet}
	if grep -q "Neighbourhood size is ${nbhSize}." bin/tests/output/automatic_launch_knn${kValue}_openmp_${dataSet}; then
		printf "Test OK.\n\n"
	else
		printf "Test FAILED.\n\n"
		# exit
	fi
	echo "OpenMP k-hop computation for k=$kValue using the ${dataSet} dataset" >> bin/tests/output/benchmarks
	egrep 'Avg|Std' bin/tests/output/automatic_launch_knn${kValue}_openmp_${dataSet} >> bin/tests/output/benchmarks
	echo >> bin/tests/output/benchmarks

	if [ "${LPF}" == "yes" ]; then
		HYBRID=(1 2 4)
		for ((hyb=0;hyb<${#HYBRID[@]};++hyb));
		do
			# number of processes
			P=${HYBRID[hyb]}

			echo ">>>      [x]           [x]       Testing k-NN using ${dataSet} dataset, k=$kValue, Hybrid P=${P}."
			echo
			(./set_omp.sh -q ${P} && ${LPFRUN} -np ${P} ${ACTIVE_MPI_BINDING} ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}OMP_NUM_THREADS=`./set_omp.sh ${P}` bin/tests/automatic_launch_knn_hybrid $kValue datasets/${dataSet} ${parseMode}) &> bin/tests/output/automatic_launch_knn${kValue}_hybrid_${dataSet}_${P}
			head -1 bin/tests/output/automatic_launch_knn${kValue}_hybrid_${dataSet}_${P}
			if grep -q "Neighbourhood size is ${nbhSize}." bin/tests/output/automatic_launch_knn${kValue}_hybrid_${dataSet}_${P}; then
				printf "Test OK.\n\n"
			else
				printf "Test FAILED.\n\n"
				# exit
			fi
			echo "Hybrid P=${P} k-hop computation for k=$kValue using the ${dataSet} dataset" >> bin/tests/output/benchmarks
			egrep 'Std|Avg' bin/tests/output/automatic_launch_knn${kValue}_hybrid_${dataSet}_${P} >> bin/tests/output/benchmarks
			echo >> bin/tests/output/benchmarks
		done
	fi
}

runOtherBenchMarkTests()
{
	local dataSet=$1
	local parseMode=$2
	local parseSize=$3
	local alg=$4
	local algExt=$5 # use this for now to keep results as before - should be eliminated at some point


	echo ">>>      [x]           [x]       Testing $algExt using ${dataSet} dataset, serial."
	echo
	bin/tests/automatic_launch_${alg}_serial datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_${alg}_serial_${dataSet}
	head -1 bin/tests/output/automatic_launch_${alg}_serial_${dataSet}
	if grep -q "Test OK" bin/tests/output/automatic_launch_${alg}_serial_${dataSet}; then
		printf "Test OK.\n\n"
	else
		printf "Test FAILED.\n\n"
		# exit
	fi
	echo "serial $algExt using the ${dataSet} dataset" >> bin/tests/output/benchmarks
	egrep 'Avg|Std' bin/tests/output/automatic_launch_${alg}_serial_${dataSet} >> bin/tests/output/benchmarks
	echo >> bin/tests/output/benchmarks

	if [ "${LPF}" == "yes" ]; then
		BSP1D=(1 2 4)
		for ((bsp=0;bsp<${#BSP1D[@]};++bsp));
		do
			# number of processes
			P=${BSP1D[bsp]}

			echo ">>>      [x]           [x]       Testing $algExt using ${dataSet} dataset, BSP1D P=${P}."
			echo
			${LPFRUN} -np ${P} bin/tests/automatic_launch_${alg} datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_${alg}_${dataSet}_${P}
			head -1 bin/tests/output/automatic_launch_${alg}_${dataSet}_${P}
			if grep -q "Test OK" bin/tests/output/automatic_launch_${alg}_${dataSet}_${P}; then
				printf "Test OK.\n\n"
			else
				printf "Test FAILED.\n\n"
				# exit
			fi
			echo "BSP1D P=${P} $algExt using the ${dataSet} dataset" >> bin/tests/output/benchmarks
			egrep 'Std|Avg' bin/tests/output/automatic_launch_${alg}_${dataSet}_${P} >> bin/tests/output/benchmarks
			echo >> bin/tests/output/benchmarks
		done
	fi

	echo ">>>      [x]           [x]       Testing $algExt using ${dataSet} dataset, OpenMP."
	echo
	bin/tests/automatic_launch_${alg}_openmp datasets/${dataSet} ${parseMode} &> bin/tests/output/automatic_launch_${alg}_openmp_${dataSet}
	head -1 bin/tests/output/automatic_launch_${alg}_openmp_${dataSet}
	if grep -q "Test OK" bin/tests/output/automatic_launch_${alg}_openmp_${dataSet}; then
		printf "Test OK.\n\n"
	else
		printf "Test FAILED.\n\n"
		# exit
	fi
	echo "OpenMP $algExt using the ${dataSet} dataset" >> bin/tests/output/benchmarks
	egrep 'Std|Avg' bin/tests/output/automatic_launch_${alg}_openmp_${dataSet} >> bin/tests/output/benchmarks
	echo >> bin/tests/output/benchmarks

	if [ "${LPF}" == "yes" ]; then
		HYBRID=(1 2 4)
		for ((hyb=0;hyb<${#HYBRID[@]};++hyb));
		do
			# number of processes
			P=${HYBRID[hyb]}

			echo ">>>      [x]           [x]       Testing $algExt using ${dataSet} dataset, Hybrid P=${P}."
			echo
			if [ ${P} -gt ${NUM_SOCKETS} ]; then
				ACTIVE_MPI_BINDING=${MPI_OVERBINDING_ARGS}
			else
				ACTIVE_MPI_BINDING=${MPI_BINDING_ARGS}
			fi
			(./set_omp.sh -q ${P} && ${LPFRUN} -np ${P} ${ACTIVE_MPI_BINDING} ${MPI_PASS_ENV} ${LPFRUN_PASSTHROUGH}OMP_NUM_THREADS=`./set_omp.sh ${P}` bin/tests/automatic_launch_${alg}_hybrid datasets/${dataSet} ${parseMode}) &> bin/tests/output/automatic_launch_${alg}_hybrid_${dataSet}_${P}
			head -1 bin/tests/output/automatic_launch_${alg}_hybrid_${dataSet}_${P}
			if grep -q "Test OK" bin/tests/output/automatic_launch_${alg}_hybrid_${dataSet}_${P}; then
				printf "Test OK.\n\n"
			else
				printf "Test FAILED.\n\n"
				# exit
			fi
			echo "Hybrid P=${P} $algExt using the ${dataSet} dataset" >> bin/tests/output/benchmarks
			egrep 'Avg|Std' bin/tests/output/automatic_launch_${alg}_hybrid_${dataSet}_${P} >> bin/tests/output/benchmarks
			echo >> bin/tests/output/benchmarks
		done
	fi
}


if [ -z "$EXPTYPE" ] || ! [ "$EXPTYPE" == "KERNEL" ]; then

	for ((i=0;i<${#DATASETS[@]};++i));
	do


		if [ ! -z "$DATASETTORUN" ] && [ "$DATASETTORUN" != "${DATASETS[i]}" ]; then
			continue
		fi

		# initialise parameters
		DATASET=${DATASETS[i]}
		PARSE_MODE=${DATASET_MODES[i]}
		PARSE_SIZE=${DATASET_SIZES[i]}
		KNN4SOL=${KNN4SOL[i]}
		KNN6SOL=${KNN6SOL[i]}

		# test for file
		if [ ! -f datasets/${DATASET} ]; then
			echo "Warning: dataset/${DATASET} not found. Provide the dataset to enable performance tests with it."
			continue
		fi

		if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "KNN" ]; then

			# ---------------------------------------------------------------------
			# k-NN k=4
			runKNNBenchMarkTests 4 "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "$KNN4SOL"

			# ---------------------------------------------------------------------
			# k-NN k=6
			runKNNBenchMarkTests 6 "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "$KNN6SOL"

		fi
		if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "LABEL" ]; then

			# ---------------------------------------------------------------------
			# label propagation
			runOtherBenchMarkTests "$DATASET" "$PARSE_MODE" "$PARSE_SIZE" "label" "label propagation"

		fi
		if [ -z "$EXPTYPE" ] || [ "$EXPTYPE" == "PAGERANK" ]; then

			# ---------------------------------------------------------------------
			# pagerank
			runOtherBenchMarkTests "$DATASET" "$PARSE_MODE" 0 "pagerank" "pagerank"

		fi

	done

fi

