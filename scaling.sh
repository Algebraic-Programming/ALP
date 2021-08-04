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

LPF=yes
if [ -z "${LPFRUN}" ]; then
	echo "LPFRUN is not set; corresponding performance tests will be disabled"
	LPF=no
else
	echo "Info: LPFRUN is set to ${LPFRUN}"
fi

PATH=`pwd`/deps/bin/:${PATH}
if [ -z "$1" ]; then
	DATASETS=( 1000 1000000 10000000 )
else
	DATASETS=( $1 )
fi
TESTS=(1 2 3 4)

export HYDRA_BINDING=socket

rm -f bin/tests/output/scaling || true
echo "*****************************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE  DESCRIPTION  (square matrix size, kernel to be benchmarked, backend) "
echo "-----------------------------------------------------------------------------------------------------"

for ((d=0;d<${#DATASETS[@]};++d));
do

	# initialise parameters
	DATASET=${DATASETS[d]}

	for ((t=0;t<${#TESTS[@]};++t));
	do

		# initialise parameters
		TEST=${TESTS[t]}

		# ---------------------------------------------------------------------
		# serial

		echo ">>>      [x]           [x]       Benchmark (${DATASET},${TEST},serial)."
		echo
		bin/tests/automatic_launch_scaling_serial ${DATASET} ${TEST} 0 &> bin/tests/output/automatic_launch_scaling_serial_${DATASET}_${TEST}
		head -1 bin/tests/output/automatic_launch_scaling_serial_${DATASET}_${TEST}
		echo "serial scaling using the ${DATASET} dataset and test ${TEST}" >> bin/tests/output/scaling
		grep -A4 'Overall timings' bin/tests/output/automatic_launch_scaling_serial_${DATASET}_${TEST} | egrep -i 'overall|avg|max|std' >> bin/tests/output/scaling
		tail -2 bin/tests/output/automatic_launch_scaling_serial_${DATASET}_${TEST} | tee -a bin/tests/output/scaling

		# ---------------------------------------------------------------------
		# BSP1D

		if [ "${LPF}" == "yes" ]; then
			BSP1D=(1 2 4)
			for ((bsp=0;bsp<${#BSP1D[@]};++bsp));
			do
				# number of processes
				P=${BSP1D[bsp]}

				echo ">>>      [x]           [x]       Benchmark (${DATASET},${TEST},BSP1D P=${P})."
				echo
				${LPFRUN} -np ${P} -probe 5 bin/tests/automatic_launch_scaling ${DATASET} ${TEST} 0 &> bin/tests/output/automatic_launch_scaling_${DATASET}_${P}_${TEST}
				head -1 bin/tests/output/automatic_launch_scaling_${DATASET}_${P}_${TEST}
				echo "BSP1D P=${P} scaling using the ${DATASET} dataset and test ${TEST}" >> bin/tests/output/scaling
				grep -A4 'Overall timings' bin/tests/output/automatic_launch_scaling_${DATASET}_${P}_${TEST} | egrep -i 'overall|avg|max|std' >> bin/tests/output/scaling
				tail -2 bin/tests/output/automatic_launch_scaling_${DATASET}_${P}_${TEST} | tee -a bin/tests/output/scaling
			done
		fi

		# ---------------------------------------------------------------------
		# OpenMP

		echo ">>>      [x]           [x]       Benchmark (${DATASET},${TEST},OpenMP)."
		echo
		bin/tests/automatic_launch_scaling_openmp ${DATASET} ${TEST} 0 &> bin/tests/output/automatic_launch_scaling_openmp_${DATASET}_${TEST}
		head -1 bin/tests/output/automatic_launch_scaling_openmp_${DATASET}_${TEST}
		echo "OpenMP scaling using the ${DATASET} dataset and test ${TEST}" >> bin/tests/output/scaling
		grep -A4 'Overall timings' bin/tests/output/automatic_launch_scaling_openmp_${DATASET}_${TEST} | egrep -i 'overall|avg|max|std' >> bin/tests/output/scaling
		tail -2 bin/tests/output/automatic_launch_scaling_openmp_${DATASET}_${TEST} | tee -a bin/tests/output/scaling

		# ---------------------------------------------------------------------
		# Hybrid

		if [ "${LPF}" == "yes" ]; then
			HYBRID=(1 2 4)
			for ((hyb=0;hyb<${#HYBRID[@]};++hyb));
			do
				# number of processes
				P=${HYBRID[hyb]}

				echo ">>>      [x]           [x]       Benchmark (${DATASET},${TEST},Hybrid BSP1D P=${P} + OpenMP)."
				echo
				(./set_omp.sh -q ${P} && ${LPFRUN} -np ${P} -probe 5 -mpirun,-genv -mpirun,OMP_NUM_THREADS=`./set_omp.sh ${P}` bin/tests/automatic_launch_scaling_hybrid ${DATASET} ${TEST} 0) &> bin/tests/output/automatic_launch_scaling_hybrid_${DATASET}_${P}_${TEST}
				head -1 bin/tests/output/automatic_launch_scaling_hybrid_${DATASET}_${P}_${TEST}
				echo "Hybrid P=${P} scaling using the ${DATASET} dataset and test ${TEST}" >> bin/tests/output/scaling
				grep -A4 'Overall timings' bin/tests/output/automatic_launch_scaling_hybrid_${DATASET}_${P}_${TEST} | egrep -i 'overall|avg|max|std' >> bin/tests/output/scaling
				tail -2 bin/tests/output/automatic_launch_scaling_hybrid_${DATASET}_${P}_${TEST} | tee -a bin/tests/output/scaling
			done
		fi
	done
done

