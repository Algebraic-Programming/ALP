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

if [ -z "${LPFRUN}" ]; then
	echo "These tests require LPF but LPFRUN environment variable was not set -- skipping tests."
	exit 0;
fi

echo "Info: LPFRUN is set to ${LPFRUN}"

DATASET_SIZES=(256 8 32 4096)
RESULTS=(9 4 6 13)

echo "*****************************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "-----------------------------------------------------------------------------------------------------"

for ((i=0;i<${#DATASET_SIZES[@]};++i));
do

	# initialise parameters
	PARSE_SIZE=${DATASET_SIZES[i]}
	DATASET=$PARSE_SIZE
	RESULT=${RESULTS[i]}

	# ---------------------------------------------------------------------
	# label propagation

	echo ">>>      [x]           [ ]       Testing label propagation using ${DATASET} dataset, serial."
	echo
	bin/tests/automatic_launch_label_serial_test ${PARSE_SIZE} &> bin/tests/output/automatic_launch_label_serial_test_${DATASET}
	head -1 bin/tests/output/automatic_launch_label_serial_test_${DATASET}
	(grep -q "${RESULT} total iterations" bin/tests/output/automatic_launch_label_serial_test_${DATASET} && grep "Test OK." bin/tests/output/automatic_launch_label_serial_test_${DATASET}) || (printf "Test FAILED.\n")
	echo

	BSP1D=(1 2 4)
	for ((bsp=0;bsp<${#BSP1D[@]};++bsp));
	do
		# number of processes
		P=${BSP1D[bsp]}

		echo ">>>      [x]           [ ]       Testing label propagation using ${DATASET} dataset, BSP1D P=${P}."
		echo
		${LPFRUN} -np ${P} -probe 5 bin/tests/automatic_launch_label_test ${PARSE_SIZE} &> bin/tests/output/automatic_launch_label_test_${DATASET}_${P}
		head -1 bin/tests/output/automatic_launch_label_test_${DATASET}_${P}
		(grep -q "${RESULT} total iterations" bin/tests/output/automatic_launch_label_test_${DATASET}_${P} && grep "Test OK." bin/tests/output/automatic_launch_label_test_${DATASET}_${P}) || (printf "Test FAILED.\n")
		echo
	done

	echo ">>>      [x]           [ ]       Testing label propagation using ${DATASET} dataset, OpenMP."
	echo
	bin/tests/automatic_launch_label_openmp_test ${PARSE_SIZE} &> bin/tests/output/automatic_launch_label_openmp_test_${DATASET}
	head -1 bin/tests/output/automatic_launch_label_openmp_test_${DATASET}
	(grep -q "${RESULT} total iterations" bin/tests/output/automatic_launch_label_openmp_test_${DATASET} && grep "Test OK." bin/tests/output/automatic_launch_label_openmp_test_${DATASET}) || (printf "Test FAILED.\n")
	echo

	HYBRID=(1 2 4)
	for ((hyb=0;hyb<${#HYBRID[@]};++hyb));
	do
		# number of processes
		P=${HYBRID[hyb]}

		echo ">>>      [x]           [ ]       Testing label propagation using ${DATASET} dataset, Hybrid P=${P}."
		echo
		./set_omp.sh -q ${HYBRID} && ${LPFRUN} -np ${P} -probe 5 -mpirun,-genv -mpirun,OMP_NUM_THREADS=`./set_omp.sh ${HYBRID}` -mpirun,-bind-to -mpirun,socket bin/tests/automatic_launch_label_hybrid_test ${PARSE_SIZE} &> bin/tests/output/automatic_launch_label_hybrid_test_${DATASET}_${P}
		head -1 bin/tests/output/automatic_launch_label_hybrid_test_${DATASET}_${P}
		(grep -q "${RESULT} total iterations" bin/tests/output/automatic_launch_label_hybrid_test_${DATASET}_${P} && grep "Test OK." bin/tests/output/automatic_launch_label_hybrid_test_${DATASET}_${P}) || (printf "Test FAILED.\n")
		echo
	done

done

