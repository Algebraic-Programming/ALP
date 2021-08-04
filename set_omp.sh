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

divideBy=1
divSet=false
prntToStdOut=true
while [[ $# -gt 0 ]]
do
	arg="$1"
	case $arg in
		-q|--quiet)
			prntToStdOut=false
			shift
		;;
		-h|--help)
			echo "Usage: $0 <processes>"
			echo "Sets the OMP_NUM_THREADS environment to the number of hardware threads divided by processes. Also returns this same number to stdout."
			echo
			echo "<processes> is optional. Its default value is 1."
			echo
			echo "Example use: ./set_omp.sh 2 && bsprun -engine mpimsg -probe 5 -np 2 -- -genv OMP_NUM_THREADS=`./set_omp.sh` ./a.out"
			echo "             (this sets both the OMP_NUM_THREADS envvar in the master shell, which might not get passed depending on which MPI is used; AND"
			echo "              passes the environment variable explicitly via -genv, which might not be supported depending on which MPI is used. Doing both"
			echo "              seems to be robust against interchanging various different often-used MPIs)."
			echo
			echo "Options:"
			echo " -q or --quiet the output to stdout is surpressed (only the environment variable is set)."
			echo " -h, or --help print this help message and exit."
			echo
			exit 0
		;;
		*)
			if ${divSet};
			then
				echo "Cannot parse argument $1. Did not attempt to interpret this as the <processes> argument since it already was already set to ${divideBy}. Will now exit with error."
				exit 1
			fi
			divideBy=${arg}
			divSet=true
			shift
		;;
	esac
done
THREADS=`grep processor /proc/cpuinfo | tail -1 | cut -d':' -f2- | sed 's/\ //g'`
THREADS=$(( ${THREADS} + 1 ))
export OMP_NUM_THREADS=$(( ${THREADS} / ${divideBy} ))
if [[ ${OMP_NUM_THREADS} -eq 0 ]] ; then
	export OMP_NUM_THREADS=1
fi
if ${prntToStdOut}; then echo "${OMP_NUM_THREADS}"; fi
exit 0

