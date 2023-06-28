#!/bin/bash

#
#   Copyright 2023 Huawei Technologies Co., Ltd.
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

### runner for validation: runs the command passed in input, typically with piping or other
# Bash operators (&&, ||, ...)

# log of test
infile=$1
# path to output file for the validation
outfile=$2
shift 2

# validation command
validation_command=$@

echo ">>> running: ${validation_command}"
eval "${validation_command}"
retcode=$?

if [[ "${retcode}" != "0" && -f "${infile}" ]]; then
	cp "${infile}" "${outfile}"
	echo "-- copying original input file \"${infile}\" into \"${outfile}\""
	# report attachment for Gitlab CI
	if [[ ! -z "${CI_PROJECT_DIR}" ]]; then
		rel_path=$(realpath --relative-to="${CI_PROJECT_DIR}" "${outfile}")
		echo "[[ATTACHMENT|${rel_path}]]"
	fi
fi

exit ${retcode}
