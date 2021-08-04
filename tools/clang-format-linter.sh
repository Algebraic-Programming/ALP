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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# assuming we are in a subdirectory of the code (currently 'tools')
REPO_ROOT="$(dirname ${SCRIPT_DIR})"
CURDIR="$(pwd)"
SCRIPT_NAME=`basename "$0"`
CXX_EXT='.*\.\(cpp\|hpp\|cc\|cxx\|hh\|hxx\)'
MIN_VER=11
CF_COMMAND="clang-format-${MIN_VER}"
LINTER="${CF_COMMAND} -style=file"
INPLACE=no
TARGET="file"

function print_synopsis() {
	echo "SYNOPSIS: ${SCRIPT_NAME} [OPTIONS] <file(s)...>"
	echo " OPTIONS:"
	echo "  --help, -h prints this help"
	echo "  --in-place, -i to lint the file(s) in place"
	echo "  --lint-whole-grb lints the whole GraphBLAS codebase"
	echo "  --tree, -t <dir> lints all the files in the given directory"
	echo " <file(s)...> lints the given file(s)"
}

(which ${CF_COMMAND} &> /dev/null)
FOUND=$?
if [[ "${FOUND}" -ne "0" ]]; then
	echo -e "Cannot find the command '${CF_COMMAND}'"
	exit -1
fi

if [[ $# -eq 0 ]]; then 
	echo -e "No argument given!"
	print_synopsis
	exit -1
fi

VERSION="$(${CF_COMMAND} --version | sed -r 's/.*version\s+([0-9]+).*/\1/g')"
if [[ "${VERSION}" -lt "${MIN_VER}" ]]; then
	echo -e "Detected ${CF_COMMAND} version ${VERSION}, while version ${MIN_VER} or greater is expected."
	echo -e "The applied format may not be as expected or the tool may unexpectedly terminate: cannot proceed!"
	exit -1
fi

ALL_ARGS=("$@")
while test -n "$1"
do
	case "$1" in
		--help|-h)
			print_synopsis
			exit 0
			;;
		--lint-whole-grb)
			TARGET="tree"
			shift
			;;
		--in-place|-i)
			INPLACE=yes
			shift
			;;
		--tree|-t)
			TARGET="tree"
			REPO_ROOT=$(cd "$2" && pwd)
			if [[ "$?" != "0" ]]; then
				echo -e "'$2' is not a valid directory"
				exit -1
			fi
			shift 2
			;;
		-*)
			echo -e "unknown option $1"
			print_synopsis
			exit -1
		   ;;
		*)
			break
			;;
	esac
done

files="$@"
if [[ "${TARGET}" = "tree" ]]; then
	echo "Linting all source files inside ${REPO_ROOT}"
	files="$(find "${REPO_ROOT}" -regex "${CXX_EXT}" )"
fi

# to amortize for format parsing, chunk the list of files
# into chunks of g length and pass each list to clang-format,
# which ignores too long lists
g=10
files=( ${files} )
for (( i=0; i < ${#files[@]}; i+=g ))
do
	part=( "${files[@]:i:g}" )
	if [ "x${INPLACE}" = "xyes" ]; then
		sed -i 's/#pragma omp/\/\/#pragma omp/' ${part[*]}
		${LINTER} -i ${part[*]}
		sed -i 's/\/\/#pragma omp/#pragma omp/' ${part[*]}
	else
		sed 's/#pragma omp/\/\/#pragma omp/' ${part[*]} | ${LINTER} | sed 's/\/\/#pragma omp/#pragma omp/'
	fi
done
echo "=> To check all files are properly linted until convergence, you should run the same command again <=" >&2

