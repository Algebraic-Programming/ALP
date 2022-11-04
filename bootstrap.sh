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

GRB_NAME='ALP/GraphBLAS'

check_cmd() {
	cmd_path=$(command -v "${1}")
	if [[ -z "${cmd_path}" ]]; then
		echo " Command \"${1}\" not found!"
		exit -1
	else
		echo "${cmd_path}"
	fi
}

check_cc_cpp_comp() {
	printf "Checking for C compiler..."
	if [[ -z "${CC}" ]]; then
		CC="cc"
	fi
	check_cmd "${CC}"
	printf "Checking for C++ compiler..."
	if [[ -z "${CXX}" ]]; then
		CXX="c++"
	fi
	check_cmd "${CXX}"
}

check_lpf() {
	printf "Checking for lpfcc..."
	check_cmd "$1lpfcc"
	printf "Checking for lpfcxx..."
	check_cmd "$1lpfcxx"
	printf "Checking for lpfrun..."
	check_cmd "$1lpfrun"
}

# validates the path and returns the absolute one WITHOUT trailing '/'
validate_command_result() {
	err_val=$1
	error_msg=$2
	if [[ "${err_val}" != "0" ]]; then
		echo "${error_msg}"
		echo
		exit -1;
	fi
}

print_help() {
	echo "Usage: $0 --prefix=<path> [--with-lpf[=<path>]]\
 [--with-banshee=<path>] [--with-snitch=<path>] [--no-reference] [--debug-build] [--generator=<value>] [--show] [--delete-files]"
	echo " "
	echo "Required arguments:"
	echo "  --prefix=<path/to/install/directory/>"
	echo " "
	echo "Optional arguments:"
	echo "  --with-lpf[=<path/to/lpf/install>]  - enable LPF backend, optionally specifying \
the location where LPF is installed"
	echo "  --with-banshee=<path/>              - path to the the tools to compile the banshee backend"
	echo "  --with-snitch=<path/>               - path to the tools for Snitch support within the banshee backend"
	echo "  --no-alp-reference                  - to compile without support for dense algebraic programming"
	echo "  --no-reference                      - disables the reference and reference_omp backends"
	echo "  --debug-build                       - build the project with debug options (tests will run much slower!)"
	echo "  --generator=<value>                 - set the generator for CMake (otherwise use CMake's default)"
	echo "  --show                              - show generation commands instead of running them"
	echo "  --delete-files                      - delete files in the current directory without asking for confirmation"
	echo "  --help                              - prints this help"
	echo " "
	echo "Notes:"
	echo "  - If the install directory does not exist, it will be created."
	echo "  - The --prefix path is mandatory. No make targets will execute"
	echo "    without configuring this path first."
	echo "  - This $0 script is re-entrant and will simply overwrite"
	echo "    *all* previous setting, but will not clean up anything. For best"
	echo "    results, execute $0 only once on a clean source directory."
}

reference=yes
banshee=no
lpf=no
alp_reference=yes
alp_omp=yes
show=no
FLAGS=$''
LPF_INSTALL_PATH=
BANSHEE_PATH=
SNITCH_PATH=
debug_build=no
generator=
delete_files=no

if [[ "$#" -lt 1 ]]; then
	echo "No argument given, at least --prefix=<path/to/install/directory/> is mandatory"
	echo
	print_help
	exit 1
fi

for arg in "$@"
do
	case "$arg" in
	-h|--help)
			print_help
			exit
			;;
	--prefix=*)
			prefix="${arg#--prefix=}"
			;;
	--with-lpf*)
			suffix=${arg#--with-lpf}
			if [[ -z "${suffix}" ]]; then
				LPF_INSTALL_PATH=""
			else
				if [[ "${suffix:0:1}" != "=" ]]; then
					echo "please, specify the LPF installation path with --with-lpf=</path/to/lpf/install>"
					exit 1
				fi
				LPF_INSTALL_PATH="${suffix:1}"
				if [[ -z "${LPF_INSTALL_PATH}" ]]; then
					echo "please, specify a path after = (e.g. --with-lpf=</path/to/lpf/install>) \
or assume default paths (--with-lpf)"
				fi
			fi
			lpf=yes
			;;
	--with-banshee=*)
			BANSHEE_PATH="${arg#--with-banshee=}"
			banshee=yes
			;;
	--with-snitch=*)
			SNITCH_PATH="${arg#--with-snitch=}"
			banshee=yes
			;;
	--no-alp-reference)
			alp_reference=no
			;;
	--no-reference)
			reference=no
			;;
	--debug-build)
			debug_build=yes
			;;
	--generator=*)
			generator="${arg#--generator=}"
			;;
	--show)
			show=yes
			;;
	--delete-files) # useful for scripts
			delete_files=yes
			;;
	*)
			echo "Unknown argument ${arg}"
			exit 1
			;;
	esac
done

# VALIDATE INPUTS

if [[ -z "${prefix}" ]]; then
	echo "Error: empty install prefix. The --prefix=</install/path/> option is mandatory."
	exit 1;
fi

echo

# ~ is a valid character in file names according to UNIX standard, but some shells
# like Bash give it a special meaning and thus do the substitution ~ => $HOME
# (unless wrapped by quotes), while realpath doesn't do it; however, Bash'
# substitution excludes the usage of wrapping quotes, making Bash commands fool in
# case of paths with spaces, which would look like different arguments; hence,
# we do a manual substitution and then we call the various tools with wrapping
# quotes to support spaces as well
prefix="${prefix/#\~/$HOME}"
parent_dir_relative=$(dirname "${prefix}")
PARENT_DIR="$(realpath -e -q "${parent_dir_relative}")"
validate_command_result "$?" "Parent directory path '${parent_dir_relative}' for --prefix \
does not exist: please create it before invocation"
BASENAME=$(basename "${prefix}")
ABSOLUTE_PREFIX="${PARENT_DIR}/${BASENAME}"
if [[ ! -d "${ABSOLUTE_PREFIX}" ]]; then
	echo "Warning: install directory will be created when output is written there \
(either on \`make install' or on installing a pre-packaged dependency)."
	echo
else
	echo "Warning: any pre-existing data in ${ABSOLUTE_PREFIX} may be overwritten."
	echo
fi


if [[ "${reference}" == "yes" || "${lpf}" == "yes" ]]; then
	check_cc_cpp_comp
fi

if [[ "${lpf}" == "yes" ]]; then
	if [[ -z "${LPF_INSTALL_PATH}" ]]; then
		check_lpf
		lpfrun_path=$(command -v lpfcc)
		ABSOLUTE_LPF_INSTALL_PATH=$(dirname "${lpfrun_path}")
	else
		ABSOLUTE_LPF_INSTALL_PATH="$(realpath -e -q "${LPF_INSTALL_PATH/#\~/$HOME}")"
		validate_command_result "$?" "LPF installation path '${LPF_INSTALL_PATH}' does not exist"
		check_lpf "${ABSOLUTE_LPF_INSTALL_PATH}/bin/"
	fi
fi


if [[ "${banshee}" == "yes" ]]; then
	ABSOLUTE_BANSHEE_PATH="$(realpath -e -q "${BANSHEE_PATH/#\~/$HOME}")"
	validate_command_result "$?" "Invalid Banshee installation path '${BANSHEE_PATH}', \
please provide the path to the Banshee toolchain via --with-banshee=</path/to/banshee>"
	ABSOLUTE_SNITCH_PATH="$(realpath -e -q "${SNITCH_PATH/#\~/$HOME}")"
	validate_command_result "$?" "Invalid Snitch installation path '${SNITCH_PATH}', \
please provide the path to the Snitch toolchain via --with-snitch=</path/to/snitch>"
fi

CURRENT_DIR="$(pwd)"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# CONFIGURE CMAKE BUILDING INFRASTRUCTURE
if [[ "${reference}" == "yes" || "${lpf}" == "yes" ]]; then
	BUILD_DIR="${CURRENT_DIR}"

	printf "Checking for cmake..."
	check_cmd "cmake"
	echo
	# if show==yes, do not check for in-source build or for existing files
	# but proceed to finally show the build command
	if [[ "${show}" == "no" ]]; then

		if [[ "${BUILD_DIR}" == "${SCRIPT_DIR}" ]]; then
			echo "You should not build ${GRB_NAME} inside its source directory \"${SCRIPT_DIR}\""
			echo "Please, create a new directory anywhere, move into it and invoke this script from there"
			echo -e "Example:\n  mkdir build\n  cd build\n  ../$(basename "$0") $@"
			echo
			exit -1
		fi

		files=$(ls "${BUILD_DIR}")
		# if there are files, ask the user to delete them, unless --delete-files was given
		if [[ ! -z "${files}" ]]; then

			if [[ "${delete_files}" != "yes" ]]; then
				read -p "The current directory \"${BUILD_DIR}\" is not empty: do you agree \
on deleting its content [yes/No] " -r REPLY
			else
				REPLY="yes"
			fi
			if [[ "${REPLY}" != "yes" ]]; then
				echo "You answered \"${REPLY}\", so configuration cannot proceed: you should empty \
the current directory before invocation or confirm the deletion of its content with \"yes\""
				echo
				exit -1
			fi
			echo "Deleting the content of \"${BUILD_DIR}\"..."
			rm -rf "'${BUILD_DIR}'/*"
			echo
		fi
		echo "*** CONFIGURING CMake inside \"${BUILD_DIR}\" ***"
		echo
	fi

	CMAKE_OPTS="-DCMAKE_INSTALL_PREFIX='${ABSOLUTE_PREFIX}'"

	if [[ "${debug_build}" == "yes" ]]; then
		CMAKE_OPTS+=" -DCMAKE_BUILD_TYPE=Debug"
	else
		CMAKE_OPTS+=" -DCMAKE_BUILD_TYPE=Release"
	fi

	DATASETS_DIR="${CURRENT_DIR}/datasets"
	if [[ -d "${DATASETS_DIR}" ]]; then
		CMAKE_OPTS+=" -DDATASETS_DIR='${DATASETS_DIR}'"
	fi
	# GNN_DATASET_PATH is not needed, because unittests.sh sets a default

	if [[ "${reference}" == "no" ]]; then
		CMAKE_OPTS+=" -DWITH_REFERENCE_BACKEND=OFF -DWITH_OMP_BACKEND=OFF"
	fi
	if [[ "${alp_reference}" == "no" ]]; then
		CMAKE_OPTS+=" -DWITH_ALP_REFERENCE_BACKEND=OFF"
	else
		CMAKE_OPTS+=" -DWITH_ALP_REFERENCE_BACKEND=ON"
	fi
	if [[ "${alp_omp}" == "no" ]]; then
		CMAKE_OPTS+=" -DWITH_ALP_OMP_BACKEND=OFF"
	else
		CMAKE_OPTS+=" -DWITH_ALP_OMP_BACKEND=ON"
	fi
	if [[ "${lpf}" == "yes" ]]; then
		CMAKE_OPTS+=" -DLPF_INSTALL_PATH='${ABSOLUTE_LPF_INSTALL_PATH}'"
	fi

	if [[ ! -z "${generator}" ]]; then
		CMAKE_OPTS+=" -G '${generator}'"
	fi

	CONFIG_COMMAND="cmake ${CMAKE_OPTS} ${SCRIPT_DIR}/"
	if [[ "${show}" == "yes" ]]; then
		echo "Configuration command:"
		echo "${CONFIG_COMMAND}"
	else
		bash -c "${CONFIG_COMMAND}"
		if [[ $? -ne 0 ]]; then
			echo "Error during generation of the CMake infrastructure"
			echo
			exit -1
		fi
		echo
		echo "*** you can build ${GRB_NAME} inside \"${BUILD_DIR}\" ***"
	fi
fi

if [[ "${banshee}" == "yes" ]]; then
	echo
	echo "************************* WARNING *************************"
	echo "You have selected the _Banshee_ backend, which uses dedicated"
	echo "Makefile to build and is considered experimental"
	echo
	echo "*** CONFIGURING Makefile inside ${SCRIPT_DIR} ***"

	if [[ "${show}" == "yes" ]]; then
		echo
		echo "paths.mk contents:"
		echo "GRB_INSTALL_PATH=${ABSOLUTE_PREFIX}"
		echo "LPF_INSTALL_PATH=${ABSOLUTE_LPF_INSTALL_PATH}"
		echo "BANSHEE_PATH=${ABSOLUTE_BANSHEE_PATH}"
		echo "SNITCH_PATH=${ABSOLUTE_SNITCH_PATH}"
	else
		echo "GRB_INSTALL_PATH=${ABSOLUTE_PREFIX}" > "${SCRIPT_DIR}"/paths.mk
		echo "LPF_INSTALL_PATH=${ABSOLUTE_LPF_INSTALL_PATH}" >> "${SCRIPT_DIR}"/paths.mk
		echo "BANSHEE_PATH=${ABSOLUTE_BANSHEE_PATH}" >> "${SCRIPT_DIR}"/paths.mk
		echo "SNITCH_PATH=${ABSOLUTE_SNITCH_PATH}" >> "${SCRIPT_DIR}"/paths.mk

		echo
		echo "*** you can build ${GRB_NAME} - _Banshee_ in \"${SCRIPT_DIR}\" ***"
	fi
fi

echo
echo "Configure done."

