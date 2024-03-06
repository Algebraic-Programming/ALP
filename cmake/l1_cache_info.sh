#!/bin/bash

#
#   Copyright 2024 Huawei Technologies Co., Ltd.
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

function symbolic_to_bytes {
	local symbolic_size="$1"
	local _symbolic_size=${symbolic_size//M/*1024*1024}
	local _byte_size=${_symbolic_size//K/*1024}
	echo $((_byte_size))
}

function test_not_zero {
	if [[ "$1" == "0" ]]; then
		echo "0 value: makes no sense!"
		exit 1
	fi
}

INFO_ROOT="/sys/devices/system/cpu/cpu0/cache"

# look for Data cache (Harvard architecture); Unified is also accepted
for f in ${INFO_ROOT}/index*; do
	level=$(cat ${f}/level)
	if [[ "$?" != "0" ]]; then
		echo "error detecting the cache level"
		exit 1
	fi
	if [[ "${level}" != "1" ]]; then
		continue
	fi
	type=$(cat ${f}/type)

	if [[ "${type}" == "Data" || "${type}" == "Unified" ]]; then
		cache_dir=${f}
		break
	fi
done

if [[ -z "${cache_dir}" ]]; then
	echo "cannot find cache info"
	exit 1
fi

echo "TYPE: ${type}"

cache_symbolic_size=$(cat ${cache_dir}/size)
cache_byte_size=$(symbolic_to_bytes ${cache_symbolic_size})
test_not_zero "${cache_byte_size}"
echo "SIZE: ${cache_byte_size}"

symbolic_line_size=$(cat ${cache_dir}/coherency_line_size)
line_byte_size=$(symbolic_to_bytes ${symbolic_line_size})

test_not_zero "${line_byte_size}"
echo "LINE: ${line_byte_size}"
