#!/bin/bash

SCRIPTS="compile_and_run_movedataOp-v01.sh compile_and_run_addOp.sh compile_and_run_addOpv1.sh compile_and_run_softmaxOp.sh compile_and_run_softmaxOp-v1.sh compile_and_run_softmaxOp-v3.sh compile_and_run_softmaxOp-v4.sh compile_and_run_onlinesoftmaxOp.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
DEF='\033[0m'

echo ""

BUILD=$(pwd)/build_alp/
rm -rf $BUILD
mkdir $BUILD

PAD_LEN=$(for script in $SCRIPTS ; do echo $script ; done | wc --max-line-length)
PAD_LEN="$((PAD_LEN-16))"

for script in $SCRIPTS
do
	testname=$(echo -n ${script:16:-3})
	BUILD_DIR=$BUILD ./$script 2&>> /dev/null
	if [ $? -ne 0 ]
	then
		printf "%-${PAD_LEN}s ${RED}FAILED${DEF} \n" $testname
		exit 1
	else
		printf "%-${PAD_LEN}s ${GREEN}PASSED${DEF} \n" $testname
	fi
done
echo -e "\nAll tests OK!"
rm -rf $BUILD
