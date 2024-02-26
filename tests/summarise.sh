#!/bin/bash

if [ $# -lt 1 ]; then
	echo "Usage: $0 <test log> <output file 1> <output file 2> ..."
	echo " - test log: mandatory log of unittests, smoketests, or perftests output"
	echo " - output files: mandatory output files to be reported if log was OK"
	echo "The number of mandatory output files is optional and can be zero"
	exit 1
fi

if [ ! -f "$1" ]; then
	echo "Given log file (${1}) was not found"
	exit 50
fi

NUM_OK=`grep -i "Test OK" "$1" | wc -l`
NUM_DISABLED=`grep -i "Test DISABLED" "$1" | wc -l`
NUM_CDISABLED=`grep -i "Tests DISABLED" "$1" | wc -l`
NUM_FAILED=`grep -i "Test FAILED" "$1" | wc -l`

echo "Summary of $1:"
printf "  %4s PASSED\n" ${NUM_OK}
printf "  %4s SKIPPED\n" ${NUM_DISABLED}
printf "  %4s FAILED\n" ${NUM_FAILED}
printf "  %4s TEST CATEGORIES SKIPPED\n" ${NUM_CDISABLED}

if [ ${NUM_FAILED} -gt 0 ]; then
	printf "\nOne or more failures detected. Log contents:\n\n"
	cat "$1"
	exit 100
fi

if [ ${NUM_OK} -eq 0 ]; then
	printf "\nZero tests succeeded. Log contents:\n\n"
	cat "$1"
	exit 200
fi

shift

for OUTPUT in "$@"
do
	if [ ! -f "${OUTPUT}" ]; then
		echo "Mandatory output file (${OUTPUT}) was not found"
		cat "$1"
		exit 250
	fi
	printf "\nContents of ${OUTPUT}:"
	printf "\n------------------------------------\n\n"
	cat ${OUTPUT}
done

exit 0

