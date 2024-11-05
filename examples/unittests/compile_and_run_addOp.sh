TNAME="addOp"

. compile.sh

#generate input data in "input" directory
# and the reference output data in "output" directory
cd bin
rm -f runtime*.csv
for n in {0..0}
do
    for axes in "256 2048" "512 2048" "1024 2048" "2048 2048" "4096 2048" "8192 2048"
    do
	rm -rf input output

	echo "generate input"
	mkdir -p input
	mkdir -p output
	python3 ../make_data_addOp.py ${axes} || { echo "$TARGET make data failed" && exit 1; }

	echo "run ascend example"
	echo "./$TARGET"
	./$TARGET ${axes} || { echo "$TARGET returned error" && exit 1; }

	#check the result correctness
	echo "compare md5sum : ";md5sum output/*.bin
	md5_ref=($(md5sum output/golden.bin))
	md5_res=($(md5sum output/param2.bin))
	RED='\033[0;31m'
	GREEN='\033[0;32m'
	DEF='\033[0m'
	if [ "$md5_ref" == "$md5_res" ]
	then
	    printf "${GREEN}Test OK!${DEF}\n"
	else
	    printf "${RED}Test FAILED!${DEF}\n"
	    exit 1
	fi
    done
done



