TNAME="softmaxOpv1"

. compile.sh

#generate input data in "input" directory
# and the reference output data in "output" directory
cd bin/
rm -f runtime*.csv
for n in {0..0}
do
    for axes in "8 32 64" "8 32 128" "8 256 128"  "32 128 128"
    do
	rm -rf input output

	echo "generate input"
	mkdir -p input
	mkdir -p output
	python3 ../make_data_softmaxOp-v1.py $axes || { echo "$TARGET make data failed" && exit 1; }

	echo "run ascend example"
	echo "./$TARGET $axes"
	./$TARGET $axes || { echo "$TARGET failed" && exit 1; }

	python3 ../check_data_softmaxOp-v1.py $axes || { echo "$TARGET check failed" && exit 1; }
    done

done
