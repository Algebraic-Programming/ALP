TNAME="softmaxOpv4"

. compile.sh

#generate input data in "input" directory
# and the reference output data in "output" directory
cd bin/
rm -f runtime*.csv
for n in {0..0}
do
    for axes in "16 4 16 8 4 16"  "16 4 16 8 4 128" "32 4 16 8 4 16"  "16 4 32 8 4 16"
    do
	rm -rf input output

	echo "generate input"
	mkdir -p input
	mkdir -p output
	python3 ../make_data_softmaxOp-v4.py $axes || { echo "$TARGET make data failed" && exit 1; }

	echo "run ascend example"
	echo "./$TARGET $axes"
	./$TARGET $axes || { echo "$TARGET failed" && exit 1; }

	python3 ../check_data_softmaxOp-v4.py $axes || { echo "$TARGET check failed" && exit 1; }
    done

done
