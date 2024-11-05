TNAME="softmaxOp"

. compile.sh

#generate input data in "input" directory
# and the reference output data in "output" directory
cd bin/
rm -f runtime*.csv
for n in {0..0}
do
    for axes in "1024 32 128" "1024 128 64" "1024 128 128" "1024 256 64"
    do
	rm -rf input output

	echo "generate input"
	mkdir -p input
	mkdir -p output
	python3 ../make_data_softmaxOp.py $axes || { echo "$TARGET data generation failed" && exit 1; }

	echo "run ascend example"
	echo "./$TARGET $axes"
	./$TARGET $axes || { echo "$TARGET  failed" && exit 1; }

	python3 ../check_data_softmaxOp.py $axes || { echo "$TARGET check failed" && exit 1; }
    done

done



