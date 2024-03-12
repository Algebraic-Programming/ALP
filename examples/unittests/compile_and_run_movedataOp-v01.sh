TNAME="movedataOpv01"

. compile.sh

#generate input data in "input" directory
# and the reference output data in "output" directory
cd bin/
rm -f runtime*.csv
for n in {0..0}
do
    for axes in "32 16 16" "64 16 16" "128 16 16" "256 16 16"  "512 16 16"
    do
	echo "axes=$axes"
	rm -rf input output

	echo "generate input"
	mkdir -p input
	mkdir -p output
	python3 ../make_data_movedataOp-v01.py $axes  || { echo "$TARGET data generation failed" && exit 1; }

	echo "run ascend example"
	echo "./$TARGET $axes"
	./$TARGET $axes   || { echo "$TARGET failed" && exit 1; }

	python3 ../check_data_movedataOp-v01.py $axes || { echo "$TARGET check failed" && exit 1; }
    done
done



