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

if [ -z "${BACKENDS}" ]; then
	echo "BACKENDS is not set!"
	exit 255;
fi
echo "Info: BACKENDS is set to ${BACKENDS}"

if [[ ! -z ${OMP_NUM_THREADS} ]]; then
	echo "Warning: OMP_NUM_THREADS was set (value was \`${OMP_NUM_THREADS}')"
fi

if [[ -z ${MAX_THREADS} ]]; then
	if ! command -v nproc &> /dev/null; then
		echo "Error: nproc command does not exist while MAX_THREADS was not set."
		echo "Please set MAX_THREADS explicitly and try again."
		exit 255;
	else
		MAX_THREADS=`nproc --all`
		echo "Info: detected ${MAX_THREADS} threads"
	fi
fi

mkdir bin/tests/output || true

echo "****************************************************************************************"
echo "      FUNCTIONAL    PERFORMANCE                       DESCRIPTION      "
echo "----------------------------------------------------------------------------------------"
echo ">>>      [x]           [ ]       Testing numerical addition operator over doubles"
echo " "
bin/tests/add15d

echo ">>>      [x]           [ ]       Testing numerical addition operator over a mixed field"
echo "                                 (double, integers, and floats)"
echo " "
bin/tests/add15m

echo ">>>      [x]           [ ]       Testing numerical multiplication operator over integers"
echo " "
bin/tests/mul15i

echo ">>>      [x]           [ ]       Testing numerical multiplication operator over a mixed"
echo "                                 field (double, integers, and floats)"
echo " "
bin/tests/mul15m

echo ">>>      [x]           [ ]       Tests the built-in parser on the west0497 MatrixMarket file"
if [ -f datasets/west0497.mtx ]; then
	bin/tests/parserTest 2> bin/tests/output/parserTest.err
else
	echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
fi
echo " "

echo ">>>      [x]           [ ]       Tests the built-in parser (in graphblas/utils/parser.hpp)"
echo "                                 versus the parser in tests/parser.cpp on cit-HepTh.txt."
if [ -f datasets/cit-HepTh.txt ]; then
	bin/tests/compareParserTest &> bin/tests/output/compareParserTest
	head -1 bin/tests/output/compareParserTest
	tail -2 bin/tests/output/compareParserTest
else
	echo "Test DISABLED: cit-HepTh was not found. To enable, please provide the dataset within datasets/cit-HepTh.txt"
	echo " "
fi

echo ">>>      [x]           [ ]       Tests the built-in high-performance parser (in"
echo "                                 include/graphblas/utils/parser.h &"
echo "                                  src/graphblas/utils/parser.c) on dwt_59.mtx"
echo "                                 Parameters: P=1, no hyperthreads (half the available threads),"
echo "                                 block size = 128k, buffer size = 8M"
if [ -f datasets/dwt_59.mtx ]; then
	echo "Functional test executable: bin/tests/hpparser"
	bin/tests/hpparser 1 `./set_omp.sh 2` 131072 8388608 datasets/dwt_59.mtx 1 &> bin/tests/output/hpparser
	echo "[ 0, *] nrow =           59, ncol =           59, nnnz =          163
[ 0, *] offb =          564, fsiz =         1494, offe =         1493
[ *, *] ntot =          163" > bin/tests/output/hpparser.chk
	(diff bin/tests/output/hpparser bin/tests/output/hpparser.chk && printf "Test OK.\n\n") || printf "Test FAILED.\n\n"
else
	echo "Test DISABLED: dwt_59.mtx was not found. To enable, please provide datasets/dwt_59.mtx"
fi
echo " "

for BACKEND in ${BACKENDS[@]}; do
	if [ "$BACKEND" = "bsp1d" ]; then
		Ps=( 1 2 16 )
	fi
	if [ "$BACKEND" = "hybrid" ]; then
		Ps=( 2 7 )
	fi
	if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
		if [ -z "${LPFRUN}" ]; then
			echo "LPFRUN is not set!"
			exit 255;
		fi
		if [ -z "${MANUALRUN}" ]; then
			echo "MANUALRUN is not set!"
			exit 255;
		fi

		echo "Info: LPFRUN is set to ${LPFRUN}"
		echo "Info: MANUALRUN is set to ${MANUALRUN}"
	else
		Ps=( 1 )
	fi
	if [ "$BACKEND" = "reference_omp" ] ; then
		Pt=( 1 2 ${MAX_THREADS} )
	elif [ "$BACKEND" = "hybrid" ]; then
		MTDS=$((MAX_THREADS/7))
		if [ "$MTDS" -le "2" ]; then
			Pt=( 2 )
		else
			Pt=( 2 $((MAX_THREADS/7)) )
		fi
	else
		Pt=( 1 )
	fi
	for P in ${Ps[@]}; do
		for T in ${Pt[@]}; do

			echo "Starting standardised unit tests for the ${BACKEND} backend"
			echo "  using ${P} user processes"
			echo "  using ${T} threads"
			echo "==========================================================="

			if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
				runner="${LPFRUN} -n ${P}"
			else
				runner=
			fi

			if [ "$BACKEND" = "reference_omp" ] ; then
				export OMP_NUM_THREADS=${T}
			fi

			echo ">>>      [x]           [ ]       Testing grb::set on vectors of doubles of size"
			echo "                                 1 000 000."
			$runner bin/tests/set_${BACKEND} 1000000 2> bin/tests/output/set_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::eWiseApply using (+,0) on vectors"
			echo "                                 of doubles of size 100."
			$runner bin/tests/ewiseapply_${BACKEND} 100 &> bin/tests/output/ewiseapply_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/ewiseapply_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/ewiseapply_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::eWiseApply using (+,0) on vectors"
			echo "                                 of doubles of size 10 000 000."
			$runner bin/tests/ewiseapply_${BACKEND} 10000000 &> bin/tests/output/ewiseapply_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/ewiseapply_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/ewiseapply_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::zip on two vectors of doubles and"
			echo "                                 ints of size 10 000 000."
			$runner bin/tests/zip_${BACKEND} 10000000 &> bin/tests/output/zip_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/zip_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/zip_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::dot on two vectors of doubles and"
			echo "                                 ints of size 1874."
			$runner bin/tests/dot_${BACKEND} 1874 &> bin/tests/output/dot_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/dot_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/dot_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::dot on two vectors of doubles and"
			echo "                                 ints of size 10 000 000."
			$runner bin/tests/dot_${BACKEND} 10000000 &> bin/tests/output/dot_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/dot_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/dot_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing std::swap on two vectors of doubles of"
			echo "                                 size 100."
			$runner bin/tests/swapVector_${BACKEND} 100 &> bin/tests/output/swapVector_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/swapVector_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/swapVector_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing std::move on two vectors of doubles of"
			echo "                                 size 100."
			$runner bin/tests/moveVector_${BACKEND} 100 &> bin/tests/output/moveVector_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/moveVector_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/moveVector_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing std::vector of thirteen GraphBLAS"
			echo "                                 vectors of unsigned chars of sizes 100 and 50."
			$runner bin/tests/stdVector_${BACKEND} 100 &> bin/tests/output/stdVector_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/stdVector_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/stdVector_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::Vector's copy-constructor on a"
			echo "                                 vector of doubles of size 100."
			$runner bin/tests/copyVector_${BACKEND} 100 &> bin/tests/output/copyVector_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/copyVector_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/copyVector_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::Vector's copy-constructor on a"
			echo "                                 vector of doubles of size 10 000 000."
			$runner bin/tests/copyVector_${BACKEND} 10000000 &> bin/tests/output/copyVector_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/copyVector_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/copyVector_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::Matrix's output-iterators on"
			echo "                                 square matrices of size 15 and 1 000 000."
			$runner bin/tests/matrixIterator_${BACKEND} 1000000 2> bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.err 1> bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.log
			head -1 bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.log
			tail -n +2 bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.log | grep -v 'Test OK' | sort -n > bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.out
			if ! diff bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.out tests/matrixIterator.chk; then
				echo "Test FAILED verification"
			else
				grep 'Test OK' bin/tests/output/matrixIterator_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing copy and move constructors and assignment"
			echo "                                 of the const_iterator of grb::Vector< double > of"
			echo "                                 length 10 000 000."
			$runner bin/tests/copyAndAssignVectorIterator_${BACKEND} 10000000 &> bin/tests/output/copyAndAssignVectorIterator_${BACKEND}_${P}_${T}.log
			head -1 bin/tests/output/copyAndAssignVectorIterator_${BACKEND}_${P}_${T}.log
			grep 'Test OK' bin/tests/output/copyAndAssignVectorIterator_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			echo " "


			echo ">>>      [x]           [ ]       Testing grb::eWiseMulAdd on a vector of"
			echo "                                 doubles of size 7 000 000."
			$runner bin/tests/masked_muladd_${BACKEND} 7000000 &> bin/tests/output/masked_muladd_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/masked_muladd_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/masked_muladd_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::eWiseMulAdd on a vector of"
			echo "                                 doubles of size 10 000 000."
			$runner bin/tests/muladd_${BACKEND} 10000000 &> bin/tests/output/muladd_large_${BACKEND}_${P}_${T}
			head -1 bin/tests/output/muladd_large_${BACKEND}_${P}_${T}
			grep 'Test OK' bin/tests/output/muladd_large_${BACKEND}_${P}_${T} || echo "Test FAILED"
			echo " "
			echo ">>>      [x]           [ ]       Testing grb::buildVector and"
			echo "                                 grb::buildVectorUnique"
			$runner bin/tests/buildVector_${BACKEND} 2> bin/tests/output/buildVector_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::vectorToMatrixConverter"
			$runner bin/tests/vectorToMatrix_${BACKEND} &> bin/tests/output/vectorToMatrix_${BACKEND}_${P}_${T}.log
			head -1 bin/tests/output/vectorToMatrix_${BACKEND}_${P}_${T}.log
			grep 'Test OK' bin/tests/output/vectorToMatrix_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::clear on a 1M by 1M matrix of"
			echo "                                 doubles"
			$runner bin/tests/clearMatrix_${BACKEND} 10000000 2> bin/tests/output/clearMatrix_${BACKEND}_${P}_${T}
			echo " "

			echo ">>>      [x]           [ ]       Testing double-masked grb::vxm and grb::mxv"
			echo "                                 on a mock red-black Gauss Seidel on west0497."
			if [ -f datasets/west0497.mtx ]; then
				$runner bin/tests/RBGaussSeidel_${BACKEND} datasets/west0497.mtx &> bin/tests/output/RBGaussSeidel_${BACKEND}_${P}_${T}
				head -1 bin/tests/output/RBGaussSeidel_${BACKEND}_${P}_${T}
				grep 'Test OK' bin/tests/output/RBGaussSeidel_${BACKEND}_${P}_${T} || echo "Test FAILED"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
			fi
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::argmin"
			$runner bin/tests/argmin_${BACKEND} 2> bin/tests/output/argmin_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::argmax"
			$runner bin/tests/argmax_${BACKEND} 2> bin/tests/output/argmax_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::set (matrices)"
			$runner bin/tests/matrixSet_${BACKEND} 2> bin/tests/output/matrixSet_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing grb::eWiseLambda (matrices)"
			$runner bin/tests/eWiseMatrix_${BACKEND} 2> bin/tests/output/eWiseMatrix_${BACKEND}_${P}_${T}.err
			echo " "

			echo ">>>      [x]           [ ]       Testing the conjugate gradient algorithm for the input"
			echo "                                 matrix (17361x17361) taken from gyro_m.mtx."
			if [ -f datasets/gyro_m.mtx ]; then
				$runner bin/tests/automatic_launch_conjugate_gradient_${BACKEND} datasets/gyro_m.mtx indirect 1 1 &> bin/tests/output/automatic_launch_conjugate_gradient_${BACKEND}_${P}_${T}
				head -1 bin/tests/output/automatic_launch_conjugate_gradient_${BACKEND}_${P}_${T}
				grep 'Test OK' bin/tests/output/automatic_launch_conjugate_gradient_${BACKEND}_${P}_${T} || echo "Test FAILED"
			else
				echo "Test DISABLED: gyro_m.mtx was not found. To enable, please provide datasets/gyro_m.mtx"
			fi
			echo " "


			if [ -z ${GNN_DATASET_PATH} ]; then
				export GNN_DATASET_PATH=GraphChallengeDataset
			fi

			echo ">>>      [x]           [ ]       Testing the GNN algorithm for the input (neurons=1024,"
			echo "                                 layers=120, offset=294) taken from ${GNN_DATASET_PATH}."
			if [ -d ${GNN_DATASET_PATH} ]; then
				$runner bin/tests/automatic_launch_gnn_single_inference_${BACKEND} ${GNN_DATASET_PATH} 1024 120 294 indirect 1 1 &> bin/tests/output/automatic_launch_gnn_single_inference_${BACKEND}_${P}_${T}
				head -1 bin/tests/output/automatic_launch_gnn_single_inference_${BACKEND}_${P}_${T}
				grep 'Test OK' bin/tests/output/automatic_launch_gnn_single_inference_${BACKEND}_${P}_${T} || echo "Test FAILED"
			else
				echo "Test DISABLED: ${GNN_DATASET_PATH} was not found. To enable, please provide the dataset."
			fi
			echo " "
			
			if [ "$BACKEND" = "bsp1d" ] || [ "$BACKEND" = "hybrid" ]; then
				echo "Additional standardised unit tests not yet supported for the ${BACKEND} backend"
				echo
				continue
			fi
		
			echo ">>>      [x]           [ ]       Testing the k-means algorithm"
			$runner bin/tests/kmeans_unit_${BACKEND} &> bin/tests/output/kmeans_unit_${BACKEND}_${P}_${T}.log
			head -1 bin/tests/output/kmeans_unit_${BACKEND}_${P}_${T}.log
			tail -1 bin/tests/output/kmeans_unit_${BACKEND}_${P}_${T}.log
			echo " "

			echo ">>>      [x]           [ ]       Testing BLAS3 grb::mxm (unmasked) on simple matrices"
			echo "                                 of size 100 x 100 using the (+,*) semiring over"
			echo "                                 doubles"
			$runner bin/tests/mxm_${BACKEND} &> bin/tests/output/mxm_${BACKEND}_${P}_${T}.log
			head -1 bin/tests/output/mxm_${BACKEND}_${P}_${T}.log
			grep 'Test OK' bin/tests/output/mxm_${BACKEND}_${P}_${T}.log || echo "Test FAILED"
			echo " "

			echo ">>>      [x]           [ ]       Testing BLAS1 functions on empty vectors"
			echo " "
			$runner bin/tests/emptyVector_${BACKEND} 2> bin/tests/output/emptyVector_${BACKEND}_${P}_${T}.err
			
			echo ">>>      [x]           [ ]       Testing vector times matrix using the normal (+,*)"
			echo "                                 semiring over integers on a diagonal matrix"
			echo " "
			$runner bin/tests/vmx_${BACKEND} 2> bin/tests/output/vmx_${BACKEND}_${P}_${T}.err
			
			echo ">>>      [x]           [ ]       Testing vector times matrix using a (*,+) semiring over"
			echo "                                 doubles on a diagonal matrix"
			echo " "
			$runner bin/tests/vmxa_${BACKEND} 2> bin/tests/output/vmxa_${BACKEND}_${P}_${T}.err
		
			echo ">>>      [x]           [ ]       Testing vector times matrix using the number (+,*)"
			echo "                                 semiring over integers on a diagonal 15x15 matrix. Each"
			echo "                                 of the 15 output elements are computed element-by-element"
			echo "                                 using masked operations. The implementation should keep"
			echo "                                 both the mask and the output vector sparse."
			echo " "
			$runner bin/tests/sparse_vxm_${BACKEND} 2> bin/tests/output/sparse_vxm_${BACKEND}_${P}_${T}.err
		
			echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
			echo "                                 semiring over integers on a diagonal 15x15 matrix. Each"
			echo "                                 of the 15 output elements are computed element-by-element"
			echo "                                 using masked operations. The implementation should keep"
			echo "                                 both the mask and the output vector sparse. In this test,"
			echo "                                 the in_place variant is also tested-- there, also the"
			echo "                                 input vector shall be sparse."
			echo " "
			$runner bin/tests/sparse_mxv_${BACKEND} 2> bin/tests/output/sparse_mxv_${BACKEND}_${P}_${T}.err
			
			echo ">>>      [x]           [ ]       Testing vector times matrix using the number (+,*)"
			echo "                                 semiring over integers on a diagonal 15x15 matrix. Each"
			echo "                                 of the 15 output elements are computed, like the above"
			echo "                                 test, by masked operations. Instead of one element per"
			echo "                                 mask, this mask will have two elements. One element is"
			echo "                                 fixed to 3. All 14 combinations are tested. The in_place"
			echo "                                 specifier is tested as well."
			echo " "
			$runner bin/tests/masked_vxm_${BACKEND} 2> bin/tests/output/masked_vxm_${BACKEND}_${P}_${T}.err
			
			echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
			echo "                                 semiring over integers on a diagonal 15x15 matrix--"
			echo "                                 apart from mxv instead of vxm, this is the same test"
			echo "                                 as the above."
			echo " "
			$runner bin/tests/masked_mxv_${BACKEND} 2> bin/tests/output/masked_mxv_${BACKEND}_${P}_${T}.err

			echo ">>>      [x]           [ ]       Testing dense vector times matrix using the double (+,*)"
			echo "                                 semiring where matrix elements are doubles and vector"
			echo "                                 elements ints. The input matrix is taken from west0497."
			echo " "
			if [ -f datasets/west0497.mtx ]; then
				$runner bin/tests/vxm_${BACKEND} datasets/west0497.mtx &> bin/tests/output/vxm_${BACKEND}_${P}_${T}.west0497
				head -1 bin/tests/output/vxm_${BACKEND}_${P}_${T}.west0497
				grep 'Test OK' bin/tests/output/vxm_${BACKEND}_${P}_${T}.west0497 || echo "Test FAILED"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
			fi
			echo " "
			
			echo ">>>      [x]           [ ]       Testing matrix times dense vector using the double (+,*)"
			echo "                                 semiring where matrix elements are doubles and vector"
			echo "                                 elements ints. The input matrix is taken from west0497."
			echo " "
			if [ -f datasets/west0497.mtx ]; then
				$runner bin/tests/mxv_${BACKEND} datasets/west0497.mtx &> bin/tests/output/mxv_${BACKEND}_${P}_${T}.west0497
				head -1 bin/tests/output/mxv_${BACKEND}_${P}_${T}.west0497
				grep 'Test OK' bin/tests/output/mxv_${BACKEND}_${P}_${T}.west0497 || echo "Test FAILED"
			else
				echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
			fi
			echo " "
		done
	done

	if [ "$BACKEND" = "bsp1d" ]; then
		echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
		echo "                                 semiring over integers on a 10x10 matrix. The input vector"
		echo "                                 is sparse. Each of y=Ax, y=A^Tx, y=xA, and y=xA^T is"
		echo "                                 tested in turn. The implementation should result in a"
		echo "                                 sparse output vector sparse. Here, we use the BSP1D"
		echo "                                 back-end with P=1."
		echo " "
		echo "Functional test executable: bin/tests/automatic_launch_sparse_vxm"
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 10 1 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.10.1
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 10 2 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.10.2
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 10 3 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.10.3
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 10 4 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.10.4
		(grep -i "Test failed" bin/tests/output/automatic_launch_sparse_vxm.P1.10.?) || (grep -i "Test OK" bin/tests/output/automatic_launch_sparse_vxm.P1.10.?)
		echo " "

		echo ">>>      [x]           [ ]       Testing matrix times vector using the number (+,*)"
		echo "                                 semiring over integers on a 1000x1000 matrix. The"
		echo "                                 input vector is sparse. Each of y=Ax, y=A^Tx, y=xA,"
		echo "                                 and y=xA^T is tested in turn. The implementation"
		echo "                                 should result in a sparse output vector sparse. Here,"
		echo "                                 we use the BSP1D back-end with P=1."
		echo " "
		echo "Functional test executable: bin/tests/automatic_launch_sparse_vxm"
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 1000 1 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.1000.1
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 1000 2 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.1000.2
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 1000 3 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.1000.3
		${LPFRUN} -np 1 bin/tests/automatic_launch_sparse_vxm 1000 4 1 1 &> bin/tests/output/automatic_launch_sparse_vxm.P1.1000.4
		(grep -i "Test failed" bin/tests/output/automatic_launch_sparse_vxm.P1.1000.?) || (grep -i "Test OK" bin/tests/output/automatic_launch_sparse_vxm.P1.1000.?)
		echo " "
		
		echo ">>>      [x]           [ ]       Testing BSP1D distribution for a vector of size 100000"
		echo " "
		bin/tests/distribution_bsp1d
			
		echo ">>>      [x]           [ ]       Tests grb::Launcher on a dot-product of two vectors"
		echo "                                 of size 100000 on two processes, BSP1D implementation."
		echo "                                 The launcher is used in automatic mode. Launcher::exec"
		echo "                                 is used with statically sized input and output."
		echo " "
		${LPFRUN} -np 2 bin/tests/automatic_launch_bsp1d_dot &> bin/tests/output/automatic_launch_bsp1d_dot
		head -1 bin/tests/output/automatic_launch_bsp1d_dot
		tail -2 bin/tests/output/automatic_launch_bsp1d_dot
			
		echo ">>>      [x]           [ ]       Tests a manual call to bsp_hook via the LPF engine"
		echo "                                 configured in bsp.mk. A manual run of this test opens"
		echo "                                 additional testing possibilities and allows inspection"
		echo "                                 of the stdout corresponding to this simple ``hello"
	        echo "                                 world' test."
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_hw. Script hardcodes test for four"
		echo "separate processes running on and connecting to localhost on port 77770."
		echo " "
		bash -c "${MANUALRUN} bin/tests/manual_hook_hw localhost 0 4 77770 &> bin/tests/output/manual_hook_hw.0 & \
			${MANUALRUN} bin/tests/manual_hook_hw localhost 3 4 77770 &> bin/tests/output/manual_hook_hw.3 & \
			${MANUALRUN} bin/tests/manual_hook_hw localhost 1 4 77770 &> bin/tests/output/manual_hook_hw.1 & \
			${MANUALRUN} bin/tests/manual_hook_hw localhost 2 4 77770 &> bin/tests/output/manual_hook_hw.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_hw.1 && grep -q 'Test OK' bin/tests/output/manual_hook_hw.2 && grep -q 'Test OK' bin/tests/output/manual_hook_hw.3 && grep -q 'Test OK' bin/tests/output/manual_hook_hw.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::set"
		echo "                                 function over an array of doubles of 100 elements"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_set. Script hardcodes test for"
		echo "three separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_set localhost 0 3 77770 &> bin/tests/output/manual_hook_grb_set.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_set localhost 1 3 77770 &> bin/tests/output/manual_hook_grb_set.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_set localhost 2 3 77770 &> bin/tests/output/manual_hook_grb_set.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_set.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_set.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_set.0 && printf "Test OK.\n\n" ) || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::set"
		echo "                                 function over an array of ints of 100 000 elements"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_dot. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_dot localhost 0 4 77770 &> bin/tests/output/manual_hook_grb_dot.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_dot localhost 3 4 77770 &> bin/tests/output/manual_hook_grb_dot.3 & \
			${MANUALRUN} bin/tests/manual_hook_grb_dot localhost 1 4 77770 &> bin/tests/output/manual_hook_grb_dot.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_dot localhost 2 4 77770 &> bin/tests/output/manual_hook_grb_dot.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_dot.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_dot.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_dot.3 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_dot.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test the grb::reduce"
		echo "                                 function over an array of doubles"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_reduce. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_reduce localhost 0 4 77770 &> bin/tests/output/manual_hook_grb_reduce.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_reduce localhost 3 4 77770 &> bin/tests/output/manual_hook_grb_reduce.3 & \
			${MANUALRUN} bin/tests/manual_hook_grb_reduce localhost 1 4 77770 &> bin/tests/output/manual_hook_grb_reduce.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_reduce localhost 2 4 77770 &> bin/tests/output/manual_hook_grb_reduce.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_reduce.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_reduce.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_reduce.3 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_reduce.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas0 grb::collectives"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_collectives_blas0. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas0 localhost 0 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas0.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas0 localhost 3 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas0.3 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas0 localhost 1 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas0.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas0 localhost 2 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas0.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas0.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas0.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas0.3 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas0.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas1 grb::collectives"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_collectives_blas1. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1 localhost 0 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1 localhost 3 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1.3 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1 localhost 1 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1 localhost 2 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1.3 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Uses the same infrastructure to initialise the BSP1D"
		echo "                                 implementation of the GraphBLAS and test blas1 grb::collectives"
		echo " "
		echo "Functional test executable: bin/tests/manual_hook_grb_collectives_blas1_raw. Script hardcodes test for"
		echo "four separate processes running on and connecting to localhost on port 77770."
		bash -c "${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1_raw localhost 0 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1_raw.0 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1_raw localhost 3 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1_raw.3 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1_raw localhost 1 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1_raw.1 & \
			${MANUALRUN} bin/tests/manual_hook_grb_collectives_blas1_raw localhost 2 4 77770 &> bin/tests/output/manual_hook_grb_collectives_blas1_raw.2 & \
			wait"
		(grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1_raw.1 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1_raw.2 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1_raw.3 && grep -q 'Test OK' bin/tests/output/manual_hook_grb_collectives_blas1_raw.0 && printf "Test OK.\n\n") || (printf "Test FAILED.\n\n")
			
		echo ">>>      [x]           [ ]       Testing dense vector times matrix using the double (+,*)"
		echo "                                 semiring where matrix elements are doubles and vector"
		echo "                                 elements ints. The input matrix is taken from west0497."
		if [ -f datasets/west0497.mtx ]; then
			${LPFRUN} -np 1 bin/tests/automatic_launch_vxm datasets/west0497.mtx &> bin/tests/output/automatic_launch_vxm.west0497.P1
			${LPFRUN} -np 2 bin/tests/automatic_launch_vxm datasets/west0497.mtx &> bin/tests/output/automatic_launch_vxm.west0497.P2
			${LPFRUN} -np 3 bin/tests/automatic_launch_vxm datasets/west0497.mtx &> bin/tests/output/automatic_launch_vxm.west0497.P3
			${LPFRUN} -np 4 bin/tests/automatic_launch_vxm datasets/west0497.mtx &> bin/tests/output/automatic_launch_vxm.west0497.P4
			(grep -q 'Test OK' bin/tests/output/vxm_reference_1_1.west0497 && grep -q 'Test OK' bin/tests/output/automatic_launch_vxm.west0497.P1 && grep -q 'Test OK' bin/tests/output/automatic_launch_vxm.west0497.P2 && grep -q 'Test OK' bin/tests/output/automatic_launch_vxm.west0497.P3 && grep -q 'Test OK' bin/tests/output/automatic_launch_vxm.west0497.P4 && printf "Test OK.\n") || printf "Test FAILED.\n"
			cat bin/tests/output/vxm_reference_1_1.west0497 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n > bin/tests/output/vxm.west0497.chk
			cat bin/tests/output/automatic_launch_vxm.west0497.P1 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_vxm.west0497.P1.chk
			cat bin/tests/output/automatic_launch_vxm.west0497.P2 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_vxm.west0497.P2.chk
			cat bin/tests/output/automatic_launch_vxm.west0497.P3 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_vxm.west0497.P3.chk
			cat bin/tests/output/automatic_launch_vxm.west0497.P4 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_vxm.west0497.P4.chk
			(diff -q bin/tests/output/automatic_launch_vxm.west0497.P1.chk bin/tests/output/vxm.west0497.chk && printf "Verification (1 to serial) OK.\n") || printf "Verification (1 to serial) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_vxm.west0497.P1.chk bin/tests/output/automatic_launch_vxm.west0497.P2.chk && printf "Verification (1 to 2) OK.\n") || printf "Verification (1 to 2) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_vxm.west0497.P1.chk bin/tests/output/automatic_launch_vxm.west0497.P3.chk && printf "Verification (1 to 3) OK.\n") || printf "Verification (1 to 3) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_vxm.west0497.P1.chk bin/tests/output/automatic_launch_vxm.west0497.P4.chk && printf "Verification (1 to 4) OK.\n\n") || printf "Verification (1 to 4) FAILED.\n\n"
		else
			echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
		fi
		echo " "

		echo ">>>      [x]           [ ]       Testing matrix times dense vector using the double (+,*)"
		echo "                                 semiring where matrix elements are doubles and vector"
		echo "                                 elements ints. The input matrix is taken from west0497."
		echo " "
		if [ -f datasets/west0497.mtx ]; then
			${LPFRUN} -np 1 bin/tests/automatic_launch_mxv datasets/west0497.mtx &> bin/tests/output/automatic_launch_mxv.west0497.P1
			${LPFRUN} -np 2 bin/tests/automatic_launch_mxv datasets/west0497.mtx &> bin/tests/output/automatic_launch_mxv.west0497.P2
			${LPFRUN} -np 3 bin/tests/automatic_launch_mxv datasets/west0497.mtx &> bin/tests/output/automatic_launch_mxv.west0497.P3
			${LPFRUN} -np 4 bin/tests/automatic_launch_mxv datasets/west0497.mtx &> bin/tests/output/automatic_launch_mxv.west0497.P4
			(grep -q 'Test OK' bin/tests/output/mxv_reference_1_1.west0497 && grep -q 'Test OK' bin/tests/output/automatic_launch_mxv.west0497.P1 && grep -q 'Test OK' bin/tests/output/automatic_launch_mxv.west0497.P2 && grep -q 'Test OK' bin/tests/output/automatic_launch_mxv.west0497.P3 && grep -q 'Test OK' bin/tests/output/automatic_launch_mxv.west0497.P4 && printf "Test OK.\n") || printf "Test FAILED.\n"
			cat bin/tests/output/mxv_reference_1_1.west0497 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n > bin/tests/output/mxv.west0497.chk
			cat bin/tests/output/automatic_launch_mxv.west0497.P1 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_mxv.west0497.P1.chk
			cat bin/tests/output/automatic_launch_mxv.west0497.P2 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_mxv.west0497.P2.chk
			cat bin/tests/output/automatic_launch_mxv.west0497.P3 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_mxv.west0497.P3.chk
			cat bin/tests/output/automatic_launch_mxv.west0497.P4 | grep '^[0-9][0-9]* [ ]*[-]*[0-9]' | sort -n  > bin/tests/output/automatic_launch_mxv.west0497.P4.chk
			(diff -q bin/tests/output/automatic_launch_mxv.west0497.P1.chk bin/tests/output/mxv.west0497.chk && printf "Verification (1 to serial) OK.\n") || printf "Verification (1 to serial) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_mxv.west0497.P1.chk bin/tests/output/automatic_launch_mxv.west0497.P2.chk && printf "Verification (1 to 2) OK.\n") || printf "Verification (1 to 2) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_mxv.west0497.P1.chk bin/tests/output/automatic_launch_mxv.west0497.P3.chk && printf "Verification (1 to 3) OK.\n") || printf "Verification (1 to 3) FAILED.\n"
			(diff -q bin/tests/output/automatic_launch_mxv.west0497.P1.chk bin/tests/output/automatic_launch_mxv.west0497.P4.chk && printf "Verification (1 to 4) OK.\n\n") || printf "Verification (1 to 4) FAILED.\n\n"
		else
			echo "Test DISABLED: west0497.mtx was not found. To enable, please provide datasets/west0497.mtx"
		fi
		echo " "

		echo ">>>      [x]           [ ]       Testing BSP1D distribution."
		echo " "
		${LPFRUN} -np 1 bin/tests/distribution
	fi
done

echo ">>>      [x]           [ ]       Testing threadlocal storage, parallel, double values,"
echo "                                 including checks for const-correctness."
echo " "
bin/tests/thread_local_storage

echo
echo "*****************************************************************************************"
echo "All unit tests done."
echo " "

