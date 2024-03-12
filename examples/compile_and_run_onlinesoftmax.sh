#set current directory  
CWD=$(pwd)

bashargn=$#
#echo "bashargn  $bashargn"
if [[ "$bashargn" == 2 ]]
then
    opfile=$1
    hostfile=$2
else
    opfile="op.cpp"
    hostfile="$CWD/onlinesoftmax_custom_main.cpp"

    #cleanup any previous output 
    rm -f a_npu input/*.bin output/output_z.bin *.o op.cpp
    rm -rf  /tmp/build_alp/

    #build ALP code gnerator, i.e. ascend_onlinesoftmaxOp_ascend executable
    mkdir /tmp/build_alp/ && cd /tmp/build_alp/ && cmake $CWD/../ && make ascend_onlinesoftmaxOp_ascend && cd $CWD

    #run ALP code generator generate, store it into op.cpp
    /tmp/build_alp/examples/./ascend_onlinesoftmaxOp_ascend > op.cpp

    cat op.cpp
fi

echo "compile: $opfile and $hostfile"

#compile ascend code
# set the compiler path and the ASCEND_TOOLKIT_INSTALL_PATH
ASCEND_TOOLKIT_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
ccec_compiler="/home/HwHiAiUser/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin/ccec"

#compile generated kernel code, i.e. $opfile 
$ccec_compiler -xcce -DTILING_KEY_VAR=0  -I"$ASCEND_TOOLKIT_INSTALL_PATH/acllib/include" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw/impl" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw/interface" -I"$ASCEND_TOOLKIT_INSTALL_PATH/tools/tikicpulib/lib/include"  -O2 -std=c++17 --cce-aicore-arch=dav-c100  --cce-auto-sync -fPIC -pthread -o $opfile.o -c $opfile

#compile template host code, i.e. $hostfile
$ccec_compiler -xcce -DTILING_KEY_VAR=0  -I"$ASCEND_TOOLKIT_INSTALL_PATH/acllib/include" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw/impl" -I"$ASCEND_TOOLKIT_INSTALL_PATH/compiler/tikcpp/tikcfw/interface" -I"$ASCEND_TOOLKIT_INSTALL_PATH/tools/tikicpulib/lib/include" -O2 -std=c++17 --cce-aicore-arch=dav-c100  --cce-auto-sync -fPIC -pthread -o $hostfile.o -c $hostfile

#link the executable, i.e. a_npu
$ccec_compiler --cce-fatobj-link --cce-aicore-arch=dav-c100 $opfile.o $hostfile.o -o a_npu -L"$ASCEND_TOOLKIT_INSTALL_PATH/runtime/lib64" -L"$ASCEND_TOOLKIT_INSTALL_PATH/tools/simulator/Ascend910A/lib" -L"$ASCEND_TOOLKIT_INSTALL_PATH/tools/tikicpulib/lib/Ascend910A" -lstdc++ -lruntime -lascendcl

rm -f runtime*.csv
rm -rf input output
echo "generate input"
echo "python3 onlinesoftmax_custom.py"
mkdir -p input
mkdir -p output
python3 onlinesoftmax_custom.py

#run ascend example, run ./a_npu on 910
echo "run ascend example"
echo "./a_npu ${vec_length}"
./a_npu #${vec_length}

#python3 onlinesoftmax_print.py
python3 softmax_check-v5.py
#echo "NO onlinesoftmax_custom.py"



