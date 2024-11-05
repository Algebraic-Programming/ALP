#set current directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ALP_ROOT=$( realpath $SCRIPT_DIR/../../ )
CWD=$(pwd)
KERNELNAME=$TNAME
if [[ "$ASCEND_CPU_MODE" == "ON" ]] ; then MODE="cpu"; else MODE="npu"; fi
TARGET=${TNAME}_${MODE}
TARGET_cmake=alp_ascend_${TNAME}_ascend
mkdir -p $CWD/src
hostfile_default="$CWD/src/host_ascend_${TNAME}.cpp"
[ -z ${HOST_TEST_TEMPLATE} ] && HOST_TEST_TEMPLATE=$(pwd)/HOST_TEST_TEMPLATE.cpp
HOST_CODE_INP=$CWD/src/generate_host_code_${TNAME}.inp

bashargn=$#

[ -z ${ASCEND_VERSION} ] &&  echo "ASCEND_VERSION not set"  && exit 1
! [[ "$ASCEND_VERSION" =~ (910A|910B) ]] &&  echo "ASCEND_VERSION possible values: 910A 910B"  && exit 1
[ -z ${ALP_ROOT} ] &&  echo "ALP_ROOT not set"  && exit 1
[ -z ${KERNELNAME} ] &&  echo "KERNELNAME not set"  && exit 1
[ -z ${TARGET} ] &&  echo "TARGET not set"  && exit 1
[ -z ${TARGET_cmake} ] &&  echo "TARGET_cmake not set"  && exit 1
[ -z ${CWD} ] &&  echo "CWD not set"  && exit 1
[ -z ${HOST_TEST_TEMPLATE} ] &&  echo "HOST_TEST_TEMPLATE not set"  && exit 1
[ -z ${HOST_CODE_INP} ] &&  echo "HOST_CODE_INP not set"  && exit 1
if [ -z ${ASCEND_HOME_PATH} ]
then
   trydir01="/usr/local/Ascend/ascend-toolkit/latest"
   if [ -d "$trydir01" ]
   then
      ASCEND_HOME_PATH="$trydir01"
   fi
fi
[ -z ${ASCEND_HOME_PATH} ] &&  echo "ASCEND_HOME_PATH not set"  && exit 1


if [[ "$bashargn" == 2 ]]
then
    #use provided code
    opfile=$1
    hostfile=$2

    opfile=$(realpath $opfile)
    hostfile=$(realpath $hostfile)
    echo "opfile=$opfile"
    echo "hostfile=$hostfile"
else
    #generate the code
    mkdir -p src
    opfile="src/${KERNELNAME}_npu_op.cpp"
    hostfile="$hostfile_default"

    #cleanup any previous output
    mkdir -p bin
    #build ALP code gnerator, i.e. ascend_softmaxOp_ascend executable
    if [ -z "$BUILD_DIR" ]
    then
	echo "BUILD_DIR is not set, create tmp BUILD_DIR in /tmp/build_alp/";
	rm -rf  /tmp/build_alp/
	mkdir /tmp/build_alp/ && cd /tmp/build_alp/ && cmake $ALP_ROOT && make -j$(nproc) $TARGET_cmake && cd $CWD || { echo "codegen build failed" && exit 1; }
	BUILD_DIR=/tmp/build_alp/
    else
	echo "reuse BUILD_DIR";
	mkdir -p $BUILD_DIR
	cd $BUILD_DIR && cmake $ALP_ROOT && make -j$(nproc) $TARGET_cmake && cd $CWD || { echo "codegen build failed" && exit 1; }

    fi

    # make devicecode
    cd src
    make $opfile devicefile="$opfile" target_cmake="$BUILD_DIR/examples/$TARGET_cmake" -f ${CWD}/Makefile || { echo "generate device code failed " && exit 1; }
    cd ..
    ls $opfile || { echo "$opfile not generated" && exit 1; }

    generate_host=$(pwd)/generate_host_code.py

    opfile=$(realpath $opfile)
    hostfile=$(realpath $hostfile)

    # make hostcode
    make $hostfile hostfile="$hostfile" generate_host="$generate_host" target_cmake="$BUILD_DIR/examples/$TARGET_cmake" host_template="$HOST_TEST_TEMPLATE" host_code_inp="$HOST_CODE_INP" ALP_ROOT="$ALP_ROOT" ASCEND_HOME_PATH="$ASCEND_HOME_PATH" ASCEND_VERSION="$ASCEND_VERSION" ASCEND_CPU_MODE="$ASCEND_CPU_MODE" -f ${CWD}/Makefile || { echo "generate host code failed " && exit 1; }

fi



mkdir -p bin
cd bin
make $TARGET target=$TARGET  hostfile="$hostfile" devicefile="$opfile" ALP_ROOT="$ALP_ROOT" ASCEND_HOME_PATH="$ASCEND_HOME_PATH" ASCEND_VERSION="$ASCEND_VERSION" ASCEND_CPU_MODE="$ASCEND_CPU_MODE" -f ${CWD}/Makefile || { echo "ascend build failed" && exit 1; }
cd ../

