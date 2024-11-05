ASCEND_TOOLKIT_INSTALL_PATH=$(ASCEND_HOME_PATH)

CXX=/usr/bin/c++
ccec_compiler=$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/ccec_compiler/bin/ccec

ifeq ($(ASCEND_VERSION),910A)
	ccec_falgs=-I$(ALP_ROOT)/include/graphblas/ascend/ -I$(ALP_ROOT)/include/asclib -xcce -DTILING_KEY_VAR=0  -I$(ASCEND_TOOLKIT_INSTALL_PATH)/acllib/include -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw/impl -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw/interface -I$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/tikicpulib/lib/include  -O2 -std=c++17 --cce-aicore-arch=dav-c100  --cce-auto-sync -fPIC -pthread
	ccec_link_falgs=--cce-fatobj-link --cce-aicore-arch=dav-c100 -L$(ASCEND_TOOLKIT_INSTALL_PATH)/runtime/lib64 -L$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/simulator/Ascend910A/lib -L$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/tikicpulib/lib/Ascend910A -lstdc++ -lruntime -lascendcl -lm
endif

ifeq ($(ASCEND_VERSION),910B)
	ccec_falgs=-I$(ALP_ROOT)/include/graphblas/ascend/ -I$(ALP_ROOT)/include/asclib -xcce -DTILING_KEY_VAR=0  -I$(ASCEND_TOOLKIT_INSTALL_PATH)/acllib/include -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw/impl -I$(ASCEND_TOOLKIT_INSTALL_PATH)/compiler/tikcpp/tikcfw/interface -I$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/tikicpulib/lib/include  -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec -mllvm -cce-aicore-function-stack-size=16000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform  --cce-auto-sync -fPIC -pthread -DASCEND910B
	ccec_link_falgs=--cce-fatobj-link --cce-aicore-arch=dav-c220-vec -L$(ASCEND_TOOLKIT_INSTALL_PATH)/runtime/lib64 -L$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/simulator/Ascend910B1/lib -L$(ASCEND_TOOLKIT_INSTALL_PATH)/tools/tikicpulib/lib/Ascend910B1 -lstdc++ -lruntime -lascendcl -lm -DASCEND910B
endif

MODE=npu

ifeq ($(ASCEND_CPU_MODE),ON)
ifeq ($(ASCEND_VERSION),910A)
MODE=cpu
ccec_compiler=$(CXX)

ccec_falgs=-I$(ALP_ROOT)/include/graphblas/ascend/ -I$(ALP_ROOT)/include/asclib -D__CCE_AICORE__=100 -D__CCE_KT_TEST__=1 -D__DAV_C100__  -I${ASCEND_HOME_PATH}/acllib/include -isystem ${ASCEND_HOME_PATH}/tools/tikicpulib/lib/include -isystem ${ASCEND_HOME_PATH}/tools/tikicpulib/../../compiler/tikcpp/tikcfw -isystem ${ASCEND_HOME_PATH}/tools/tikicpulib/../../compiler/tikcpp/tikcfw/impl -isystem ${ASCEND_HOME_PATH}/tools/tikicpulib/../../compiler/tikcpp/tikcfw/interface -std=gnu++1z -g -std=c++17

ccec_link_falgs=-L${ASCEND_HOME_PATH}/tools/tikicpulib/lib  -L${ASCEND_HOME_PATH}/tools/tikicpulib/lib/Ascend910A  -L${ASCEND_HOME_PATH}/tools/tikicpulib/../simulator/Ascend910A/lib  -L${ASCEND_HOME_PATH}/tools/tikicpulib/../../lib64  -Wl,-rpath,${ASCEND_HOME_PATH}/tools/tikicpulib/lib:${ASCEND_HOME_PATH}/tools/tikicpulib/lib/Ascend910A:${ASCEND_HOME_PATH}/tools/tikicpulib/../simulator/Ascend910A/lib:${ASCEND_HOME_PATH}/tools/tikicpulib/../../lib64 -lascendcl -Wl,--no-as-needed -l_pvmodel -ltikcpp_debug ${ASCEND_HOME_PATH}/tools/tikicpulib/lib/libtikicpulib_cceprint.so ${ASCEND_HOME_PATH}/tools/tikicpulib/lib/libtikicpulib_npuchk.so ${ASCEND_HOME_PATH}/tools/tikicpulib/lib/libtikicpulib_stubreg.so -Wl,--as-needed

endif
endif

$(hostfile): $(target_cmake) $(host_template) $(host_code_inp) $(generate_host)
	@echo python3 $(generate_host) $(host_template) $(hostfile) $(host_code_inp) 10 0 8
	@python3 $(generate_host) $(host_template) $(hostfile) $(host_code_inp) 10 0 8

$(host_code_inp): $(target_cmake)
	@echo @$(target_cmake)
	@$(target_cmake)

$(devicefile): $(target_cmake)
	@$(target_cmake)

$(basename $(notdir $(hostfile))).$(MODE).o : $(hostfile)
	@echo $(ccec_compiler) $(ccec_falgs) -c $(hostfile) -o $(basename $(notdir $(hostfile))).$(MODE).o
	$(ccec_compiler) $(ccec_falgs) -c $(hostfile) -o $(basename $(notdir $(hostfile))).$(MODE).o

$(basename $(notdir $(devicefile))).$(MODE).o : $(devicefile)
	@echo $(ccec_compiler) $(ccec_falgs) -c $(devicefile) -o $(basename $(notdir $(devicefile))).$(MODE).o
	$(ccec_compiler) $(ccec_falgs) -c $(devicefile) -o $(basename $(notdir $(devicefile))).$(MODE).o

$(target): $(basename $(notdir $(devicefile))).$(MODE).o $(basename $(notdir $(hostfile))).$(MODE).o
	$(ccec_compiler) $(basename $(notdir $(devicefile))).$(MODE).o $(basename $(notdir $(hostfile))).$(MODE).o -o $(target) $(ccec_link_falgs)

clean:
	@rm -f $(basename $(notdir $(devicefile))).$(MODE).o $(basename $(notdir $(hostfile))).$(MODE).o $(target)

cleanall:
	@rm -f .*o
