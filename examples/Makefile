
# Makefile for the Ascend examples

.PHONY: all

all:
	ascendcc -b 910B -c -o op.o op.cpp
	ascendcc -I/home/yzelman/Packages/CANN/samples/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/kernel_template/ -b 910B -c -o op_host.o ../examples/ascend_host.cpp
	ascendcc -o main.exe op.o op_host.o
	LD_LIBRARY_PATH=/home/yzelman/Packages/CANN/x86_64/ascend-toolkit/latest/x86_64-linux/lib64/:/home/yzelman/Packages/CANN/x86_64/ascend-toolkit/latest/x86_64-linux/devlib/x86_64/ ./main.exe

