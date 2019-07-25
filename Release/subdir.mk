################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../ExperimentManager.cu \
../maintest.cu 

CPP_SRCS += \
../main.cpp 

OBJS += \
./ExperimentManager.o \
./main.o \
./maintest.o 

CU_DEPS += \
./ExperimentManager.d \
./maintest.d 

CPP_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -m64 -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_62,code=compute_62 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_72,code=compute_72 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -m64 -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -O3 --compile -m64  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


