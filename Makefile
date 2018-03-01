################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= convolutionCUDA
# Cuda source files (compiled with cudacc)
CUFILES		:= kernel.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= \
	convolutionCUDA.cpp

################################################################################
# Rules and targets

#include ../../common/common.mk
