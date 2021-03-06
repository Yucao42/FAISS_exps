# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modify this to where FAISS is installed.
FAISS_HOME   = ${HOME}/faiss
CUDA_HOME = /usr/local/cuda-10.0

# If using cims machine
ifeq (${HOME},/home/yc3390)
	FAISS_HOME   = ${HOME}/Documents/faiss/faiss
	CUDA_HOME = /usr/local/stow/cuda-10.0
	# FAISS_HOME   = ${HOME}/Documents/faiss_time/faiss
endif

# FAISS_HOME   = ${HOME}/Documents/faiss_time/faiss

CXX          = g++ -std=c++11
CXXCPP       = g++ -std=c++11 -E
CPPFLAGS     = -DFINTEGER=int  -fopenmp -I${CUDA_HOME}/include  -g \
-I${FAISS_HOME} -I${FAISS_HOME}/.. -lfaiss -L$(FAISS_HOME)
CXXFLAGS     = -fPIC -m64 -Wno-sign-compare -g -O3 -Wall -Wextra
CPUFLAGS     = -mpopcnt -msse4
LDFLAGS      = -fopenmp  -L${CUDA_HOME}/lib64
LIBS         = -lopenblas   -lcudart -lcublas
PYTHONCFLAGS =   -I${HOME}/anaconda3/include
SWIGFLAGS    = -DSWIGWORDSIZE64
RTFLAGS      = -Wl,-rpath=${FAISS_HOME}

NVCC         = ${CUDA_HOME}/bin/nvcc
CUDA_ROOT    = ${CUDA_HOME}
CUDA_ARCH    = -gencode=arch=compute_35,code=compute_35 \
-gencode=arch=compute_52,code=compute_52 \
-gencode=arch=compute_60,code=compute_60 \
-gencode=arch=compute_61,code=compute_61 \
-gencode=arch=compute_70,code=compute_70 \
-gencode=arch=compute_75,code=compute_75
NVCCFLAGS    = -I $(CUDA_ROOT)/targets/x86_64-linux/include/ \
-Xcompiler -fPIC \
-Xcudafe --diag_suppress=unrecognized_attribute \
$(CUDA_ARCH) \
-lineinfo \
-ccbin $(CXX) -DFAISS_USE_FLOAT16

OS = $(shell uname -s)

SHAREDEXT   = so
SHAREDFLAGS = -shared

ifeq ($(OS),Darwin)
	SHAREDEXT   = dylib
	SHAREDFLAGS = -dynamiclib -undefined dynamic_lookup
        SWIGFLAGS   =
endif

MKDIR_P      = /usr/bin/mkdir -p
PYTHON       = python3 
SWIG         = swig
AR          ?= ar
