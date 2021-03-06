# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include makefile.inc

CPU_TARGETS = 
GPU_TARGETS = Benchmarker

default: cpu

all: cpu gpu

cpu: $(CPU_TARGETS)

gpu: $(GPU_TARGETS)

%: %.cpp $(FAISS_HOME)/libfaiss.a 
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS) $(RTFLAGS) -g

clean:
	rm -f $(CPU_TARGETS) $(GPU_TARGETS)

.PHONY: all cpu default gpu clean
