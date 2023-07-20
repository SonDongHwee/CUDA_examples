#nvcc -ccbin=mpic++ -L/nfs/home/leejinho/dev/nccl_2018/build/lib -lnccl main.cu
CXX = g++ 
#CXXFLAGS = -g -std=c++11 -Wall -Wextra -Wno-sign-compare -O3
CXXFLAGS = -g -fopenmp -std=c++11 -Wall -Wno-sign-compare -O3

NVCXX = nvcc 
NVCXXFLAGS = -g -std=c++11  -O3


CUDALIB = /usr/local/cuda/lib64
CUDAINC = /usr/local/cuda/include
SRCDIR = src
OBJDIR = obj
CUOBJDIR = cuobj
BINDIR = bin

INCS := $(wildcard $(SRCDIR)/*.h)
SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(SRCS:$(SRCDIR)/%.cc=$(OBJDIR)/%.o)
CUSRCS := $(wildcard $(SRCDIR)/*.cu)
CUOBJS := $(CUSRCS:$(SRCDIR)/%.cu=$(CUOBJDIR)/%.o)

all:  bin/add

bin:
	mkdir -p bin


bin/add: $(OBJS) $(CUOBJS) 
	@echo "OBJ: "$(OBJS)
	@echo "CUOBJ: "$(CUOBJS)
	$(CXX) $^ -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
#$(NVCXX) $(NVCXXFLAGS) -L$(CUDALIB) -lcudart -lcuda -Iinclude -o $@  $^ 
			    @echo "Compiled "$<" successfully!"


.PHONY:	test clean

$(CUOBJS): $(CUOBJDIR)/%.o : $(SRCDIR)/%.cu
	    @echo $(NVCXX) $(NVCXXFLAGS) "-Iinclude -c" $< "-o" $@
	    @$(NVCXX) $(NVCXXFLAGS) -Iinclude -c $< -o $@
			    @echo "CUDA Compiled "$<" successfully!"

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cc
	    @echo $(CXX)  -c $< -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
	    $(CXX)  -c $< -o $@ $(CXXFLAGS) -L$(CUDALIB) -lcudart -Iinclude -I$(CUDAINC) 
			@echo "Compiled "$<" successfully!"

clean:
	rm -f $(OBJS) $(OBJS:%.o=%.d) 
	rm -f $(CUOBJS) $(CUOBJS:%.o=%.d) 
	rm -rf bin/*
