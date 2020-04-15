CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

gray: conGray.cu
	nvcc -o $@ $< $(LDLIBS)

conRgb: conRgb.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
