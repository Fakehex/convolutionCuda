CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`

conGray: conGray.cu
	nvcc -o $@ $< $(LDLIBS)

conRgb: conRgb.cu
	nvcc -o $@ $< $(LDLIBS)

conConst: conConst.cu
	nvcc -dc -o $@ $< $(LDLIBS)
.PHONY: clean

clean:
