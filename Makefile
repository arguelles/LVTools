# Compiler
CC=clang
CXX=clang++
AR=ar
LD=clang++

DYN_SUFFIX=.dylib
DYN_OPT=-dynamiclib -install_name $(LIBnuSQUIDS)/$(DYN_PRODUCT) -compatibility_version $(VERSION) -current_version $(VERSION)

VERSION=1.0.0
PREFIX=/usr/local


#PATH_nuSQUIDS=$(shell pwd)
PATH_nuSQUIDS=/usr/local
PATH_SQUIDS=$(SQUIDS_DIR)

MAINS_SRC=$(wildcard mains/*.cpp)
MAINS=$(patsubst mains/%.cpp,bin/%.exe,$(MAINS_SRC))
#$(EXAMPLES_SRC:.cpp=.exe)

CXXFLAGS= -g -std=c++11 -I./inc

# Directories

GSL_CFLAGS=-I/usr/local/Cellar/gsl/1.16/include
GSL_LDFLAGS=-L/usr/local/Cellar/gsl/1.16/lib -lgsl -lgslcblas -lm
HDF5_CFLAGS=-I/usr/local/Cellar/hdf5/1.8.15//include
HDF5_LDFLAGS=-L/usr/local/Cellar/hdf5/1.8.15/lib -L/usr/local/opt/szip/lib -lhdf5_hl -lhdf5 -lsz -lz -ldl -lm
SQUIDS_CFLAGS=-I/usr/local/include -I/usr/local/Cellar/gsl/1.16/include
SQUIDS_LDFLAGS=-L/usr/local/lib -L/usr/local/Cellar/gsl/1.16/lib -lSQuIDS -lgsl -lgslcblas -lm
PHYSTOOLS_LDFLAGS=-lPhysTools


INCnuSQUIDS=$(PATH_nuSQUIDS)/inc
LIBnuSQUIDS=$(PATH_nuSQUIDS)/lib

# FLAGS
CFLAGS= -O3 -fPIC -I$(INCnuSQUIDS) $(SQUIDS_CFLAGS) $(GSL_CFLAGS) $(HDF5_CFLAGS)
LDFLAGS= -Wl,-rpath -Wl,$(LIBnuSQUIDS) -L$(LIBnuSQUIDS)
LDFLAGS+= $(SQUIDS_LDFLAGS) $(GSL_LDFLAGS) $(HDF5_LDFLAGS) $(PHYSTOOLS_LDFLAGS)

# Compilation rules
all: $(MAINS)

bin/%.exe : mains/%.cpp mains/%.o mains/lbfgsb.o mains/linpack.o
	$(CXX) $(CXXFLAGS) $(CFLAGS) $< mains/lbfgsb.o mains/linpack.o $(LDFLAGS) -lnuSQuIDS -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $(CFLAGS) $< -o $@

mains/lbfgsb.o : ./inc/lbfgsb/lbfgsb.h ./inc/lbfgsb/lbfgsb.c
	$(CC) $(CFLAGS) ./inc/lbfgsb/lbfgsb.c -c -o ./mains/lbfgsb.o

mains/linpack.o : ./inc/lbfgsb/linpack.c
	$(CC) $(CFLAGS) ./inc/lbfgsb/linpack.c -c -o ./mains/linpack.o

.PHONY: clean
clean:
	rm -rf ./mains/*.exe ./bin/* ./mains/*.o

