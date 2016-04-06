#ifndef _CHRIS_READS_H
#define _CHRIS_READS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <hdf5.h>

//A quick and dirty multidimensional array
typedef struct{
	unsigned int rank;
	unsigned int* strides;
	double* data;
} multidim;

multidim alloc_multi(unsigned int rank, const unsigned int* strides){
	multidim m;
	m.rank=rank;
	m.strides=(unsigned int*)malloc(rank*sizeof(unsigned int));
	if(!m.strides){
		fprintf(stderr,"Failed to allocate memory");
		exit(1);
	}
	unsigned int size=1;
	for(unsigned int i=0; i<rank; i++){
		m.strides[i]=strides[i];
		size*=strides[i];
	}
	m.data=(double*)malloc(size*sizeof(double));
	if(!m.strides){
		fprintf(stderr,"Failed to allocate memory");
		exit(1);
	}
	return(m);
}

void free_multi(multidim m){
	free(m.strides);
	free(m.data);
}

double* index_multi(multidim m, unsigned int* indices){
	unsigned int idx=0;
	for(unsigned int i=0; i<m.rank; i++){
		idx*=m.strides[i];
		idx+=indices[i];
	}
	return(m.data+idx);
}

//Read a dataset into a buffer, asusming that the allocated size is correct.
//If anything goes wrong just bail out.
void readDataSet(hid_t container_id, const char* path, double* targetBuffer){
	hid_t dataset_id = H5Dopen(container_id, path, H5P_DEFAULT);
	herr_t status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, targetBuffer);
	if(status<0){
		fprintf(stderr,"Failed to read dataset '%s'\n",path);
		exit(1);
	}
	H5Dclose(dataset_id);
}

void readDoubleAttr(hid_t container_id, const char* path, const char* name, double* targetBuffer){
	hid_t attr_id = H5Aopen_by_name(container_id, path, name, H5P_DEFAULT, H5P_DEFAULT);
	herr_t status = H5Aread(attr_id, H5T_NATIVE_DOUBLE, targetBuffer);
	if(status<0){
		fprintf(stderr,"Failed to read attribute '%s::%s'\n",path,name);
		exit(1);
	}
	H5Aclose(attr_id);
}

#endif
