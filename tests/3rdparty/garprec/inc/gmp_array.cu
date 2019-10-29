#ifndef __GMP_ARRAY_CU__
#define __GMP_ARRAY_CU__

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include "gmp_array.h"
#include "cuda_header.cu"
#include "garprec.cu"

/* macro utilities */
#define ERROR_EXIT exit(EXIT_FAILURE)
inline void errorExit(const char* func, const char* msg) {
        printf("!!!ERROR@%s: %s\n", func, msg);
        ERROR_EXIT;
}

/**
* define the helper data structure gmp_array
*/

/*
struct gmp_array {

        int n_words;                    //number of words allocated for each element
        int numElement;                 //the maximum number of high precision numbers in the array
        int size;                       //total number of words allocated in this array 
        bool isAllocated;
        double* d_mpr;
        
        //allocate the array on the device, and initialized as 0
        gmp_array( const int n_words, const int numElement );
        //copy the value in the mp_real array to this gmp_array
        gmp_array( const mp_real* mp_array, const int numElement );
        gmp_array( const int n_words, const int numElement, bool isSeq );
        gmp_array( const mp_real* mp_array, const int numElement, bool isSeq );

        //copy the values in the mp_array to this gmp_array
        void toGPU( const mp_real* mp_array, const int numElment );
        void toGPU_seq( const mp_real* mp_array, const int numElement );
        void toGPU( const mp_real* mp_array1, const int numElement1,
                    const mp_real* mp_array2, const int numElement2 ); //merge two arrays

        //copy the values from the device to the mp_array on the host
        void fromGPU( mp_real* mp_array, const int numElement );
        void fromGPU_seq( mp_real* mp_array, const int numElement );
        void fromGPU( mp_real* mp_array1, const int numElement1,
                      mp_real* mp_array2, const int numElement2 ); //copy to two arrays

        //reset the memory, initialized to 0
        void reset( const int n_words, const int numElement );
        
        //release the GPU memory
        void release();
};
*/

__global__
void gmp_init_kernel( double* d_mpr, const int numElement, const int n_words ) {
        const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
        const unsigned int delta = blockDim.x*gridDim.x;

        for( int i = index; i < numElement; i += delta ) {
                d_mpr[i] = (double)n_words;
                d_mpr[numElement + i] = 0.0;
        }
}


__global__
void gmp_init_seq_kernel( double* d_mpr, const int numElement, const int n_words ) {
        const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
        const unsigned int delta = blockDim.x*gridDim.x;

        for( int i = index; i < numElement; i += delta ) {
                d_mpr[i*n_words] = (double)n_words;
                d_mpr[i*n_words + 1] = 0.0;
        }
}


void gmparray_init( double* d_a, const int numElement, const int n_words ) {

        cutilSafeCall(cudaMemset( d_a, sizeof(double)*numElement*n_words, 0 ) );
        gmp_init_kernel<<<256, 256>>>( d_a, numElement, n_words );
}

void gmparray_init_seq( double* d_a, const int numElement, const int n_words ) {

        cutilSafeCall( cudaMemset( d_a, sizeof(double)*numElement*n_words, 0 ) );
        gmp_init_seq_kernel<<<256, 256>>>( d_a, numElement, n_words );
        //cutilCheckMsg( "gmp_init_kernel" );
}



gmp_array::gmp_array( const int n_words, const int numElement, bool isCoalesced) {
	
	if(isCoalesced) {
	        this->n_words = n_words;
       		this->numElement = numElement;
	        this->size = n_words*numElement;
	        this->interval = numElement;

	        GPUMALLOC( (void**)&d_mpr, sizeof(double)*size );
	        this->isAllocated = true;
	        gmparray_init( d_mpr, numElement, n_words );
	} else {
	
	        this->n_words = n_words;
        	this->numElement = numElement;
       		this->size = n_words*numElement;
		this->interval = 1;

        	GPUMALLOC( (void**)&d_mpr, sizeof(double)*size );
        	this->isAllocated = true;
        	gmparray_init_seq( d_mpr, numElement, n_words );
	}
}

gmp_array::gmp_array( const mp_real* mp_array, const int in_numElement, bool isCoalesced ) {
	if(isCoalesced) {
	        /// check n_words 
	        this->n_words = (int)(mp_array[0].mpr[0]);
	        for( int i = 0; i < in_numElement; i++ ) {
	                if( this->n_words != (int)(mp_array[i].mpr[0]) ) {
       		                errorExit( "gmp_array::gmp_array", "n_words are not the same in this mp array" );
	                }
	        }
	        this->numElement = in_numElement;
       		this->size = n_words*in_numElement;
       		this->interval = in_numElement;

       		///allocate the memory on the device
        	GPUMALLOC( (void**)&d_mpr, sizeof(double)*size );
        	this->isAllocated = true;

        	///copy the mp_array from host to device
        	toGPU( mp_array, in_numElement );
	} else {
		/// check n_words 
		this->n_words = (int)(mp_array[0].mpr[0]);
		for( int i = 0; i < in_numElement; i++ ) {
			if( this->n_words != (int)(mp_array[i].mpr[0]) ) {
				errorExit( "gmp_array::gmp_array", "n_words are not the same in this mp array" );
			}
		}
		this->numElement = in_numElement;
		this->size = n_words*in_numElement;
		this->interval = 1;
	
		///allocate the memory on the device
		GPUMALLOC( (void**)&d_mpr, sizeof(double)*size );
		this->isAllocated = true;
	
		///copy the mp_array from host to device
		toGPU_seq( mp_array, numElement );
	}
}

void gmp_array::toGPU( const mp_real* mp_array, const int in_numElement ) {
	///check whether the memory is already allocated on the GPU
	if( !isAllocated ) {
		errorExit( "gmp_array::toGPU", "memory on the device is not allocated" );
	}
	
	///check the n_words
	this->n_words = (int)(mp_array[0].mpr[0]);
	this->numElement = in_numElement;
	this->interval = in_numElement;
		
	/// check the memory size requirement
	if( this->size < n_words*numElement ) {
		errorExit( "gmp_array::toGPU", "the allocated memory is not enought for this mp_array" );
	}
	
	///now ready to copy the values to GPU, use only one host to device memory copy
	double* h_buf = (double*)malloc( sizeof(double)*n_words*numElement );
	for( int i = 0; i < this->numElement; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			h_buf[word*numElement + i] = mp_array[i].mpr[word];
		}
	}	
	cudaMemcpy( d_mpr, h_buf, sizeof(double)*numElement*n_words, cudaMemcpyHostToDevice );	

	///free memory
	free( h_buf );	
}

void gmp_array::toGPU_seq( const mp_real* mp_array, const int in_numElement ) {
	///check whether the memory is already allocated on the GPU
	if( !isAllocated ) {
		errorExit( "gmp_array::toGPU", "memory on the device is not allocated" );
	}
	
	///check the n_words
	this->n_words = (int)(mp_array[0].mpr[0]);
	this->numElement = in_numElement;
	this->interval = 1;
		
	/// check the memory size requirement
	if( this->size < n_words*numElement ) {
		errorExit( "gmp_array::toGPU", "the allocated memory is not enought for this mp_array" );
	}
	
	///now ready to copy the values to GPU, use only one host to device memory copy
	double* h_buf = (double*)malloc( sizeof(double)*n_words*numElement );
	for( int i = 0; i < this->numElement; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			h_buf[i*n_words + word] = mp_array[i].mpr[word];
		}
	}	
	cudaMemcpy( d_mpr, h_buf, sizeof(double)*numElement*n_words, cudaMemcpyHostToDevice );	

	///free memory
	free( h_buf );	
}

void gmp_array::toGPU( const mp_real* mp_array1, const int numElement1, 
		       const mp_real* mp_array2, const int numElement2 ) {
        
	//check memory allocation
	if( !isAllocated ) {
                errorExit( "gmp_array::toGPU", "memory on the device is not allocated" );
        }

	//check n_words
	if( mp_array1[0].mpr[0] != mp_array2[0].mpr[0] ) {
		errorExit( "gmp_array::toGPU", "n_words are different in two input arrays" );
	}
	this->n_words = mp_array1[0].mpr[0];
	this->numElement = numElement1 + numElement2;	

	//check memory size
	if( this->size < (this->n_words*this->numElement) ) {
		errorExit( "gmp_array::toGPU", "the allocated memory is not enough for these two mp_array" );
	}

	//now ready to copy the mp_array from host to device
	double* h_buf = (double*)malloc( sizeof(double)*this->n_words*this->numElement );
	for( int i = 0; i < numElement1; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			h_buf[word*this->numElement + i] = mp_array1[i].mpr[word];
		}
	}
	for( int i = 0; i < numElement2; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			h_buf[word*this->numElement + numElement1 + i] = mp_array2[i].mpr[word];
		}
	}
	cudaMemcpy( d_mpr, h_buf, sizeof(double)*this->n_words*this->numElement, cudaMemcpyHostToDevice );

	//free
	free( h_buf );	

}

void gmp_array::fromGPU(mp_real* mp_array, const int out_numElement) {
	if(this->interval == 1) {
		this->fromGPU_seq(mp_array, numElement);
	} else {
		this->fromGPU_interval(mp_array, numElement);
	}
}

void gmp_array::fromGPU_interval( mp_real* mp_array, const int out_numElement ) {
	///check numElement
	if( this->numElement != out_numElement ) {
		errorExit( "gmp_array::fromGPU_interval", "ERROR: numElement on the device is different from the target mp_array" );
	}
	
	///check n_words
	//printf("n words %i mpr words %i\n", this->n_words, mp_array[0].mpr[0]);
	if( this->n_words != mp_array[0].mpr[0] ) {
		errorExit( "gmp_array::fromGPU", "ERROR: n_words on the device is different from the target mp_array" );
	}
	
	///now ready to copy from the device to host using only one device-host memory copy
	double* h_buf = (double*)malloc( sizeof(double)*n_words*numElement );
	cudaMemcpy( h_buf, d_mpr, sizeof(double)*n_words*numElement, cudaMemcpyDeviceToHost );
	for( int i = 0; i < numElement; i++ ) {
		for( int word = 0; word < n_words; word++ ) {
			mp_array[i].mpr[word] = h_buf[word*numElement + i];
		}
	}

	//free
	free( h_buf );
}

void gmp_array::fromGPU_seq( mp_real* mp_array, const int out_numElement ) {
	///check numElement
	if( this->numElement != out_numElement ) {
		errorExit( "gmp_array::fromGPU_sequence", "ERROR: numElement on the device is different from the target mp_array" );
	}

	//check the interval
	if(this->interval != 1) {
		errorExit("gmp_array::fromGPU", "ERROR: the memory layout is not sequential.");
	}
	
	///check n_words
	if( this->n_words != mp_array[0].mpr[0] ) {
		errorExit( "gmp_array::fromGPU", "ERROR: n_words on the device is different from the target mp_array" );
	}
	
	///now ready to copy from the device to host using only one device-host memory copy
	double* h_buf = (double*)malloc( sizeof(double)*n_words*numElement );
	cudaMemcpy( h_buf, d_mpr, sizeof(double)*n_words*numElement, cudaMemcpyDeviceToHost );
	for( int i = 0; i < numElement; i++ ) {
		for( int word = 0; word < n_words; word++ ) {
			mp_array[i].mpr[word] = h_buf[i*n_words + word];
		}
	}

	//free
	free( h_buf );
}

void gmp_array::fromGPU( mp_real* mp_array1, const int numElement1, 
			 mp_real* mp_array2, const int numElement2 ) {
	//check numElement
	if( (numElement1 + numElement2) != this->numElement ) {
		errorExit( "gmp_array::fromGPU", "ERROR: numElement on the device is different from the two target mp_array's" );
	}
		
	//check n_words
	if( (this->n_words != mp_array1[0].mpr[0]) || (this->n_words != mp_array2[0].mpr[0]) ) {
		errorExit( "gmp_array::fromGPU", "ERROR: n_words on the device is different from the target mp_array" );
	}

	//now ready to copy
	double* h_buf = (double*)malloc( sizeof(double)*this->n_words*this->numElement );
	cudaMemcpy( h_buf, d_mpr, sizeof(double)*this->n_words*this->numElement, cudaMemcpyDeviceToHost );
	for( int i = 0; i < numElement1; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			mp_array1[i].mpr[word] = h_buf[word*this->numElement + i];
		}
	}
	for( int i = 0; i < numElement2; i++ ) {
		for( int word = 0; word < this->n_words; word++ ) {
			mp_array2[i].mpr[word] = h_buf[word*this->numElement + numElement1 + i]; 
		}
	}

	//free
	free( h_buf );
}


void gmp_array::reset( const int new_nwords, const int new_numElement ) {
	///check memory
	if( this->size < new_nwords*new_numElement ) {
		errorExit( "gmp_array::reset", "memory size in the array is too small" );
	}

	///set info.
	this->n_words = new_nwords;
	this->numElement = new_numElement;
	this->interval = new_numElement;

	///init using kernel
	gmparray_init( d_mpr, numElement, n_words );
}

void gmp_array::release() {
	if( isAllocated ) {
		GPUFREE( d_mpr );
		d_mpr = NULL;
		isAllocated = false;
		n_words = 0;
		numElement = 0;
		size = 0;
		interval = 0;
	}
}



#endif /* __GMP_ARRAY_CU__ */
