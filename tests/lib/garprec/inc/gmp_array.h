#ifndef __GMP_ARRAY_H__
#define __GMP_ARRAY_H__


#include <arprec/mp_real.h>

struct gmp_array {

        unsigned int n_words;                    //number of words allocated for each element
        unsigned int numElement;                 //the maximum number of high precision numbers in the array
        unsigned int size;                       //total number of words allocated in this array
	    unsigned int interval;
        bool isAllocated;
        double* d_mpr;

        //allocate the array on the device, and initialized as 0
        gmp_array( const int n_words, const int numElement, bool isCoalesced = true );
        gmp_array( const mp_real* mp_array, const int numElement, bool isCoalesced = true );

        //copy the values in the mp_array to this gmp_array
        void toGPU( const mp_real* mp_array, const int numElment );
        void toGPU_seq( const mp_real* mp_array, const int numElement );
        void toGPU( const mp_real* mp_array1, const int numElement1,
                    const mp_real* mp_array2, const int numElement2 ); //merge two arrays

        //copy the values from the device to the mp_array on the host
        void fromGPU( mp_real* mp_array, const int numElement );
	    void fromGPU_interval(mp_real* mp_array, const int numElement);
        void fromGPU_seq( mp_real* mp_array, const int numElement );
        void fromGPU( mp_real* mp_array1, const int numElement1,
                      mp_real* mp_array2, const int numElement2 ); //copy to two arrays

        //reset the memory, initialized to 0
        void reset( const int n_words, const int numElement );

        //release the GPU memory
        void release();
};

#endif /* __GMP_ARRAY_H__ */


