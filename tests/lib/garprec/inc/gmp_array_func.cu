#ifndef __GMP_ARRAY_FUNC_CU__
#define __GMP_ARRAY_FUNC_CU__

#include "gmp_array_func.h"
#include "garprec.cu"

__global__ void
gmparray_add_kernel(const double *d_a, double *d_b, double *d_c,
                    const int interval, const int numElement, const int prec_words) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for (int i = index; i < numElement; i += delta) {
        gmpadd(d_a + i, interval, d_b + i, interval, d_c + i, interval, prec_words, d);
    }
}


__global__ void
gmparray_mul_kernel(const double *d_a, double *d_b, double *d_c,
                    const int interval, const int numElement, const int prec_words) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    for (int i = index; i < numElement; i += delta) {
        gmpmul(d_a + i, interval, d_b + i, interval, d_c + i, interval, prec_words, d);
    }
}

void gmparray_add(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c,
                  unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;
    if (interval > 1) {
        //printf("interval memory layout...\n");
        gmparray_add_kernel << < numBlock, numThread >> > (d_a->d_mpr, d_b->d_mpr, d_c->d_mpr, interval, numElement, prec_words);
    } else {
        printf("!!!!sequential memory layout ...\n");
    }
    //cutilCheckMsg("gmparray_add_kernel");
    cutilSafeCall(cudaThreadSynchronize());
}

void gmparray_mul(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c,
                  unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;
//    printf("numElement = %d, interval = %d\n", numElement, interval);
//    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    if (interval > 1) {
        //printf("interval memory layout...\n");
        gmparray_mul_kernel << < numBlock, numThread >> >
                                           (d_a->d_mpr, d_b->d_mpr, d_c->d_mpr, interval, numElement, prec_words);
    } else {
        printf("!!!!sequential memory layout ...\n");
    }
    //cutilCheckMsg("gmparray_mul_kernel");
    cutilSafeCall(cudaThreadSynchronize());
}

/*
void gmparray_div_device(gmp_array *d_a, gmp_array *d_b, gmp_array *d_c,
                         unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;
    printf("gmparray_div_device...\n");
    printf("numElement = %d, interval = %d\n", numElement, interval);
    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);

    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * MAX_D_SIZE * numBlock * numThread);

    if (interval > 1) {
        printf("interval memory layout...\n");
        gmparray_div_device_kernel << < numBlock, numThread >> >
                                                  (d_a->d_mpr, d_b->d_mpr, d_c->d_mpr, d_buf, interval, numElement, prec_words);
    } else {
        printf("sequential memory layout ...\n");
    }
    //cutilCheckMsg("gmparray_div_kernel");
    cutilSafeCall(cudaThreadSynchronize());

    GPUFREE(d_buf);
}
*/
/////////////////////////////////////////////////////////////////////////////////////////////////////
// the test code

/*
__global__ void
gmparray_exp_alldevice_seq_kernel(const double *d_a, double *d_b, const int interval,
                                  const int numElement, const int prec_words,
                                  double *d_d, double *sk0, double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    const int n_words = prec_words + 5;
    double *d_buf = d_d + index * (prec_words + 9);

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i * n_words, 1, d_b + i * n_words, 1, prec_words, d_buf, 1,
               sk0 + index * (prec_words + 6), sk1 + index * (prec_words + 6),
               sk2 + index * (prec_words + 6), sk3 + index * (prec_words + 6), 1);
    }
}

__global__
void gmparray_exp_alldevice_kernel(const double *d_a, double *d_b, const int interval,
                                   const int numElement, const int prec_words,
                                   double *d_buf, double *sk0, double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double *d = d_buf + index;

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words, d, delta,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index, delta);
    }
}


__global__
void gmparray_log_alldevice_kernel(const double *d_a, double *d_b, const int interval,
                                   const int numElement, const int prec_words,
                                   double *d_buf, double *sk0, double *sk1,
                                   double *sk2, double *sk3, double *sk4) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];

    for (int i = index; i < numElement; i += delta) {
        //gmplog( d_a + i, interval, d_b + i, interval, prec_words,
        //	  d_buf + index, delta,
        //        sk0 + index, sk1 + index, sk2 + index, sk3 + index, sk4 + index,
        //	  delta );

        gmplog(d_a + i, interval, d_b + i, interval, prec_words, d, 1,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index, sk4 + index,
               delta);
    }
}


__global__
void gmparray_exp_dshared_kernel(const double *d_a, double *d_b, const int interval,
                                 const int numElement, const int prec_words,
                                 double *sk0, double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    extern __shared__ double d[];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words, d + threadIdx.x, blockDim.x,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index, delta);
    }
}

__global__
void gmparray_exp_dsk0shared_kernel(const double *d_a, double *d_b, const int interval,
                                    const int numElement, const int prec_words,
                                    double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    extern __shared__ double data[];
    double *d = data;
    double *sk0 = data + (prec_words + 9) * blockDim.x;

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d + threadIdx.x, blockDim.x,
               sk0 + threadIdx.x, blockDim.x,
               sk1 + index, delta,
               sk2 + index, delta,
               sk3 + index, delta);
    }
}


void gmparray_exp_alldevice(gmp_array *d_a, gmp_array *d_b,
                            unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;

    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;

    printf("all intermediate results are stored in the device memory.\n");
    startTimer(timer);
    if (interval > 1) {
        gmparray_exp_alldevice_kernel << < numBlock, numThread >> >
                                                     (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_buf, d_sk0, d_sk1, d_sk2, d_sk3);
        //cutilCheckMsg("gmparray_exp_alldevice_kernel");
        cutilSafeCall(cudaThreadSynchronize());
        endTimer(timer, "gmparray_exp_alldevice_kernel");
    } else {
        gmparray_exp_alldevice_seq_kernel << < numBlock, numThread >> >
                                                         (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_buf, d_sk0, d_sk1, d_sk2, d_sk3);
        //cutilCheckMsg("gmparray_exp_alldevice_seq_kernel");
        cutilSafeCall(cudaThreadSynchronize());
        endTimer(timer, "gmparray_exp_alldevice_seq_kernel");
    }

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_buf);
}


void gmparray_log_alldevice(gmp_array *d_a, gmp_array *d_b,
                            unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;

    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_sk4 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk4, sizeof(double) * (prec_words + 6) * numBlock * numThread);

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;

    printf("all intermediate results are stored in the device memory.\n");
    startTimer((timer));
    if (interval > 1) {
        gmparray_log_alldevice_kernel << < numBlock, numThread >> >
                                                     (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_buf, d_sk0, d_sk1, d_sk2, d_sk3, d_sk4);
        //cutilCheckMsg("gmparray_log_alldevice_kernel");
        cutilSafeCall(cudaThreadSynchronize());
        endTimer((timer), "gmparray_log_alldevice_kernel");
    } else {
        printf("!!!ERROR: interval cannot be 1!\n");
        exit(EXIT_SUCCESS);
    }

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_sk4);
    GPUFREE(d_buf);
}

void gmparray_log(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread) {
    gmparray_log_alldevice(d_a, d_b, numBlock, numThread);
}


__global__
void gmparray_sqrt_dshared_kernel(const double *d_a, double *d_b, const int interval,
                                  const int numElement, const int prec_words,
                                  double *d_sk0, double *d_sk1) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    extern __shared__ double d_buf[];

    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d_buf + threadIdx.x, blockDim.x,
                d_sk0 + index, delta,
                d_sk1 + index, delta);
    }
}

__global__
void gmparray_sqrt_dsk0shared_kernel(const double *d_a, double *d_b, const int interval,
                                     const int numElement, const int prec_words,
                                     double *d_sk1) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    extern __shared__ double data[];
    double *d_buf = data;
    double *d_sk0 = data + blockDim.x * (prec_words + 9);

    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d_buf + threadIdx.x, blockDim.x,
                d_sk0 + threadIdx.x, blockDim.x,
                d_sk1 + index, delta);
    }
}


void gmparray_sqrt_shared(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = numElement;
    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;
    unsigned int sharedMemSize = 0;

    printf("d_buf is stored in the shared memory.\n");
    sharedMemSize = sizeof(double) * (prec_words + 9) * numThread;
    if (sharedMemSize > MAX_SHARED_MEM_SIZE) {
        printf("!!!ERROR for the shared memory size, numThread = %d, sharedMemSize = %d\n",
               numThread, sharedMemSize);
        return;
    }
    startTimer((timer));
    gmparray_sqrt_dshared_kernel << < numBlock, numThread, sharedMemSize >> >
                                                           (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_sk0, d_sk1);
    //cutilCheckMsg("gmparray_sqrt_dshared_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_dshared_kernel");


    printf("d_buf, d_sk0 are stored in the shared memory.\n");
    sharedMemSize = sizeof(double) * (prec_words + 9 + prec_words + 7) * numThread;
    if (sharedMemSize > MAX_SHARED_MEM_SIZE) {
        printf("!!!ERROR for the shared memory size, numThread = %d, sharedMemSize = %d\n",
               numThread, sharedMemSize);
        return;
    }
    startTimer((timer));
    gmparray_sqrt_dsk0shared_kernel << < numBlock, numThread, sharedMemSize >> >
                                                              (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_sk1);
    //cutilCheckMsg("gmparray_sqrt_dsk0shared_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_dsk0shared_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_buf);
}


void gmparray_exp_shared(gmp_array *d_a, gmp_array *d_b,
                         unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;
    unsigned int sharedMemSize = 0;

    printf("d_buf is stored in the shared memory.\n");
    sharedMemSize = sizeof(double) * (prec_words + 9) * numThread;
    if (sharedMemSize > MAX_SHARED_MEM_SIZE) {
        printf("!!!ERROR for the shared memory size, numThread = %d, sharedMemSize = %d\n",
               numThread, sharedMemSize);
        return;
    }
    startTimer((timer));
    gmparray_exp_dshared_kernel << < numBlock, numThread, sharedMemSize >> >
                                                          (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk0, d_sk1, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp_dshared_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_dshared_kernel");


    printf("d_buf and d_sk0 are stored in the shared memory.\n");
    sharedMemSize = sizeof(double) * (prec_words + 9 + prec_words + 6) * numThread;
    if (sharedMemSize > MAX_SHARED_MEM_SIZE) {
        printf("!!!ERROR for the shared memory size, numThread = %d, sharedMemSize = %d\n",
               numThread, sharedMemSize);
        return;
    }
    startTimer((timer));
    gmparray_exp_dsk0shared_kernel << < numBlock, numThread, sharedMemSize >> >
                                                             (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk1, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp_dsk0shared_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_dsk0shared_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_buf);
}


__global__
void gmparray_exp2_kernel0(const double *d_a, double *d_b, const int interval,
                           const int numElement, const int prec_words,
                           double *d_buf, double *sk0, double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;

    double *d = d_buf + index;
    const int d_interval = delta;

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words, d, d_interval,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index, delta);
    }
}


__global__
void gmparray_exp2_kernel1(const double *d_a, double *d_b, const int interval,
                           const int numElement, const int prec_words,
                           double *sk0, double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words, d, 1,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index, delta);
    }
}

__global__
void gmparray_exp2_kernel2(const double *d_a, double *d_b, const int interval,
                           const int numElement, const int prec_words,
                           double *sk1, double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    double sk0[EXP_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               sk0, 1,
               sk1 + index, delta,
               sk2 + index, delta,
               sk3 + index, delta);
    }
}

__global__
void gmparray_exp2_kernel3(const double *d_a, double *d_b, const int interval,
                           const int numElement, const int prec_words,
                           double *sk2, double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    double sk0[EXP_TMP_SIZE];
    double sk1[EXP_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               sk0, 1,
               sk1, 1,
               sk2 + index, delta,
               sk3 + index, delta);
    }
}

__global__
void gmparray_exp2_kernel4(const double *d_a, double *d_b, const int interval,
                           const int numElement, const int prec_words,
                           double *sk3) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    double sk0[EXP_TMP_SIZE];
    double sk1[EXP_TMP_SIZE];
    double sk2[EXP_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               sk0, 1,
               sk1, 1,
               sk2, 1,
               sk3 + index, delta);
    }
}


__global__ void
gmparray_exp2_kernel5(const double *d_a, double *d_b, const int interval,
                      const int numElement, const int prec_words) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    //extern __shared__ double d[];
    double sk0[EXP_TMP_SIZE];
    double sk1[EXP_TMP_SIZE];
    double sk2[EXP_TMP_SIZE];
    double sk3[EXP_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               sk0, 1,
               sk1, 1,
               sk2, 1,
               sk3, 1);
    }
}


__global__ void
gmparray_exp2_shared_kernel(const double *d_a, double *d_b, const int interval,
                            const int numElement, const int prec_words) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    extern __shared__ double d[];
    double sk0[EXP_TMP_SIZE];
    double sk1[EXP_TMP_SIZE];
    double sk2[EXP_TMP_SIZE];
    double sk3[EXP_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpexp(d_a + i, interval, d_b + i, interval, prec_words,
               d + threadIdx.x, blockDim.x,
               sk0, 1,
               sk1, 1,
               sk2, 1,
               sk3, 1);
    }
}


__global__ void
gmparray_sqrt_alldevice_kernel(const double *d_a, double *d_b, const int interval,
                               const int numElement, const int prec_words,
                               double *d_buf, double *d_sk0, double *d_sk1) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;


    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d_buf + index, delta,
                d_sk0 + index, delta,
                d_sk1 + index, delta);
    }
}

__global__ void
gmparray_sqrt_dregister_kernel(const double *d_a, double *d_b, const int interval,
                               const int numElement, const int prec_words,
                               double *d_sk0, double *d_sk1) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d, 1,
                d_sk0 + index, delta,
                d_sk1 + index, delta);
    }
}


__global__ void
gmparray_sqrt_dsk0register_kernel(const double *d_a, double *d_b, const int interval,
                                  const int numElement, const int prec_words,
                                  double *d_sk1) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    double sk0[SQRT_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d, 1,
                sk0, 1,
                d_sk1 + index, delta);
    }
}


__global__ void
gmparray_sqrt_dsk0sk1register_kernel(const double *d_a, double *d_b, const int interval,
                                     const int numElement, const int prec_words) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE];
    double sk0[SQRT_TMP_SIZE];
    double sk1[SQRT_TMP_SIZE];

    for (int i = index; i < numElement; i += delta) {
        gmpsqrt(d_a + i, interval, d_b + i, interval, prec_words,
                d, 1,
                sk0, 1,
                sk1, 1);
    }
}


void gmparray_sqrt_alldevice(gmp_array *d_a, gmp_array *d_b,
                             unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = numElement;
    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    StopWatchInterface *timer;

    printf("*** all are stored in the global memory ***\n");
    startTimer((timer));
    gmparray_sqrt_alldevice_kernel << < numBlock, numThread >> >
                                                  (d_a->d_mpr, d_b->d_mpr, interval,
                                                          numElement, prec_words,
                                                          d_buf, d_sk0, d_sk1);
    //cutilCheckMsg("gmparray_sqrt_alldevice_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_alldevice_kernel");

    GPUFREE(d_buf);
    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
}

void gmparray_sqrt_registers(gmp_array *d_a, gmp_array *d_b,
                             unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = numElement;
    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 7) * numBlock * numThread);
    StopWatchInterface *timer;

    printf("*** all are stored in the global memory ***\n");
    startTimer((timer));
    gmparray_sqrt_alldevice_kernel << < numBlock, numThread >> >
                                                  (d_a->d_mpr, d_b->d_mpr, interval,
                                                          numElement, prec_words,
                                                          d_buf, d_sk0, d_sk1);
    //cutilCheckMsg("gmparray_sqrt_alldevice_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_alldevice_kernel");

    printf("*** d_buf is stored using registers ***\n");
    startTimer((timer));
    gmparray_sqrt_dregister_kernel << < numBlock, numThread >> >
                                                  (d_a->d_mpr, d_b->d_mpr, interval,
                                                          numElement, prec_words,
                                                          d_sk0, d_sk1);
    //cutilCheckMsg("gmparray_sqrt_dregister_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_dregister_kernel");

    printf("*** d_buf, sk0 are stored using registers *** \n");
    startTimer((timer));
    gmparray_sqrt_dsk0register_kernel << < numBlock, numThread >> >
                                                     (d_a->d_mpr, d_b->d_mpr, interval,
                                                             numElement, prec_words,
                                                             d_sk1);
    //cutilCheckMsg("gmparray_sqrt_dsk0register_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_dsk0register_kernel");


    printf("*** d_buf, sk0, sk1 are stored using registuers ***\n");
    startTimer((timer));
    gmparray_sqrt_dsk0sk1register_kernel << < numBlock, numThread >> >
                                                        (d_a->d_mpr, d_b->d_mpr, interval,
                                                                numElement, prec_words);
    //cutilCheckMsg("gmparray_sqrt_dsk0sk1register_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sqrt_dsk0sk1register_kernel");


    GPUFREE(d_buf);
    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
}

void gmparray_exp_registers(gmp_array *d_a, gmp_array *d_b,
                            unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    StopWatchInterface *timer;

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);

    printf("none is stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel0 << < numBlock, numThread >> >
                                         (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_buf, d_sk0, d_sk1, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");


    printf("d_buf is stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel1 << < numBlock, numThread >> >
                                         (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk0, d_sk1, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");

    printf("d_buf, d_sk0 is stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel2 << < numBlock, numThread >> >
                                         (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk1, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");

    printf("d_buf, d_sk0, d_sk1 is stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel3 << < numBlock, numThread >> >
                                         (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk2, d_sk3);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");


    printf("d_buf, d_sk0, d_sk1, d_sk2 is stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel4 << < numBlock, numThread >> >
                                         (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words, d_sk3);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");

    printf("d_buf, d_sk0, d_sk1, d_sk2, d_sk3 are stored in the registers.\n");
    startTimer((timer));
    gmparray_exp2_kernel5 << < numBlock, numThread >> > (d_a->d_mpr, d_b->d_mpr, numElement, numElement, prec_words);
    //cutilCheckMsg("gmparray_exp2_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_exp_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_buf);
}


__global__
void gmparray_sin_alldevice_kernel(const double *d_a, double *d_b, const int interval,
                                   const int numElement, const int prec_words,
                                   double *d_buf, double *sk0, double *sk1, double *sk2, double *sk3,
                                   double *sk4, double *sk5, double *sk6,
                                   const double *d_sin_table, const double *d_cos_table) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE + 1];

    for (int i = index; i < numElement; i += delta) {

        gmpsin(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               d_sin_table, d_cos_table,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index,
               sk4 + index, sk5 + index, sk6 + index, delta);

    }
}


__global__
void gmparray_cos_alldevice_kernel(const double *d_a, double *d_b, const int interval,
                                   const int numElement, const int prec_words,
                                   double *d_buf, double *sk0, double *sk1, double *sk2, double *sk3,
                                   double *sk4, double *sk5, double *sk6,
                                   const double *d_sin_table, const double *d_cos_table) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE + 1];

    for (int i = index; i < numElement; i += delta) {

        gmpcos(d_a + i, interval, d_b + i, interval, prec_words,
               d, 1,
               d_sin_table, d_cos_table,
               sk0 + index, sk1 + index, sk2 + index, sk3 + index,
               sk4 + index, sk5 + index, sk6 + index, delta);

    }
}


__global__
void gmparray_cssn_alldevice_kernel(const double *d_a, double *d_x, double *d_y,
                                    const int interval, const int numElement, const int prec_words,
                                    double *d_buf, double *sk0, double *sk1, double *sk2, double *sk3,
                                    double *sk4, double *sk5, double *sk6,
                                    const double *d_sin_table, const double *d_cos_table) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int delta = blockDim.x * gridDim.x;
    double d[MAX_D_SIZE + 1];

    for (int i = index; i < numElement; i += delta) {

        gmpcssn(d_a + i, interval,
                d_x + i, interval,
                d_y + i, interval,
                prec_words,
                d, 1,
                d_sin_table, d_cos_table,
                sk0 + index, sk1 + index, sk2 + index,
                sk3 + index, sk4 + index, sk5 + index,
                sk6 + index, delta);
    }
}


void gmparray_tan_alldevice(gmp_array *d_a, gmp_array *d_b, unsigned int numBlock, unsigned int numThread) {

    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;

    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_sk4 = NULL;
    double *d_sk5 = NULL;
    double *d_sk6 = NULL;
    double *d_buf = NULL;
    double *d_x = NULL;
    double *d_y = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk4, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk5, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk6, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_x, sizeof(double) * (prec_words + 5) * numElement);
    GPUMALLOC((void **) &d_y, sizeof(double) * (prec_words + 5) * numElement);
    const unsigned int n_words = prec_words + 5;
    gmp_init_kernel << < 256, 256 >> > (d_x, numElement, n_words);
    //cutilCheckMsg("gmp_init_kernel_1");
    gmp_init_kernel << < 256, 256 >> > (d_y, numElement, n_words);
    //cutilCheckMsg("gmp_init_kernel_2");

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    printf("numElement = %d, interval = %d\n", numElement, interval);
    StopWatchInterface *timer;

    printf("all intermediate results are stored in the device memory.\n");
    printf("Additional global memory allocated: %.3f MB\n",
           sizeof(double) * (numBlock * numThread * (prec_words + 6) * 8 + numElement * (prec_words + 5) * 2) / 1024.0 /
           1024.0);
    startTimer((timer));
    gmparray_cssn_alldevice_kernel << < numBlock, numThread >> >
                                                  (d_a->d_mpr, d_x, d_y, interval, numElement, prec_words, d_buf,
                                                          d_sk0, d_sk1, d_sk2, d_sk3, d_sk4, d_sk5, d_sk6,
                                                          d_sin_table, d_cos_table);
    //cutilCheckMsg("gmparray_cssn_alldevice_kernel");
    gmparray_div_kernel << < numBlock, numThread >> > (d_y, d_x, d_b->d_mpr, numElement, numElement, prec_words);
    //cutilCheckMsg("gmparray_div_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_tan_alldevice_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_sk4);
    GPUFREE(d_sk5);
    GPUFREE(d_sk6);
    GPUFREE(d_buf);
    GPUFREE(d_x);
    GPUFREE(d_y);
}


void gmparray_sin_alldevice(gmp_array *d_a, gmp_array *d_b,
                            unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;

    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_sk4 = NULL;
    double *d_sk5 = NULL;
    double *d_sk6 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk4, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk5, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk6, sizeof(double) * (prec_words + 6) * numBlock * numThread);

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;

    printf("all intermediate results are stored in the device memory.\n");
    startTimer((timer));
    gmparray_sin_alldevice_kernel << < numBlock, numThread >> >
                                                 (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_buf,
                                                         d_sk0, d_sk1, d_sk2, d_sk3, d_sk4, d_sk5, d_sk6,
                                                         d_sin_table, d_cos_table);
    //cutilCheckMsg("gmparray_sin_alldevice_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_sin_alldevice_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_sk4);
    GPUFREE(d_sk5);
    GPUFREE(d_sk6);
    GPUFREE(d_buf);
}


void gmparray_cos_alldevice(gmp_array *d_a, gmp_array *d_b,
                            unsigned int numBlock, unsigned int numThread) {
    const unsigned int numElement = d_a->numElement;
    const unsigned int interval = d_a->interval;

    double *d_sk0 = NULL;
    double *d_sk1 = NULL;
    double *d_sk2 = NULL;
    double *d_sk3 = NULL;
    double *d_sk4 = NULL;
    double *d_sk5 = NULL;
    double *d_sk6 = NULL;
    double *d_buf = NULL;
    GPUMALLOC((void **) &d_buf, sizeof(double) * (prec_words + 9) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk0, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk1, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk2, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk3, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk4, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk5, sizeof(double) * (prec_words + 6) * numBlock * numThread);
    GPUMALLOC((void **) &d_sk6, sizeof(double) * (prec_words + 6) * numBlock * numThread);

    printf("numBlock = %d, numThread = %d\n", numBlock, numThread);
    StopWatchInterface *timer;

    printf("all intermediate results are stored in the device memory.\n");
    startTimer((timer));
    gmparray_cos_alldevice_kernel << < numBlock, numThread >> >
                                                 (d_a->d_mpr, d_b->d_mpr, interval, numElement, prec_words, d_buf,
                                                         d_sk0, d_sk1, d_sk2, d_sk3, d_sk4, d_sk5, d_sk6,
                                                         d_sin_table, d_cos_table);
    //cutilCheckMsg("gmparray_cos_alldevice_kernel");
    cutilSafeCall(cudaThreadSynchronize());
    endTimer((timer), "gmparray_cos_alldevice_kernel");

    GPUFREE(d_sk0);
    GPUFREE(d_sk1);
    GPUFREE(d_sk2);
    GPUFREE(d_sk3);
    GPUFREE(d_sk4);
    GPUFREE(d_sk5);
    GPUFREE(d_sk6);
    GPUFREE(d_buf);
}

*/
#endif /* __GMP_ARRAY_FUNC_CU__ */


