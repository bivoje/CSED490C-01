// my old CRT.py code
// https://gist.github.com/bivoje/ad7d9c57c3ecb4673bf6dea6bbafe0b3 

#include <assert.h>
#include <stdio.h>
#include <gmp.h>

// =====================================
// host CRT

void host_ui2crt(unsigned *ns, unsigned K, unsigned *ms, unsigned n) {
    for(int k=0; k<K; k++) ns[k] = n % ms[k];
}

void host_int2crt(unsigned *_ns, unsigned K, unsigned *_ms, mpz_t n) {
    mpz_t t;
    mpz_init(t);

    for(int k=0; k<K; k++) {
        mpz_mod_ui(t, n, _ms[k]);
        _ns[k] = mpz_get_ui(t);
    }
}

void host_crt2int(mpz_t n, unsigned K, unsigned *_ms, unsigned *_ns) {
    mpz_set_ui(n, 0);

    mpz_t _M;
    mpz_init(_M);
    mpz_set_ui(_M, 1);
    for(int k=0; k<K; k++) mpz_mul_ui(_M, _M, _ms[k]);
    //gmp_printf("_M = %Zd\n", _M);

    mpz_t m, M, term;
    mpz_inits(m, M, term, NULL);
    for(int k=0; k<K; k++) {
        mpz_set_ui(m, _ms[k]);
        mpz_divexact(M, _M, m);
        mpz_mul_ui(term, M, _ns[k]);
        int ok = mpz_invert(M, M, m);
        assert(ok);
        mpz_mul(term, term, M);
        mpz_add(n, n, term);
        mpz_mod(n, n, _M);
    }

    mpz_clears(term, M, _M, NULL);
}


// =====================================
// host collatz

int _host_collatz_delay(mpz_t n, unsigned limit, int log=0) {
    int stops = 0;
    while(limit == 0 || stops < limit && mpz_cmp_ui(n, 1) != 0) {
        if(mpz_even_p(n)) {
            if(log==1) printf("0 ");
            mpz_divexact_ui(n, n, 2);
        } else {
            if(log==1) printf("1 ");
            mpz_mul_ui(n, n, 3);
            //mpz_mul(n, n, n);
            mpz_add_ui(n, n, 1);
        }
        stops += 1;
    }
    if(log==1) printf("\n");

    if(mpz_cmp_ui(n, 1) == 0)
        return stops;
    else
        return -1;
}

#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>

unsigned long long current_time_micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((unsigned long long)tv.tv_sec * 1000000ULL) + tv.tv_usec;
}

int host_collatz_delay(mpz_t _n, unsigned limit, float *time, int log=0) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);

    unsigned long long begin = current_time_micros();
    int delay = _host_collatz_delay(n, limit, log);
    unsigned long long end = current_time_micros();
    unsigned long long time_spent = end - begin;

    if(time != NULL) *time += time_spent / 1000.0;

    return delay;
}

void host_collatz_steps(mpz_t _n, int repeat, int log=0) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);
    gmp_printf("%d C| -> %Zd\n", 0, n);
    for(int i=1; i<=repeat; i++) {
        _host_collatz_delay(n, 1, log);
        gmp_printf("%d C| -> %Zd\n", i, n);
    }
}

size_t host_collatz_maxbits(mpz_t _n, unsigned limit, int base, int log=0) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);
    size_t maxbits = mpz_sizeinbase(n, base);

    int stops = 0;
    while(limit == 0 || stops < limit && mpz_cmp_ui(n, 1) != 0) {
        if(mpz_even_p(n)) {
            mpz_divexact_ui(n, n, 2);
        } else {
            mpz_mul_ui(n, n, 3);
            mpz_add_ui(n, n, 1);
        }
        size_t bits = mpz_sizeinbase(n, base);
        maxbits = maxbits>bits? maxbits: bits;
    }

    return maxbits;
}


// =====================================
// device collatz

#define CUDACHECK(stmt)                                                  \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("Failed to run stmt %s\n", #stmt);                      \
      printf("Got CUDA error ...  %s\n", cudaGetErrorString(err));   \
    }                                                                     \
  } while (0)


#define saturating_div(A,B) (((A) + (B) - 1) / (B))
#define BLOCK_MULT 2


__device__ bool __all_block_sync(bool v) {
    __shared__ int vv[32];
    if(threadIdx.x < 32) {
        vv[threadIdx.x] = 1;
    }

    v = __all_sync(0xFFFFFFFF, v);

    if(threadIdx.x % 32 == 0)
        vv[threadIdx.x / 32] = v;

    __syncthreads();

    if(threadIdx.x < 32) {
        vv[threadIdx.x] = __all_sync(0xFFFFFFFF, vv[threadIdx.x]);
    }

    __syncthreads();

    return vv[threadIdx.x / 32];
}

__device__ unsigned div2(unsigned m, unsigned n) {
    // assume n < m

    // FIXME (check) this should not diverge the warp...
    // compiler probably would optimize it in 'cmov' style.
    if(n % 2 == 0) {
        return n / 2;
    } else {
        // equal to (m+n)/2 but without overflowing
        return n + (m-n)/2;
    }
}

// FIXME should use int not unsigned
__device__ int mod_inv(int m, int x) {
    int r0 = m, r1 = x, s0 = 1, s1 = 0, t0 = 0, t1 = 1;

    while(1) {
        int q1 = r0 / r1;
        int r2 = r0 - r1 * q1;

        if(r2 == 0) {
            assert(r1 == 1);
            return ((t1%m)+m) % m;
        }

        int s2 = s0 - s1 * q1;
        int t2 = t0 - t1 * q1;

        r0 = r1; r1 = r2; s0 = s1; s1 = s2; t0 = t1; t1 = t2;
    }
}

__device__ double modf_p(double v, bool *p) {
    double tfrac, tint;
    tfrac = modf(v, &tint);
    *p ^= 1 & ((int) tint);
    return tfrac;
}

__global__ void collatz_delay_kernel(int *ret, unsigned K, unsigned *ms, unsigned *ns, unsigned limit, int log=0) {
    unsigned m[BLOCK_MULT];
    unsigned M_[BLOCK_MULT];
    unsigned n[BLOCK_MULT];

    for(int j=0; j<BLOCK_MULT; j++) {
        unsigned k = j * blockDim.x + threadIdx.x;
        m[j] = ms[k];
        m[j] = ms[j * blockDim.x + threadIdx.x];
    }

    for(int j=0; j<BLOCK_MULT; j++) {
        unsigned long M = 1;
        unsigned k = j * blockDim.x + threadIdx.x;
        for(int i=0; i<K; i++) if(k != i) { M *= ms[i]; M %= m[j]; }
        M_[j] = mod_inv(m[j], M);
    }

    for(int j=0; j<BLOCK_MULT; j++) {
        unsigned k = j * blockDim.x + threadIdx.x;
        n[j] = ns[blockIdx.x * K + k];
    }

    int stops = 0;

    __shared__ unsigned rfracs[32];
    __shared__ bool parity[32];
    if (threadIdx.x < 32) {
        rfracs[threadIdx.x] = 0;
        parity[threadIdx.x] = 0;
    }

    while(limit == 0 || stops < limit) {

        bool is_one = true;
        for(int j=0; j<BLOCK_MULT; j++)
            is_one &= n[j] == 1;
        is_one = __all_block_sync(is_one);

        bool is_even; {

            unsigned f = 0;
            bool p = false;

            {
                double rank = ((double) M_[0]) / (double) m[0];
                bool pp = 0;
                rank = modf_p(rank, &pp);
                rank = modf_p(((double) n[0]) * rank, &pp);
                unsigned rank_rfracs = rank * (1U<<31);
                bool term_parity = (1 & n[0]) * (1 & M_[0]);

                f = rank_rfracs;
                p = pp ^ term_parity; 
            }

            for(int j=1; j<BLOCK_MULT; j++) 
            {
                double rank = ((double) M_[j]) / (double) m[j];
                bool pp = 0;
                rank = modf_p(rank, &pp);
                rank = modf_p(((double) n[j]) * rank, &pp);
                unsigned rank_rfracs = rank * (1U<<31);
                bool term_parity = (1 & n[j]) * (1 & M_[j]);

                f += rank_rfracs;
                p ^= pp ^ term_parity;
                p ^= (f >> 31);
                f &= 0x7FFFFFFFU;
            }


            for(unsigned stride = 32/2; stride > 0; stride >>= 1) {
                f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                p ^= (f >> 31);
                f &= 0x7FFFFFFFU;
            }

            if(threadIdx.x % 32 == 0) {
                rfracs[threadIdx.x / 32] = f;
                parity[threadIdx.x / 32] = p;
            }

            __syncthreads();

            if (threadIdx.x < 32) {
                f = rfracs[threadIdx.x];
                p = parity[threadIdx.x];

                for(unsigned stride = 32/2; stride > 1; stride >>= 1) {
                    f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                    p ^= (f >> 31);
                    f &= 0x7FFFFFFFU;
                }

                unsigned stride = 1; {
                    f += __shfl_down_sync(0xFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFF, p, stride, 32);
                    f += 0x00100000U;
                    p ^= (f >> 31);
                }

                if(threadIdx.x == 0) parity[0] = p;
            }

            __syncthreads();

            is_even = !parity[0];
        }

        if (is_one) break;

        if(is_even) {
            if(log==1) if(threadIdx.x==0) printf("0 ");
            for(int j=0; j<BLOCK_MULT; j++) {
                n[j] = div2(m[j], n[j]); n[j] %= m[j];
            }
        } else {
            if(log==1) if(threadIdx.x==0) printf("1 ");
            // FIXME assume m < UINTMAX/3
            for(int j=0; j<BLOCK_MULT; j++) {
                n[j] *= 3; n[j] += 1; n[j] %= m[j];
                //n *= n; n += 1; n %= m;
            }
        }
        stops += 1;
    }

    for(int j=0; j<BLOCK_MULT; j++) {
        unsigned k = j * blockDim.x + threadIdx.x;
        ns[blockIdx.x * K + k] = n[j];
    }

    if(log==1) if(threadIdx.x==0) printf("\n");

    if(threadIdx.x == 0)
        if(limit == 0 || stops < limit) 
            ret[blockIdx.x] = stops;
        else
            ret[blockIdx.x] = -1;
}

struct collatz_delay_t {
    unsigned K, T;
    unsigned *host_ns;
    unsigned *device_ms, *device_ns;
    cudaEvent_t start, stop;
};

void device_collatz_init(struct collatz_delay_t *A, unsigned K, unsigned *host_ms, unsigned T, mpz_t *n) {
    A->K = K;
    A->T = T;

    A->host_ns = (unsigned*) malloc(T * K * sizeof(unsigned));
    for(int t=0; t<T; t++) host_int2crt(&A->host_ns[K*t], K, host_ms, n[t]);

    CUDACHECK( cudaMalloc((void **)&A->device_ms, K * sizeof(unsigned)) );
    CUDACHECK( cudaMalloc((void **)&A->device_ns, T * K * sizeof(unsigned)) );
    CUDACHECK( cudaMemcpy(A->device_ms, host_ms, K * sizeof(unsigned), cudaMemcpyHostToDevice) );
    CUDACHECK( cudaMemcpy(A->device_ns, A->host_ns, T * K * sizeof(unsigned), cudaMemcpyHostToDevice) );

    CUDACHECK( cudaEventCreate(&A->start) );
    CUDACHECK( cudaEventCreate(&A->stop) );
}

void device_collatz_free(struct collatz_delay_t *A) {
    cudaFree(A->device_ns);
    cudaFree(A->device_ms);

    free(A->host_ns);

    cudaEventDestroy(A->start);
    cudaEventDestroy(A->stop);
}

void device_collatz_kernel(int *delay, struct collatz_delay_t *A, unsigned limit, float *time, int log=0) {
    assert(A->K % BLOCK_MULT == 0); // FIXME
    int blockDim = saturating_div(A->K, BLOCK_MULT);
    int gridDim = A->T;

    int *device_delay;
    CUDACHECK( cudaMalloc((void **)&device_delay, A->T * sizeof(int)) );

    CUDACHECK( cudaEventRecord(A->start, 0) );
    if(log) printf("collatz_delay_kernel<<<%d,%d>>>(ret, K=%d, ms, ns, limit=%d);\n", gridDim,blockDim,A->K,limit);
    cudaError_t err = cudaGetLastError();
    printf("CUDAERR: %s\n", cudaGetErrorString(err));
    collatz_delay_kernel<<<gridDim,blockDim>>>(device_delay, A->K, A->device_ms, A->device_ns, limit, log);
    CUDACHECK( cudaEventRecord(A->stop, 0) );

    CUDACHECK( cudaMemcpy(delay, device_delay, A->T * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDACHECK( cudaMemcpy(A->host_ns, A->device_ns, A->K * sizeof(unsigned), cudaMemcpyDeviceToHost) );

    CUDACHECK( cudaFree(device_delay) );

    float ms;
    cudaEventElapsedTime(&ms, A->start, A->stop);
    if(time != NULL) *time += ms;
}

void device_collatz_delay(int *delay, unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, unsigned limit, float *time, int log=0) {
    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, T, n);
    device_collatz_kernel(delay, &A, limit, time, log);
    device_collatz_free(&A);
}

// T = 1 assumed. but for some reason, I have to get 'n' as a pointer
void device_collatz_steps(unsigned K, unsigned *host_ms, mpz_t *n, int repeat, int log=0) {
    gmp_printf("%d G| -> %Zd\n", 0, *n);
    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, 1, n);
    int ret;
    for(int i=1; i<=repeat; i++) {
        device_collatz_kernel(&ret, &A, 1, NULL, log);
        mpz_t x; mpz_init(x);
        host_crt2int(x, K, host_ms, A.host_ns);
        gmp_printf("%d G| -> %Zd\n", i, x);
    }
    device_collatz_free(&A);
}

void test_compare_str(unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, int sel, unsigned limit, int log) {
    int *delay_host = (int*) malloc(T * sizeof(int));
    memset(delay_host, 0, T * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<T; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu, log);
    gmp_printf("cpu time: \t%f ms\t\n", time_spent_cpu);

    int *delay_device = (int*) malloc(T * sizeof(int));
    memset(delay_device, 0, T * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, T, n, limit, &time_spent_gpu, log);
    gmp_printf("gpu time: \t%f ms\n", time_spent_gpu);

    for(int t=0; t<T; t++) // if(delay_host!=delay_device)
        gmp_printf(": %3d %c %3d\n", delay_host[t], (delay_host[t]==delay_device[t])?' ':'|', delay_device[t]);
}

void test_compare_str_(unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, int sel, unsigned limit, int log) {
    int *delay_host = (int*) malloc(T * sizeof(int));
    memset(delay_host, 0, T * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<T; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu, log);

    int *delay_device = (int*) malloc(T * sizeof(int));
    memset(delay_device, 0, T * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, T, n, limit, &time_spent_gpu, log);

    gmp_printf("%d\t%f\t%f\n", T, time_spent_cpu, time_spent_gpu);
}

void test_compare_str__(unsigned K, unsigned *host_ms, unsigned N, mpz_t *n, int sel, unsigned limit, int log) {
    int *delay_host = (int*) malloc(1 * sizeof(int));
    memset(delay_host, 0, 1 * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<1; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu, log);

    int *delay_device = (int*) malloc(1 * sizeof(int));
    memset(delay_device, 0, 1 * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, 1, n, limit, &time_spent_gpu, log);

    gmp_printf("%d\t%d\t%d\t%f\t%f\n", N, delay_host[0], delay_device[0], time_spent_cpu, time_spent_gpu);
}


unsigned host_ms8[] = {10007,3,5,7,11,13,17,19}; 

unsigned host_ms64[] = {100003,100019,100043,100049,100057,100069,100103,100109,100129,100151,100153,100169,100183,100189,100193,100207,100213,100237,100267,100271,100279,100291,100297,100313,100333,100343,100357,100361,100363,100379,100391,100393,100403,100411,100417,100447,100459,100469,100483,100493,100501,100511,100517,100519,100523,100537,100547,100549,100559,100591,100609,100613,100621,100649,100669,100673,100693,100699,100703,100733,100741,100747,100769,100787,}; 

unsigned host_ms10000[] = {
    #include "primes.list"
};

#include <math.h>

int main(int argc, char **argv) {
    char *num = "10";
    if(argc > 1) {
        num = argv[1];
        if(num[0] == '0')
            num = 
                #include "num"
            ;
    }

    int size= 8;
    if(argc > 2) {
        size= atoi(argv[2]);
        assert(size > 0 && "size needs to be positive");
    }

    int sel = 3;
    if(argc > 3) {
        sel = argv[3][0] & 7;
        assert(sel);
    }

    unsigned limit = 2147483647u * 2 + 1;
    if(argc > 4) {
        limit = atoi(argv[4]);
    }

    unsigned *host_ms;
    if(size <= 8) {
        host_ms = host_ms8;
    } else if(size <= 64) {
        host_ms = host_ms64;
    } else if(size <= 10000) {
        host_ms = host_ms10000;
    } else {
        assert(false && "size too big");
    }
    unsigned K = size;

    double maxlg = 0;
    for(int k=0; k<K; k++) maxlg += log10(host_ms[k]);
    double integer, fractional;
    fractional = modf(maxlg, &integer);
    int max_exponent = (int) integer;
    double max_mantissa = pow(10, fractional);

    int num_exponent = strlen(num) - 1;
    assert(num_exponent >= 0);
    double num_mantissa;
    for(int i=0, v=1; i<min(num_exponent, 5); i++, v*=10)
        num_mantissa += (num[i] - '0') / (double) v;

    if(max_exponent < num_exponent || max_exponent == num_exponent && max_mantissa <= num_mantissa) {
        gmp_printf("given number '%.3fe%+d' is too big for max num '%.3fe%+d\n", num_mantissa, num_exponent, max_mantissa, max_exponent);
        return 1;
    }

    if(num_exponent < 40)
        gmp_printf("num: %s (~%.3fe%+d), ", num, max_mantissa, max_exponent);
    else 
        gmp_printf("num: %.3fe%+d (~%.3fe%+d), ", num_mantissa, num_exponent, max_mantissa, max_exponent);
    gmp_printf("size: %d, sel: %c%c, limit:%u\n", size, sel&1?'c':'\0', sel&2?'g':'\0', limit);

    mpz_t n; mpz_init(n); mpz_set_str(n, num, 10);
    if(sel <= 3) 
        test_compare_str(K, host_ms, 1, &n, sel, limit, 1);
    else if(sel == 4) {
        host_collatz_steps(n, 10);
        device_collatz_steps(K, host_ms, &n, 10);
    } else if(sel == 5) {
        int base = 10;
        size_t maxbits = host_collatz_maxbits(n, limit, base);
        printf("maxbits(%d): %lld\n", base, maxbits);
    }

    return 0;
}

int __main(int argc, char **argv) {

    char *num = 
        #include "num"
    ;

    int T = 1 << atoi(argv[1]);

    mpz_t nums[T];
    for(int t=0; t<T; t++) {
        mpz_init(nums[t]);
        mpz_set_str(nums[t], num, 10);
    }

    test_compare_str_(1024, host_ms10000, T, nums, 3, 148624, 0);

    return 0;
}

char num[7000];
int _main(int argc, char **argv) {

    char *_num =
        #include "num"
    ;

    int N = 1 << atoi(argv[1]);
    strncpy(num, _num, N);

    mpz_t nums;
    mpz_init(nums);
    mpz_set_str(nums, num, 10);

    test_compare_str__(1024, host_ms10000, N, &nums, 3, 148624, 0);

    return 0;
}

// TODO baseline? https://github.com/kisupov/grns/blob/master/src/mpint.cuh 
// TODO REF:https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122105X08604/1-s2.0-S0898122105002890/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAEaCXVzLWVhc3QtMSJGMEQCIGo1xFIcj66pOtfY6J%2FYcPJvyOn%2BkUzr20%2FacIZlg4SmAiB9MkoM3AMGjPu8HAzaIimJ20J%2FAip0xTxYnAsd7hejyiqyBQhqEAUaDDA1OTAwMzU0Njg2NSIMQWrqhooi4lWCxZ4TKo8FZRHzLS5gGO4ZpPqix9%2BkifkwMuj%2FK4tZpNBetzdoAzrH0GmHHjz50jhkzhp4WdWiETH%2FLQVn10XIW%2F65r2P5Nc7pyBulLDTjgyZNEkfZRlAYm5JMPBcx%2BLu%2BJ8yIvCgWxTfN65XjvlajCWw8UoeqCSn7wm0rktUYdDaAPNaM4%2Bin0AwTic2Mnm9G%2BRnFarmCcebJKnepFN8D0kywMm62%2BjvCXSR9p0zdrtCrJC2fk3VJUKCJE2ZNvcOKIXD%2FsVYg4h2bcBAx0t3EOwQOiH0ZrMavh2XMyXEH2AfzrK2jKBgiWE3mkBOF7OcgxbT8mN18LywQXf4GKdCbZesQxp3NlUFIQx9pNUU6wfgHaSKSzlq%2BkwMqLGGnA2Q8IoLwMkPc5XjcuBB0aTXOkOkwHLQxE1TWRqaJbLdUuJG6jqLmtMhQJUyKEwS%2FlA%2BRlRrcdNtrPMYsOP0YPLmwLVwNU619XFrjvF%2Ffp5N4iq%2BUb6PHwEKu2IImlWPrQTvIGAlAFRcdnh5SsNo%2B18TA20tSaoIfrun%2F9G%2B2SCcxPrnuxHmdqwnJEg37q2lfQ2MmW4DdowlGCcnVP6oscRfpaj3fU%2BvLozc9xwv6Q1vqPCHaLvNepqiGJLyjFvIqwo7Zsp98mFV2AfMjFSsEaWkpquX3uuAovFvb3B3eS2TurQAyJetQLA5vT7GwdFEv7EjJCWvOGS9wJ6WfHqfF1aRG21WqnLXvm%2BLEZMr5jKMGqTxCeUHhs7RLlbpBdTPkFaSkycmurFNUIFpfvM%2Bxlf8aOM9GceWTupP8YYcqBeXWC6vaiynmw1oGDo2DIeXjzNMMnuiCVgcU2BiRRa9N5kYVwRPD4%2FuUh%2FGqCiZ2AlqKXNv90ukPKDCN3bmrBjqyAbcy6POXsO4CMMY2UmTuMaxRiSCkCfdepdi3Ktu9Gd%2F%2Ff2pQ3TJoe6yN3blB1YMS0ZwO1svNnAgfk8z7mAN7%2BiCMp5aBGBxH4LN8A0anwhLA%2FJcxlLWoMTno9HLiDqA2YaUfjQa4NdE5JL96oDV5MierdaKecKZ0zmXGOW3s7nXEmJvF4Xqlvl8c92IheBpt7Gkt4Ul4Y%2FXgyIG8solI%2F0r64QSdvqyAD1ejXgf1D57jhu8%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20231205T014736Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4AIUGWIT%2F20231205%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d8c84771daf4ce33785a1c6cba18745d84674cfeb3534c584412ab0c389a1c91&hash=bd0fd1bf729c1c1e273ccae0dcf1ca93cb83157d1e40f578516367132744b5fe&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0898122105002890&tid=spdf-0c438e76-d7a8-4f07-a20b-1993fbeba9bc&sid=fea44b6a49440649f418d659643dc372823fgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0d125e5251595b025202&rr=83089f3db8ceedb5&cc=kr 
// TODO REF:https://www.sciencedirect.com/science/article/pii/S0898122105002890?via%3Dihub
// TODO REF: https://link.springer.com/article/10.1007/s00224-021-10035-y#Bib1 
// TODO REF:https://sci-hub.se/https://doi.org/10.1007/s11432-008-0097-y
// TODO REF:https://link.springer.com/chapter/10.1007/978-3-031-34127-4_4#editor-information
/*
https://www.tandfonline.com/doi/full/10.1080/00207160410001708805?scroll=top&needAccess=true
https://ieeexplore.ieee.org/document/1509927
*/
