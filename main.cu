// my old CRT.py code
// https://gist.github.com/bivoje/ad7d9c57c3ecb4673bf6dea6bbafe0b3 

#include <assert.h>
#include <stdio.h>
#include <gmp.h>

//#include "primes.list"

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

int _host_collatz_delay(mpz_t n, unsigned limit, int log=0) {
    int stops = 0;
    while(limit == 0 || stops < limit && mpz_cmp_ui(n, 1) != 0) {
        if(mpz_even_p(n)) {
            if(log==1) printf("0 ");
            mpz_divexact_ui(n, n, 2);
        } else {
            if(log==1) printf("1 ");
            mpz_mul_ui(n, n, 3);
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
    for(int i=0; i<repeat; i++) {
        _host_collatz_delay(n, 1, log);
        gmp_printf("%d C| -> %Zd\n", i, n);
    }
}

void test_crt_conv() {
    const unsigned K = 6;
    unsigned _ms[K] = {10007,10009,10037,10039,10061,10067};
    unsigned _ns[K] = {0,};
    unsigned _n = 987654321;

    mpz_t n;
    mpz_set_ui(n, _n);

    host_int2crt(_ns, K, _ms, n);
    gmp_printf("ns = "); for(int k=0; k<K; k++) gmp_printf("%d, ", _ns[k]); gmp_printf("\n");
    host_crt2int(n, K, _ms, _ns);
    gmp_printf("n = %Zd\n", n);

    mpz_clear(n);
}

__device__ bool __all_block_sync(bool v) {
    __shared__ int vv[32]; // = {1,};

    v = __all_sync(0xFFFFFFFF, v);

    if(threadIdx.x % 32 == 0)
        vv[threadIdx.x / 32] = v;

    __syncthreads();

    if(threadIdx.x < 32) {
        v = __all_sync(0xFFFFFFFF, v);
        vv[threadIdx.x] = v;
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
            return ((t1%m)+m) % m; // FIXME further limits the size of m
        }

        int s2 = s0 - s1 * q1;
        int t2 = t0 - t1 * q1;

        r0 = r1; r1 = r2; s0 = s1; s1 = s2; t0 = t1; t1 = t2;
    }
}

// TODO merge with __xor_block_sync
// FIXME assumes blockDim being multiple of 2
__device__ double __sum_block_sync(double v) {
    __shared__ double vv[1024]; // = {0,};
    vv[threadIdx.x] = v;

    // TODO unroll last warp
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            double rank = vv[threadIdx.x] + vv[threadIdx.x + stride];
            //double tfrac, tint;
            //tfrac = modf(rank, &tint);
            //rank = (((int) tint) % 2) + tfrac;
            vv[threadIdx.x] = rank;
        }
        __syncthreads();
    }

    return vv[0];
}

// TODO merge with __sum_block_sync
// FIXME assumes blockDim being multiple of 2
__device__ bool __xor_block_sync(bool v) {
    __shared__ bool vv[1024]; // = {0,};
    vv[threadIdx.x] = v;

    // TODO unroll last warp
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            vv[threadIdx.x] ^= vv[threadIdx.x + stride];
        }
        __syncthreads();
    }

    return vv[0];
}

__global__ void collatz_delay(int *ret, unsigned K, unsigned *ms, unsigned *ns, unsigned limit, int log=0) {
    const unsigned k = threadIdx.x;
    const unsigned m = ms[k];
    unsigned _M_; {
        unsigned long M = 1;
        for(int i=0; i<K; i++) if(k != i) { M *= ms[i]; M %= m; }
        _M_ = mod_inv(m, M);
    }
    const unsigned M_ = _M_; 

    unsigned n = ns[k];
    int stops = 0;

    while(limit == 0 || stops < limit) {

        bool is_one = __all_block_sync(n == 1);
        bool is_even; {
            double rank = ((double) n) * (((double) M_) / m);
            double tint, tfrac;
            tfrac = modf(rank, &tint);
            rank = (((int) tint) % 2) + tfrac;
            double rank_total = __sum_block_sync(rank) + 1e-3;
            bool rank_parity = 1 & (int) floor(rank_total);
            if(log==2 && threadIdx.x==0) printf("rank=%lf, rank_total=%lf, rank_parity=%d\n", rank, rank_total, rank_parity);

            bool term_parity = (((unsigned long) n) * M_) % 2;
            bool terms_parity = __xor_block_sync(term_parity);
            if(log==2 && threadIdx.x==0) printf("term=%d, terms=%d\n", term_parity, terms_parity);

            bool parity = rank_parity ^ terms_parity;
            is_even = !parity;
            if(log==2 && threadIdx.x==0) printf("parity=%d\n", parity);
        }

        if (is_one) break;

        if(is_even) {
            if(log==1) if(threadIdx.x==0) printf("0 ");
            n = div2(m, n); n %= m;
        } else {
            if(log==1) if(threadIdx.x==0) printf("1 ");
            // FIXME assume m < UINTMAX/3
            n *= 3; n += 1; n %= m;
        }
        stops += 1;
    }

    ns[k] = n;
    if(log==1) if(threadIdx.x==0) printf("\n");

    if(k == 0)
        if(limit == 0 || stops < limit) 
            *ret = stops;
        else
            *ret = -1;
}

struct collatz_delay_t {
    unsigned K;
    unsigned *host_ns;
    unsigned *device_ms, *device_ns;
    cudaEvent_t start, stop;
};

void device_collatz_init(struct collatz_delay_t *A, unsigned K, unsigned *host_ms, mpz_t n) {
    A->K = K;

    A->host_ns = (unsigned*) malloc(K * sizeof(unsigned));
    host_int2crt(A->host_ns, K, host_ms, n);

    cudaMalloc((void **)&A->device_ms, K * sizeof(unsigned));
    cudaMalloc((void **)&A->device_ns, K * sizeof(unsigned));
    cudaMemcpy(A->device_ms, host_ms, K * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(A->device_ns, A->host_ns, K * sizeof(unsigned), cudaMemcpyHostToDevice);

    cudaEventCreate(&A->start);
    cudaEventCreate(&A->stop);
}

void device_collatz_free(struct collatz_delay_t *A) {
    cudaFree(A->device_ns);
    cudaFree(A->device_ms);

    free(A->host_ns);

    cudaEventDestroy(A->start);
    cudaEventDestroy(A->stop);
}

int device_collatz_kernel(struct collatz_delay_t *A, unsigned limit, float *time, int log=0) {
    int blockDim = A->K; // 1024;
    int gridDim = 1; //saturating_div(K, blockDim);

    int host_delay, *device_delay;
    cudaMalloc((void **)&device_delay, sizeof(int));

    cudaEventRecord(A->start, 0);
    collatz_delay<<<gridDim,blockDim>>>(device_delay, A->K, A->device_ms, A->device_ns, limit, log);
    cudaEventRecord(A->stop, 0);

    cudaMemcpy(&host_delay, device_delay, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(A->host_ns, A->device_ns, A->K * sizeof(unsigned), cudaMemcpyDeviceToHost);

    cudaFree(device_delay);

    float ms;
    cudaEventElapsedTime(&ms, A->start, A->stop);
    if(time != NULL) *time += ms;

    return host_delay;
}

int device_collatz_delay(unsigned K, unsigned *host_ms, mpz_t n, unsigned limit, float *time, int log=0) {
    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, n);
    int delay = device_collatz_kernel(&A, limit, time, log);
    device_collatz_free(&A);

    return delay;
}

void device_collatz_steps(unsigned K, unsigned *host_ms, mpz_t n, int repeat, int log=0) {

    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, n);
    for(int i=0; i<repeat; i++) {
        device_collatz_kernel(&A, 1, NULL, log);
        mpz_t t; mpz_init(t);
        host_crt2int(t, K, host_ms, A.host_ns);
        gmp_printf("%d G| -> %Zd\n", i, t);
    }
    device_collatz_free(&A);
}

void test_compare_str(unsigned K, unsigned *host_ms, mpz_t n, int sel, unsigned limit, int log) {
    int delay_host   = 0;
    float time_spent_cpu = 0;
    if(sel & 1) delay_host = host_collatz_delay(n, limit, &time_spent_cpu, log);
    gmp_printf("cpu time: \t%f ms\t\n", time_spent_cpu);

    int delay_device = 0;
    float time_spent_gpu = 0;
    if(sel & 2) delay_device = device_collatz_delay(K, host_ms, n, limit, &time_spent_gpu, log);
    gmp_printf("gpu time: \t%f ms\n", time_spent_gpu);

    //if(delay_host==delay_device)
    gmp_printf(": %3d %c %3d\n", delay_host, (delay_host==delay_device)?' ':'|', delay_device);
}

unsigned host_ms8[] = {10007,3,5,7,11,13,17,19}; 

unsigned host_ms64[] = {100003,100019,100043,100049,100057,100069,100103,100109,100129,100151,100153,100169,100183,100189,100193,100207,100213,100237,100267,100271,100279,100291,100297,100313,100333,100343,100357,100361,100363,100379,100391,100393,100403,100411,100417,100447,100459,100469,100483,100493,100501,100511,100517,100519,100523,100537,100547,100549,100559,100591,100609,100613,100621,100649,100669,100673,100693,100699,100703,100733,100741,100747,100769,100787,}; 

unsigned host_ms1024[] = {
    #include "primes.list"
};

#include <math.h>

int main(int argc, char **argv) {
    char *num = "10";
    if(argc > 1) {
        num = argv[1];
    }

    int size_level = 3;
    if(argc > 2) {
        size_level = atoi(argv[2]);
        assert(size_level > 0 && "size_level needs to be positive");
    }

    int sel = 3;
    if(argc > 3) {
        sel = argv[3][0] & 3;
        assert(sel);
    }

    unsigned limit = 2147483647u * 2 + 1;
    if(argc > 4) {
        limit = atoi(argv[4]);
    }

    unsigned *host_ms;
    if(size_level <= 3) {
        host_ms = host_ms8;
    } else if(size_level <= 6) {
        host_ms = host_ms64;
    } else if(size_level <= 10) {
        host_ms = host_ms1024;
    } else {
        assert(false && "size_level too big");
    }
    unsigned K = 1 << size_level;

    double maxlg = 0;
    for(int k=0; k<K; k++) maxlg += log10(host_ms[k]);
    double integer, fractional;
    fractional = modf(maxlg, &integer);
    int max_exponent = (int) integer;
    double max_mantissa = pow(10, fractional);

    int num_exponent = strlen(num) - 1;
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
    gmp_printf("size_level: %d, sel: %c%c, limit:%u\n", size_level, sel&1?'c':'\0', sel&2?'g':'\0', limit);

    mpz_t n; mpz_init(n); mpz_set_str(n, num, 10);
    test_compare_str(K, host_ms, n, sel, limit, 2);

    //host_collatz_steps(n, 10);
    //device_collatz_steps(K, host_ms, n, 10);

    return 0;
}

int _main(int argc, char **argv) {
    //for(int i=1000000; i>0; i--) test_compare(i, false);

    //test_compare_ui(987654321, true);

    //int _n = 1438;
    //mpz_t n; mpz_init(n); mpz_set_ui(n, _n);
    //collatz_step_device(K, host_ms, n, 1);
    //gmp_printf("%d -> %Zd\n", _n, n);

    //#define L 2000
    //char str[L]; for(int i=0; i<L; i++) str[i] = '9'; str[L-1] = 0;
    //mpz_t n; mpz_init(n); mpz_set_str(n, str, 10);
    //while(1) {
    //    mpz_sub_ui(n, n, 1);
    //    test_compare_str(n, false);
    //}

    //char *str = "9999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999995";

    char *str =
#include "num"
        ;
    mpz_t n; mpz_init(n); mpz_set_str(n, str, 10);
    test_compare_str(1024, host_ms1024, n, 3, 100000, false);

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
