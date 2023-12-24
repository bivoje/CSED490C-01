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

int _host_collatz_delay(mpz_t n, unsigned limit) {
    int stops = 0;
    while(limit == 0 || stops < limit && mpz_cmp_ui(n, 1) != 0) {
        if(mpz_even_p(n)) {
#if LOG > 0
            printf("0 ");
#endif
            mpz_divexact_ui(n, n, 2);
        } else {
#if LOG > 0
            printf("1 ");
#endif
            mpz_mul_ui(n, n, 3);
            //mpz_mul(n, n, n);
            mpz_add_ui(n, n, 1);
        }
        stops += 1;
    }
#if LOG > 0
    printf("\n");
#endif

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

int host_collatz_delay(mpz_t _n, unsigned limit, float *time) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);

    unsigned long long begin = current_time_micros();
    int delay = _host_collatz_delay(n, limit);
    unsigned long long end = current_time_micros();
    unsigned long long time_spent = end - begin;

    if(time != NULL) *time += time_spent / 1000.0;

    return delay;
}

void host_collatz_steps(mpz_t _n, int repeat) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);
    gmp_printf("%d C| -> %Zd\n", 0, n);
    for(int i=1; i<=repeat; i++) {
        _host_collatz_delay(n, 1);
        gmp_printf("%d C| -> %Zd\n", i, n);
    }
}

size_t host_collatz_maxbits(mpz_t _n, unsigned limit, int base) {
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

#if (OPT < 7) && (BLOCK_MULT != 1)
#error "BLOCK_MULT should be 1 but given " ##BLOCK_MULT
#endif

__device__ bool __all_block_sync(bool v) {
#if OPT < 1 // Original
    __shared__ int ret;
    if(threadIdx.x == 0) ret = 1;
    __syncthreads();
    atomicAnd(&ret, v);
    __syncthreads();
    return ret;

#elif OPT < 2 // Parallel reduction
    __shared__ bool vv[1024];
    vv[threadIdx.x] = v;
    for(unsigned stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) vv[threadIdx.x] &= vv[threadIdx.x+stride];
    }
    return vv[0];

#else // Reduction with warp synchronization
    __shared__ bool vv[32];
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
#endif
}

__device__ bool __xor_block_sync(bool v) {
#if OPT < 1 // Original
    __shared__ int ret;
    if(threadIdx.x == 0) ret = 0;
    __syncthreads();
    atomicXor(&ret, v);
    __syncthreads();
    return ret;

#else // Parallel reduction
    __shared__ bool vv[1024];
    vv[threadIdx.x] = v;
    for(unsigned stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) vv[threadIdx.x] ^= vv[threadIdx.x+stride];
    }
    return vv[0];
#endif
}

__device__ double __sum_block_sync(double v) {
#if OPT < 1 // Original
    __shared__ double ret;
    if(threadIdx.x == 0) ret = 0;
    for(int i=0; i<blockDim.x; i++) {
        __syncthreads();
        if(i == threadIdx.x) ret += v;
    }
    __syncthreads();
    return ret;

#else // Parallel reduction
    __shared__ double vv[1024];
    vv[threadIdx.x] = v;
    for(unsigned stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) vv[threadIdx.x] += vv[threadIdx.x+stride];
    }
    return vv[0];
#endif
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

#if OPT < 6
__device__ double modf_p(double v, int *p) {
    double tfrac, tint;
    tfrac = modf(v, &tint);
    *p += ((int) tint);
    return tfrac;
}

#else
__device__ double modf_p(double v, bool *p) {
    double tfrac, tint;
    tfrac = modf(v, &tint);
    *p ^= 1 & ((int) tint);
    return tfrac;
}
#endif

__global__ void collatz_delay_kernel(int *ret, unsigned K, unsigned *ms, unsigned *ns, unsigned limit) {
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

#if OPT < 5
    __shared__ double ranks[1024];
    __shared__ int rankp[1024];
    __shared__ int termp[1024];

#elif OPT < 6
    __shared__ unsigned long long ranks[1024];
    __shared__ int rankp[1024];
    __shared__ int termp[1024];

#elif OPT < 7
    __shared__ unsigned ranks[1024];
    __shared__ bool rankp[1024];

#elif OPT < 8
    __shared__ unsigned fracs[1024];
    __shared__ bool parity[1024];

#else
    __shared__ unsigned rfracs[32];
    __shared__ bool parity[32];
    if (threadIdx.x < 32) {
        rfracs[threadIdx.x] = 0;
        parity[threadIdx.x] = 0;
    }
#endif

    while(limit == 0 || stops < limit) {

        bool is_one = true;
        for(int j=0; j<BLOCK_MULT; j++)
            is_one &= n[j] == 1;
        is_one = __all_block_sync(is_one);

#if OPT < 3 // Original
        bool is_even; {
            double rank = ((double) n[0]) * M_[0] / m[0];
            double rank_total = __sum_block_sync(rank) + 1e-3;
            bool rank_parity = 1 & (int) floor(rank_total);

            bool term_parity = (((unsigned long) n[0]) * M_[0]) % 2;
            bool terms_parity = __xor_block_sync(term_parity);

            bool parity = rank_parity ^ terms_parity;
            is_even = !parity;
        }

#elif OPT < 4 // Rank float destruct; (80089a) N <= ?
        bool is_even; {
            double rank = ((double) n[0]) * (((double) M_[0]) / m[0]);
            double tint, tfrac;
            tfrac = modf(rank, &tint);
            rank = (((int) tint) % 2) + tfrac;
            double rank_total = __sum_block_sync(rank) + 1e-3;
            bool rank_parity = 1 & (int) floor(rank_total);

            bool term_parity = (((unsigned long) n[0]) * M_[0]) % 2;
            bool terms_parity = __xor_block_sync(term_parity);

            bool parity = rank_parity ^ terms_parity;
            is_even = !parity;
        }

#elif OPT < 5 // Aggressive rank float dustruct; (695f0f5) N <= 512
        bool is_even; {
            double rank = ((double) M_[0]) / (double) m[0];
            int pp = 0;
            rank = modf_p(rank, &pp);
            rank = modf_p(((double) n[0]) * rank, &pp);

            ranks[threadIdx.x] = rank;
            rankp[threadIdx.x] = pp;
            for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // TODO unroll last warp
                if (threadIdx.x < stride) {
                    double r = ranks[threadIdx.x] + ranks[threadIdx.x + stride];
                    int p = rankp[threadIdx.x] + rankp[threadIdx.x + stride];
                    r = modf_p(r, &p);
                    ranks[threadIdx.x] = r;
                    rankp[threadIdx.x] = p;
                }
                __syncthreads();
            }
            if(threadIdx.x == 0) ranks[0] = modf_p(ranks[0] + 1e-3, &rankp[0]);
            __syncthreads();
            int rank_parity = rankp[0];

            int term_parity = (((unsigned long) n[0]) * M_[0]) % 2;
            termp[threadIdx.x] = term_parity;
            for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // TODO unroll last warp
                if (threadIdx.x < stride) {
                    termp[threadIdx.x] += termp[threadIdx.x + stride];
                }
                __syncthreads();
            }
            bool terms_parity = 1 & termp[0];

            bool parity = 1 & (rank_parity ^ terms_parity);
            is_even = !parity;
        }

#elif OPT < 6 // Fixed point rank; (392af8) <= 1024
        bool is_even; {
            double rank = ((double) M_[0]) / (double) m[0];
            int pp = 0;
            rank = modf_p(rank, &pp);
            rank = modf_p(((double) n[0]) * rank, &pp);

            unsigned long long rank_fracs = rank * (1ULL<<63);

            ranks[threadIdx.x] = rank_fracs;
            rankp[threadIdx.x] = pp;
            for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // TODO unroll last warp
                __syncthreads();
                if (threadIdx.x < stride) {
                    unsigned long long r = ranks[threadIdx.x] + ranks[threadIdx.x + stride];
                    int p = rankp[threadIdx.x] + rankp[threadIdx.x + stride];
                    ranks[threadIdx.x] = r & 0x7FFFFFFFFFFFFFFFULL; // r & ((1ULL<<63)-1);
                    rankp[threadIdx.x] = p + (1 & (r >> 63));
                }
                __syncthreads();
            }
            if(threadIdx.x == 0) {
                ranks[0] += 0x0010000000000000ULL;
                rankp[0] += 1 & (ranks[0] >> 63);
                ranks[0] &= 0x7FFFFFFFFFFFFFFFULL; // r & ((1ULL<<63)-1);
            }
            __syncthreads();
            //double rank_total = ((double) ranks[0]) / ((double) (1ULL<<63));
            int rank_parity = rankp[0];

            __syncthreads();

            int term_parity = (((unsigned long) n[0]) * M_[0]) % 2;
            termp[threadIdx.x] = term_parity;
            for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // TODO unroll last warp
                if (threadIdx.x < stride) {
                    termp[threadIdx.x] += termp[threadIdx.x + stride];
                }
                __syncthreads();
            }
            bool terms_parity = 1 & termp[0];

            bool parity = 1 & (rank_parity ^ terms_parity);
            is_even = !parity;
        }

#elif OPT < 7 // Single reduction parity calculation; (880742a894f) N <= 1024
        bool is_even; {
            double rank = ((double) M_[0]) / (double) m[0];
            bool pp = 0;
            rank = modf_p(rank, &pp);
            rank = modf_p(((double) n[0]) * rank, &pp);
            unsigned rank_fracs = rank * (1U<<31);
            bool term_parity = (1 & n[0]) * (1 & M_[0]);

            ranks[threadIdx.x] = rank_fracs;
            rankp[threadIdx.x] = pp ^ term_parity;

            for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) { // TODO unroll last warp
                __syncthreads();
                if (threadIdx.x < stride) {
                    unsigned r = ranks[threadIdx.x] + ranks[threadIdx.x + stride];
                    bool p = rankp[threadIdx.x] ^ rankp[threadIdx.x + stride];
                    ranks[threadIdx.x] = r & 0x7FFFFFFFU; // r & ((1ULL<<63)-1);
                    rankp[threadIdx.x] = p ^ (1 & (r >> 31));
                }
            }
            if(threadIdx.x == 0) {
                ranks[0] += 0x00100000U;
                rankp[0] ^= 1 & (ranks[0] >> 31);
                ranks[0] &= 0x7FFFFFFFU; // r & ((1ULL<<63)-1);
            }
            __syncthreads();

            bool parity = rankp[0];
            is_even = !parity;
        }

#elif OPT < 8 // Rank reduction with loop unroll; (824e934930) N <= 1024
        bool is_even; {
            double rank = ((double) M_[0]) / (double) m[0];
            bool pp = 0;
            rank = modf_p(rank, &pp);
            rank = modf_p(((double) n[0]) * rank, &pp);
            unsigned rank_fracs = rank * (1U<<31);
            bool term_parity = (1 & n[0]) * (1 & M_[0]);

            fracs[threadIdx.x] = rank_fracs;
            parity[threadIdx.x] = pp ^ term_parity;

            for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) { // TODO unroll last warp
                __syncthreads();
                if (threadIdx.x < stride) {
                    unsigned f = fracs[threadIdx.x] + fracs[threadIdx.x + stride];
                    bool p = parity[threadIdx.x] ^ parity[threadIdx.x + stride];
                    fracs[threadIdx.x] = f & 0x7FFFFFFFU; // f & ((1ULL<<63)-1);
                    parity[threadIdx.x] = p ^ (f >> 31);
                }
            }
            if (threadIdx.x < 32) {

                unsigned stride = 32;
                unsigned f = fracs[threadIdx.x] + fracs[threadIdx.x + stride];
                bool p = parity[threadIdx.x] ^ parity[threadIdx.x + stride];
                p ^= (f >> 31);
                f &= 0x7FFFFFFFU;

                stride = 16; {
                    f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                    p ^= (f >> 31);
                    f &= 0x7FFFFFFFU;
                }

                stride = 8; {
                    f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                    p ^= (f >> 31);
                    f &= 0x7FFFFFFFU;
                }

                stride = 4; {
                    f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                    p ^= (f >> 31);
                    f &= 0x7FFFFFFFU;
                }
                stride = 2; {
                    f += __shfl_down_sync(0xFFFFFFFF, f, stride, 32);
                    p ^= __shfl_down_sync(0xFFFFFFFF, p, stride, 32);
                    p ^= (f >> 31);
                    f &= 0x7FFFFFFFU;
                }

                stride = 1; {
                    f += __shfl_down_sync(0xFFFFFFF, f, 1, 32);
                    p ^= __shfl_down_sync(0xFFFFFFF, p, 1, 32);
                    f += 0x00100000U;
                    p ^= (f >> 31);
                }

                if(threadIdx.x == 0) parity[0] = p;
            }

            __syncthreads();

            is_even = !parity[0];
        }

#elif OPT < 9 // Rank reduction using warp synchronization with loop unroll; (f4d9bc9137) N <= 1024
        bool is_even; {
            double rank = ((double) M_[0]) / (double) m[0];
            bool pp = 0;
            rank = modf_p(rank, &pp);
            rank = modf_p(((double) n[0]) * rank, &pp);
            unsigned rank_rfracs = rank * (1U<<31);
            bool term_parity = (1 & n[0]) * (1 & M_[0]);

            unsigned f = rank_rfracs;
            bool p = pp ^ term_parity;

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

#elif OPT < 10 // Multi value for thread on register; (1e44b32710) N <= 2048
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
//#elif OPT < 11 // Multi value for thread on shared memory; (1e44b32710) N <= 2048
#endif

        if (is_one) break;

        if(is_even) {
#if LOG > 0
            if(threadIdx.x==0) printf("0 ");
#endif
            for(int j=0; j<BLOCK_MULT; j++) {
                n[j] = div2(m[j], n[j]); n[j] %= m[j];
            }
        } else {
#if LOG > 0
            if(threadIdx.x==0) printf("1 ");
#endif
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

#if LOG > 0
    if(threadIdx.x==0) printf("\n");
#endif

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

void device_collatz_kernel(int *delay, struct collatz_delay_t *A, unsigned limit, float *time) {
    assert(A->K % BLOCK_MULT == 0); // FIXME
    int blockDim = saturating_div(A->K, BLOCK_MULT);
    int gridDim = A->T;

    int *device_delay;
    CUDACHECK( cudaMalloc((void **)&device_delay, A->T * sizeof(int)) );

    CUDACHECK( cudaEventRecord(A->start, 0) );
#if LOG > 0
    printf("collatz_delay_kernel<<<%d,%d>>>(ret, K=%d, ms, ns, limit=%d);\n", gridDim,blockDim,A->K,limit);
#endif
    cudaError_t err = cudaGetLastError();
    collatz_delay_kernel<<<gridDim,blockDim>>>(device_delay, A->K, A->device_ms, A->device_ns, limit);
    CUDACHECK( cudaEventRecord(A->stop, 0) );

    CUDACHECK( cudaMemcpy(delay, device_delay, A->T * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDACHECK( cudaMemcpy(A->host_ns, A->device_ns, A->K * sizeof(unsigned), cudaMemcpyDeviceToHost) );

    CUDACHECK( cudaFree(device_delay) );

    float ms;
    cudaEventElapsedTime(&ms, A->start, A->stop);
    if(time != NULL) *time += ms;
}

void device_collatz_delay(int *delay, unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, unsigned limit, float *time) {
    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, T, n);
    device_collatz_kernel(delay, &A, limit, time);
    device_collatz_free(&A);
}

// T = 1 assumed. but for some reason, I have to get 'n' as a pointer
void device_collatz_steps(unsigned K, unsigned *host_ms, mpz_t *n, int repeat) {
    gmp_printf("%d G| -> %Zd\n", 0, *n);
    struct collatz_delay_t A;
    device_collatz_init(&A, K, host_ms, 1, n);
    int ret;
    for(int i=1; i<=repeat; i++) {
        device_collatz_kernel(&ret, &A, 1, NULL);
        mpz_t x; mpz_init(x);
        host_crt2int(x, K, host_ms, A.host_ns);
        gmp_printf("%d G| -> %Zd\n", i, x);
    }
    device_collatz_free(&A);
}

void test_compare_str(unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, int sel, unsigned limit) {
    int *delay_host = (int*) malloc(T * sizeof(int));
    memset(delay_host, 0, T * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<T; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu);
    gmp_printf("cpu time: \t%f ms\t\n", time_spent_cpu);

    int *delay_device = (int*) malloc(T * sizeof(int));
    memset(delay_device, 0, T * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, T, n, limit, &time_spent_gpu);
    gmp_printf("gpu time: \t%f ms\n", time_spent_gpu);

    for(int t=0; t<T; t++) // if(delay_host!=delay_device)
        gmp_printf(": %3d %c %3d\n", delay_host[t], (delay_host[t]==delay_device[t])?' ':'|', delay_device[t]);
}

void test_compare_str_(unsigned K, unsigned *host_ms, unsigned T, mpz_t *n, int sel, unsigned limit) {
    int *delay_host = (int*) malloc(T * sizeof(int));
    memset(delay_host, 0, T * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<T; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu);

    int *delay_device = (int*) malloc(T * sizeof(int));
    memset(delay_device, 0, T * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, T, n, limit, &time_spent_gpu);

    gmp_printf("%d\t%f\t%f\n", T, time_spent_cpu, time_spent_gpu);
}

void test_compare_str__(unsigned K, unsigned *host_ms, unsigned N, mpz_t *n, int sel, unsigned limit) {
    int *delay_host = (int*) malloc(1 * sizeof(int));
    memset(delay_host, 0, 1 * sizeof(int));
    float time_spent_cpu = 0;
    if(sel & 1) for(int t=0; t<1; t++) delay_host[t] = host_collatz_delay(n[t], limit, &time_spent_cpu);

    int *delay_device = (int*) malloc(1 * sizeof(int));
    memset(delay_device, 0, 1 * sizeof(int));
    float time_spent_gpu = 0;
    if(sel & 2) device_collatz_delay(delay_device, K, host_ms, 1, n, limit, &time_spent_gpu);

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
        test_compare_str__(K, host_ms, 1, &n, sel, limit);
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

int _main(int argc, char **argv) {

    char *num = 
        #include "num"
    ;

    int T = 1 << atoi(argv[1]);

    mpz_t nums[T];
    for(int t=0; t<T; t++) {
        mpz_init(nums[t]);
        mpz_set_str(nums[t], num, 10);
    }

    //test_compare_str_(1024, host_ms10000, T, nums, 3, 148624);
    test_compare_str_(2048, host_ms10000, T, nums, 2, 300982);

    return 0;
}

char num[13000];
int __main(int argc, char **argv) {

    char *_num =
        #include "num"
    ;

    int N = 1 << atoi(argv[1]);
    //int N = 12800;
    strncpy(num, _num, N);

    mpz_t nums;
    mpz_init(nums);
    mpz_set_str(nums, num, 10);

    //test_compare_str__(1024, host_ms10000, N, &nums, 3, 148624);
    test_compare_str__(2048, host_ms10000, N, &nums, 3, 300982);

    return 0;
}
