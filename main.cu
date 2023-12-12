// my old CRT.py code
// https://gist.github.com/bivoje/ad7d9c57c3ecb4673bf6dea6bbafe0b3 

#include <assert.h>
#include <stdio.h>
#include <gmp.h>

#define K 8
#define saturating_div(A,B) (((A) + (B) - 1) / (B))

//#include "primes.list"

void host_ui2crt(unsigned *ns, unsigned *ms, unsigned n) {
    for(int k=0; k<K; k++) ns[k] = n % ms[k];
}

void host_int2crt(unsigned *_ns, unsigned *_ms, mpz_t n) {
    mpz_t t, ms[K];
    mpz_init(t);
    for(int k=0; k<K; k++) mpz_init(ms[k]);
    for(int k=0; k<K; k++) mpz_set_ui(ms[k], _ms[k]);

    for(int k=0; k<K; k++) {
        mpz_mod_ui(t, n, _ms[k]);
        _ns[k] = mpz_get_ui(t);
    }
}

void host_crt2int(mpz_t n, unsigned *_ms, unsigned *_ns) {
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

int collatz_delay_host(mpz_t _n, int limit, bool log=false) {
    mpz_t n; mpz_init(n); mpz_set(n, _n);
    int stops = 0;
    while(stops < limit && mpz_cmp_ui(n, 1) != 0) {
        if(mpz_even_p(n)) {
            if(log) printf("0 ");
            mpz_divexact_ui(n, n, 2);
        } else {
            if(log) printf("1 ");
            mpz_mul_ui(n, n, 3);
            mpz_add_ui(n, n, 1);
        }
        stops += 1;
    }
    if(log) printf("\n");

    if(mpz_cmp_ui(n, 1) == 0)
        return stops;
    else
        return -1;
}

void test_collatz_host() {
    mpz_t n;
    mpz_init(n);
    mpz_set_str(n,
            "11"
        //#include "num"
        , 10);

    int delay = collatz_delay_host(n, 300000000);
    printf("%d\n", delay);
    mpz_clear(n);
}

void test_crt_conv() {
    unsigned _ms[K] = {10007,10009,10037,10039,10061,10067};
    unsigned _ns[K] = {0,};
    unsigned _n = 987654321;

    mpz_t n;
    mpz_set_ui(n, _n);

    host_int2crt(_ns, _ms, n);
    printf("ns = "); for(int k=0; k<K; k++) printf("%d, ", _ns[k]); printf("\n");
    host_crt2int(n, _ms, _ns);
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
__device__ float __sum_block_sync(float v) {
    __shared__ float vv[1024]; // = {0,};
    vv[threadIdx.x] = v;

    // TODO unroll last warp
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            vv[threadIdx.x] += vv[threadIdx.x + stride];
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

__global__ void collatz_delay(int *ret, unsigned *ms, unsigned *ns, int limit, bool log=false) {
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

    while(stops < limit) {

        bool is_one = __all_block_sync(n == 1);
        bool is_even; {
            double rank = ((double) n) * M_ / m;
            double rank_total = __sum_block_sync(rank) + 1e-3;
            bool rank_parity = 1 & (int) floor(rank_total);
            //printf("rank=%lf, rank_total=%lf, rank_parity=%d\n", rank, rank_total, rank_parity);

            bool term_parity = (((unsigned long) n) * M_) % 2;
            bool terms_parity = __xor_block_sync(term_parity);
            //printf("term=%d, terms=%d\n", term_parity, terms_parity);

            bool parity = rank_parity ^ terms_parity;
            is_even = !parity;
            //printf("parity=%d\n", parity);
        }

        if (is_one) break;

        if(is_even) {
            if(log) if(threadIdx.x==0) printf("0 ");
            n = div2(m, n); n %= m;
        } else {
            if(log) if(threadIdx.x==0) printf("1 ");
            // FIXME assume m < UINTMAX/3
            n *= 3; n += 1; n %= m;
        }
        stops += 1;
        if(log) ns[k] = n;
    }
    if(log) if(threadIdx.x==0) printf("\n");

    if(k == 0)
        if(stops < limit) 
            *ret = stops;
        else
            *ret = -1;
}

int collatz_delay_device(mpz_t n, int limit, bool log=false) {
    unsigned host_ms[K] = {3,5,7,11,13,17,19,10007};
    unsigned host_ns[K] = {0,};
    unsigned *device_ms;
    unsigned *device_ns;
    int host_delay;
    int *device_delay;

    host_int2crt(host_ns, host_ms, n);

    cudaMalloc((void **)&device_delay, sizeof(int));
    cudaMalloc((void **)&device_ms, K * sizeof(unsigned));
    cudaMalloc((void **)&device_ns, K * sizeof(unsigned));
    cudaMemcpy(device_ms, host_ms, K * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ns, host_ns, K * sizeof(unsigned), cudaMemcpyHostToDevice);

    int blockDim = K; // 1024;
    int gridDim = 1; //saturating_div(K, blockDim);

    collatz_delay<<<gridDim,blockDim>>>(device_delay, device_ms, device_ns, limit, log);

    cudaMemcpy(&host_delay, device_delay, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_delay);
    cudaFree(device_ns);
    cudaFree(device_ms);

    return host_delay;
}

void test_compare(int i, bool log) {
    mpz_t n;
    mpz_init(n);

        mpz_set_ui(n, i);
        int delay_host   = collatz_delay_host  (n, 200000, log);
        int delay_device = collatz_delay_device(n, 200000, log);
        if(delay_host != delay_device)
        printf("%d: %3d %c %3d\n", i, delay_host, (delay_host==delay_device)?' ':'|', delay_device);

    mpz_clear(n);
}

void collatz_step_device(mpz_t n) {
    unsigned host_ms[K] = {3,5,7,11,13,17,19,10007};
    unsigned host_ns[K] = {0,};
    unsigned *device_ms;
    unsigned *device_ns;
    int *device_delay;

    host_int2crt(host_ns, host_ms, n);

    cudaMalloc((void **)&device_delay, sizeof(int));
    cudaMalloc((void **)&device_ms, K * sizeof(unsigned));
    cudaMalloc((void **)&device_ns, K * sizeof(unsigned));
    cudaMemcpy(device_ms, host_ms, K * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(device_ns, host_ns, K * sizeof(unsigned), cudaMemcpyHostToDevice);

    int blockDim = K; // 1024;
    int gridDim = 1; //saturating_div(K, blockDim);

    collatz_delay<<<gridDim,blockDim>>>(device_delay, device_ms, device_ns, 1, true);

    cudaMemcpy(&host_ns, device_ns, K * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_delay);
    cudaFree(device_ns);
    cudaFree(device_ms);

    host_crt2int(n, host_ms, host_ns);
}

int main(int argc, char **argv) {
    for(int i=1; i<100000; i++) test_compare(i, false);

    //test_compare(27, true);

    //int _n = 1438;
    //mpz_t n; mpz_init(n); mpz_set_ui(n, _n);
    //collatz_step_device(n);
    //gmp_printf("%d -> %Zd\n", _n, n);

    return 0;
}
