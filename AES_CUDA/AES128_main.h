#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <nvml.h>
#include <thread>
#include <mutex>

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__ 
#else
#define CONSTANT const
#endif

// Definations
typedef unsigned char BYTE;
extern CONSTANT BYTE SBOX[256];

#define BLOCKSIZE 16
#define NUMOFKEYS 11

const char Nk = 4;	// Number of keys
const char Nb = 4;	// Number of words in a block
const char Nr = 10; // Number of rounds

struct word {
    BYTE byte[4];
    __host__ __device__ word operator^(word x) {
        word z;
        z.byte[0] = x.byte[0] ^ this->byte[0];
        z.byte[1] = x.byte[1] ^ this->byte[1];
        z.byte[2] = x.byte[2] ^ this->byte[2];
        z.byte[3] = x.byte[3] ^ this->byte[3];
        return z;
    }
};

struct block_t {
    union {
        word state[4] = {};
        uint64_t state64[2];
    };
    //word state[4] = {};
    __host__ __device__ block_t operator^(block_t x) {
        block_t z;
        z.state[0] = x.state[0] ^ this->state[0];
        z.state[1] = x.state[1] ^ this->state[1];
        z.state[2] = x.state[2] ^ this->state[2];
        z.state[3] = x.state[3] ^ this->state[3];
        return z;
    }
};

__host__ __device__ BYTE xtimes(BYTE x);
__host__ __device__ void addroundkey(block_t* block, block_t* expandedkeys);
__host__ __device__ void sbox_substitute(block_t* block, BYTE* SBOX);
__host__ __device__ void shift_rows(block_t* block);
__host__ __device__ void mix_columns(block_t* block);

__global__ void gpu_cipher(block_t* block, block_t* expandedkeys);

void cpu_cipher(block_t* block, block_t* expandedkeys);
void cpu_cipher_text(block_t* text, block_t* expandedkeys, int NumberOfBlocks);