#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Definations
typedef unsigned char BYTE;
extern const BYTE SBOX[256];

#define BLOCKSIZE 16
#define NUMOFKEYS 11

const char Nk = 4;	// Number of keys
const char Nb = 4;	// Number of words in a block
const char Nr = 10; // Number of rounds

struct word {
    BYTE byte[4];
        word operator^(word x) {
        word z;
        z.byte[0] = x.byte[0] ^ this->byte[0];
        z.byte[1] = x.byte[1] ^ this->byte[1];
        z.byte[2] = x.byte[2] ^ this->byte[2];
        z.byte[3] = x.byte[3] ^ this->byte[3];
        return z;
    }
};

struct block_t {
    word state[4] = {};
    block_t operator^(block_t x) {
        block_t z;
        z.state[0] = x.state[0] ^ this->state[0];
        z.state[1] = x.state[1] ^ this->state[1];
        z.state[2] = x.state[2] ^ this->state[2];
        z.state[3] = x.state[3] ^ this->state[3];
        return z;
    }
};

__host__ __device__ BYTE xtimes(BYTE x);