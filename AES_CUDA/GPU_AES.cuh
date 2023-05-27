#pragma once
#include "AES_CUDA.h"
//__device__ extern BYTE gSBOX[256];


struct gword {
    BYTE byte[4];
    __host__ __device__ gword operator^(gword x) {
        gword z;
        z.byte[0] = x.byte[0] ^ this->byte[0];
        z.byte[1] = x.byte[1] ^ this->byte[1];
        z.byte[2] = x.byte[2] ^ this->byte[2];
        z.byte[3] = x.byte[3] ^ this->byte[3];
        return z;
    }
};

struct gblock_t {
    gword gstate[4] = {};
    __host__ __device__ gblock_t operator^(gblock_t x) {
        gblock_t z;
        z.gstate[0] = x.gstate[0] ^ this->gstate[0];
        z.gstate[1] = x.gstate[1] ^ this->gstate[1];
        z.gstate[2] = x.gstate[2] ^ this->gstate[2];
        z.gstate[3] = x.gstate[3] ^ this->gstate[3];
        return z;
    }
};


__device__ void gpu_addroundkey(gblock_t* block, gblock_t* expandedkeys);
__device__ void gpu_sbox_substitute(gblock_t* block);
__device__ void gpu_shift_rows(gblock_t* block);
__device__ void gpu_mix_columns(gblock_t* block);
__global__ void aes_kernal(gblock_t* block, gblock_t* expandedkeys);