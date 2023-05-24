#include "AES_CUDA.h"

void cpu_add_roundkey(block_t* block, block_t* expanded_keys) { // ADD round key with a block
	for (int i = 0; i < 4; i++) {
		block->state[i] = block->state[i] ^ expanded_keys->state[i];
	}
}


void cpu_sbox_substitute(block_t* block) { //Performs an S-box substitution on a block
	for (int i = 0; i < BLOCKSIZE; i++) { 
		block->state->byte[i] = SBOX[block->state->byte[i]];
	}
}

void cpu_shift_rows(block_t* block) {  // Performs shift rows operation on a block
    block_t out;
    //On per-row basis (+1 shift X each row)
    //Row 1
    out.state->byte[0] = block->state->byte[0];
    out.state->byte[4] = block->state->byte[4];
    out.state->byte[8] = block->state->byte[8];
    out.state->byte[12] = block->state->byte[12];
    //Row 2
    out.state->byte[1] = block->state->byte[5];
    out.state->byte[5] = block->state->byte[9];
    out.state->byte[9] = block->state->byte[13];
    out.state->byte[13] = block->state->byte[1];
    //Row 3
    out.state->byte[2] = block->state->byte[10];
    out.state->byte[6] = block->state->byte[14];
    out.state->byte[10] = block->state->byte[2];
    out.state->byte[14] = block->state->byte[6];
    //Row 4
    out.state->byte[3] = block->state->byte[15];
    out.state->byte[7] = block->state->byte[3];
    out.state->byte[11] = block->state->byte[7];
    out.state->byte[15] = block->state->byte[11];

    for (int i = 0; i < BLOCKSIZE; i++)
    {
        block->state->byte[i] = out.state->byte[i];
    }
}


void cpu_mix_columns(block_t* block) {
    block_t out;

    for (int i = 0; i < BLOCKSIZE; i += 4) {
        out.state->byte[i + 0] = xtimes(block->state->byte[i + 0]) ^ xtimes(block->state->byte[i + 1]) ^ block->state->byte[i + 1] ^ block->state->byte[i + 2] ^ block->state->byte[i + 3];
        out.state->byte[i + 1] = block->state->byte[i + 0] ^ xtimes(block->state->byte[i + 1]) ^ xtimes(block->state->byte[i + 2]) ^ block->state->byte[i + 2] ^ block->state->byte[i + 3];
        out.state->byte[i + 2] = block->state->byte[i + 0] ^ block->state->byte[i + 1] ^ xtimes(block->state->byte[i + 2]) ^ xtimes(block->state->byte[i + 3]) ^ block->state->byte[i + 3];
        out.state->byte[i + 3] = xtimes(block->state->byte[i + 0]) ^ block->state->byte[i + 0] ^ block->state->byte[i + 1] ^ block->state->byte[i + 2] ^ xtimes(block->state->byte[i + 3]);
    }

    for (int i = 0; i < BLOCKSIZE; i++)
    {
        block->state->byte[i] = out.state->byte[i];
    }
}

void cpu_cipher(block_t* block, block_t* expandedkeys) {
    cpu_add_roundkey(block, expandedkeys);

    for (int round = 1; round < Nr; round++) {
        cpu_sbox_substitute(block);
        cpu_shift_rows(block);
        cpu_mix_columns(block);
        cpu_add_roundkey(block, (expandedkeys + round));
    }
    cpu_sbox_substitute(block);
    cpu_shift_rows(block);
    cpu_add_roundkey(block, (expandedkeys + Nr));
}

void cpu_cipher_text(block_t* text, block_t* expandedkeys, int NumberOfBlocks) {
    for (int i = 0; i <= NumberOfBlocks; i++) {
        cpu_cipher((text + i), expandedkeys);
    }

}