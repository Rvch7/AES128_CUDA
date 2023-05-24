#include "AES_CUDA.h"
#include "key_expansion.h"


void key_expansionCore(word* in, unsigned i) {
    BYTE t = in->byte[0];
    in->byte[0] = in->byte[1];
    in->byte[1] = in->byte[2];
    in->byte[2] = in->byte[3];
    in->byte[3] = t;

    //S_box look up
    in->byte[0] = SBOX[in->byte[0]];
    in->byte[1] = SBOX[in->byte[1]];
    in->byte[2] = SBOX[in->byte[2]];
    in->byte[3] = SBOX[in->byte[3]];

    //R_con
    in->byte[0] ^= RCON[i];

}

void key_expansion(block_t* key, block_t* expandedkeys) {
    word temp;
    int keyGenerated = 0;
    int rconIteration = 1;

    while (keyGenerated < Nk) {
        expandedkeys->state[keyGenerated] = key->state[keyGenerated];
        keyGenerated++;
    }

    keyGenerated = Nk;

    while (keyGenerated < Nb * (Nr + 1)) {

        temp = expandedkeys->state[keyGenerated - 1];

        if (keyGenerated % Nk == 0)
            key_expansionCore(&temp, rconIteration++);

        expandedkeys->state[keyGenerated] = expandedkeys->state[keyGenerated - Nk] ^ temp;
        keyGenerated++;

    }


}