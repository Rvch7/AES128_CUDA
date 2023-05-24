#ifndef KEY_EXP_H
#define KEY_EXP_H
#include "AES_CUDA.h"

void key_expansionCore(word* in, unsigned i);
void key_expansion(block_t* key, block_t* expandedkeys);


#endif // !KEY_EXP_H

