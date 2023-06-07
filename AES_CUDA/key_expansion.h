#pragma once
#include "AES128_main.h"

void key_expansionCore(word* in, unsigned i);
void key_expansion(block_t* key, block_t* expandedkeys);