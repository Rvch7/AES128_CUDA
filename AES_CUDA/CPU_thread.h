#pragma once
#include "AES128_main.h"

void spawn_threads(block_t* textblocks, block_t* expandedkeys, int NumofBlocks);
void finish_all_threads();