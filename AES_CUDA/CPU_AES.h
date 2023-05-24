#pragma once
void cpu_add_roundkey(block_t* block, block_t* expanded_keys);
void cpu_sbox_substitute(block_t* block);
void cpu_shift_rows(block_t* block);
void cpu_mix_columns(block_t* block);
void cpu_cipher(block_t* block, block_t* expandedkeys);
void cpu_cipher_text(block_t* text, block_t* expandedkeys, int NumberOfBlocks);