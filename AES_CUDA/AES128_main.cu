#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>
#include "AES128_main.h"
#include "key_expansion.h"



__host__ __device__ BYTE xtimes(BYTE x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}



int main()
{
    block_t* key = (block_t*)malloc(sizeof(block_t));                           // storage for key    
    if (key == NULL) { printf("error key allocation"); return -1; } 
    block_t* textfile = (block_t*)malloc(sizeof(block_t));              
    if (textfile == NULL) { printf("error text allocation"); return -1; }       // storage for inputfile
    word* expandedkeys = (word*)malloc(sizeof(word) * (Nb * (Nr + 1))); 
    if (expandedkeys == NULL) { printf("error text allocation"); return -1; }   // storage for expanded key - 44 words
    
    
    FILE* pKeyFile = fopen("./key.txt", "rb");                          
    if (pKeyFile == NULL) { printf("error opening key file"); return -1; }
    FILE* pInFile = fopen("./text.txt", "rb");                          
    if (pKeyFile == NULL) { printf("error opening input file"); return -1; }

    fseek(pInFile, 0, SEEK_END);
    float file_len = ftell(pInFile);
    rewind(pInFile);
    int NumofBlocks = ceil(file_len / BLOCKSIZE);
    block_t* textblocks = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    block_t* encryptedtext = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    if (textblocks == NULL) { printf("error allocation textblocks"); return -1; }

    for (BYTE i = 0; i < 16; i++) {
        int ret = fscanf(pKeyFile, "%hhx", &key->state->byte[i]); //Read the private key in -  fscanf reads files and saves space sperated hex values to a local memory
    }

    for (BYTE i = 0; i < 32; i++) {
        int ret = fscanf(pInFile, "%hhx", &textblocks->state->byte[i]); //Read the private key in -  fscanf reads files and saves space sperated hex values to a local memory
    }
    //fread(textblocks, sizeof(BYTE), file_len, pInFile);

    key_expansion(key, (block_t*)expandedkeys);

    block_t* d_textblocks;
    block_t* d_expandedkeys;
    cudaMalloc(&d_textblocks,(sizeof(block_t)*NumofBlocks));
    cudaMalloc(&d_expandedkeys, (sizeof(block_t) * NUMOFKEYS));

    //cpu_cipher_text(textblocks, (block_t*)expandedkeys, NumofBlocks);
   
    cudaMemcpy(d_textblocks, textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expandedkeys, expandedkeys, (sizeof(block_t) * NUMOFKEYS), cudaMemcpyHostToDevice);

    cpu_cipher(textblocks, (block_t*)expandedkeys);
    gpu_cipher <<<(NumofBlocks/512), 512 >>> (d_textblocks, d_expandedkeys); // A round key for single block --  later used for cuda
    cudaMemcpy(textblocks, d_textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyDeviceToHost);


    

};