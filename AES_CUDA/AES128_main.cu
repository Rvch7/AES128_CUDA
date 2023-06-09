#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <thread>


#include "AES128_main.h"
#include "key_expansion.h"
#include "power_usage.h"
#include "CPU_thread.h"


int main()
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaError_t ret;
    block_t* key = (block_t*)malloc(sizeof(block_t));                           // storage - key    
    if (key == NULL) { printf("error key allocation"); return -1; } 
    word* expandedkeys = (word*)malloc(sizeof(word) * (Nb * (Nr + 1))); 
    if (expandedkeys == NULL) { printf("error text allocation"); return -1; }   // storage for expanded key - 44 words
    
    
    // Read pain text and key files
    FILE* pKeyFile = fopen("./key.txt", "rb");                          
    if (pKeyFile == NULL) { printf("error opening key file"); return -1; }
    for (BYTE i = 0; i < 16; i++) {
        int ret = fscanf(pKeyFile, "%hhx", &key->state->byte[i]); //Read the private key in -  fscanf reads files and saves space sperated hex values to a local memory
    }


    FILE* pInFile = fopen("./text.txt", "rb");                          
    if (pInFile == NULL) { printf("error opening input file"); return -1; }

    // Find plain text file size
    fseek(pInFile, 0, SEEK_END);
    float file_len = ftell(pInFile);
    rewind(pInFile);

    // Find 16-byte blocks in plain text files
    int NumofBlocks = ceil(file_len / BLOCKSIZE);
    //int NumofThrds = 512;

    //  host block allocations
    block_t* textblocks = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    if (textblocks == NULL) { printf("error allocation textblocks"); return -1; }
    block_t* encryptedtext = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    if (encryptedtext == NULL) { printf("error allocation textblocks"); return -1; }
    
    // copy text to local memory
    fread(textblocks, sizeof(BYTE), file_len, pInFile);


    // key expansion to create a key for each round
    key_expansion(key, (block_t*)expandedkeys);

    auto startTime = std::chrono::high_resolution_clock::now();
    spawn_threads(textblocks, (block_t*)expandedkeys, NumofBlocks);
    finish_all_threads();
    auto endTime = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print the duration
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    free(textblocks);
    free(expandedkeys);

};