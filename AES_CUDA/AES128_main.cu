#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>

#include "AES128_main.h"
#include "key_expansion.h"
#include "power_usage.h"

int main()
{
    cudaEvent_t start, stop;
    cudaError_t ret;

    const unsigned int nStreams = 10;
    cudaStream_t streams[nStreams];

    for (int i = 0; i < nStreams; ++i) {
        ret = cudaStreamCreate(&streams[i]);
        if (ret != cudaSuccess) { printf("CUDA: error creating streams"); return -1; };
    }
    


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int copyEngines = deviceProp.asyncEngineCount;



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
    int NumofThrds = 512;
    int streamSize = NumofBlocks / nStreams;

    int cpuLength = 0.07 * NumofBlocks;
    int gpuLength = NumofBlocks - cpuLength;

    //  host block allocations
    block_t* textblocks = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    if (textblocks == NULL) { printf("error allocation textblocks"); return -1; }
    block_t* encryptedtext = (block_t*)calloc(NumofBlocks, sizeof(block_t));
    if (encryptedtext == NULL) { printf("error allocation textblocks"); return -1; }

    
    //for (BYTE i = 0; i < 32; i++) {
    //    int ret = fscanf(pInFile, "%hhx", &textblocks->state->byte[i]); //Read the private key in -  fscanf reads files and saves space sperated hex values to a local memory
    //}
    
    // copy text to local memory
    fread(textblocks, sizeof(BYTE), file_len, pInFile);


    // key expansion to create a key for each round
    key_expansion(key, (block_t*)expandedkeys);




    // device blocks allocations
    block_t* d_textblocks;
    block_t* d_expandedkeys;
    ret = cudaMalloc(&d_textblocks,(sizeof(block_t)*NumofBlocks));
    if( ret != cudaSuccess) { printf("CUDA: error allocation d_textblocks"); return -1; }
    ret =  cudaMalloc(&d_expandedkeys, (sizeof(block_t) * NUMOFKEYS));
    if (ret != cudaSuccess) { printf("CUDA: error allocation d_expandedkeys"); return -1; }
    ret = cudaMemcpy(d_expandedkeys, expandedkeys, (sizeof(block_t) * NUMOFKEYS), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_expandedkeys"); return -1; };


    /// <summary> V1 - src: https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
    /// 2 streams - 150ms
    /// 10 streams - 160 ms
    /// no improvement from baseline - No streams
    /// </summary>
    /// <returns></returns>
    /*for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        ret = cudaMemcpyAsync((d_textblocks + offset), (textblocks + offset), (sizeof(block_t) * streamSize), cudaMemcpyHostToDevice, streams[i]);
        if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_textblocks"); return -1; };
        gpu_cipher<<<(NumofBlocks / (nStreams*NumofThrds)), NumofThrds,0,streams[i]>>> ((d_textblocks+offset), d_expandedkeys);
        //gpu_cipher<<<1, 70,0,streams[i]>>> ((d_textblocks+offset), d_expandedkeys);
        cudaMemcpyAsync((textblocks + offset), (d_textblocks + offset), (sizeof(block_t) * streamSize), cudaMemcpyDeviceToHost, streams[i]);
    }*/

    int cudaBlockSize = ceil((float) NumofBlocks / (nStreams * NumofThrds));

    nvml_start();

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        ret = cudaMemcpyAsync((d_textblocks + offset), (textblocks + offset), (sizeof(block_t) * streamSize), cudaMemcpyHostToDevice, streams[i]);
        if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_textblocks"); return -1; };
    }

    cudaEventRecord(start);
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        gpu_cipher <<<cudaBlockSize, NumofThrds, 0, streams[i] >>> ((d_textblocks + offset), d_expandedkeys);
    }
    cudaEventRecord(stop);

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaMemcpyAsync((textblocks + offset), (d_textblocks + offset), (sizeof(block_t)* streamSize), cudaMemcpyDeviceToHost, streams[i]);
    }


    nvml_stop();



    //// send data: host to device
    //cudaMemcpy(d_textblocks, textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyHostToDevice);
    //if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_textblocks"); return -1; }
    //cudaMemcpy(d_expandedkeys, expandedkeys, (sizeof(block_t) * NUMOFKEYS), cudaMemcpyHostToDevice);
    //if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_expandedkeys"); return -1; }

    ////cpu_cipher(textblocks, (block_t*)expandedkeys);
    //// GPU cipher kernal 
    //nvml_start();
    //cudaEventRecord(start);
    //gpu_cipher <<<(NumofBlocks/NumofThrds), NumofThrds >>> (d_textblocks, d_expandedkeys); // A round key for single block --  later used for cuda
    //cudaDeviceSynchronize();
    //cudaEventRecord(stop);
    //nvml_stop();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time elapsed: %f", milliseconds);

    //// send data: device to host
    //cudaMemcpy(textblocks, d_textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyDeviceToHost);
    //if (ret != cudaSuccess) { printf("CUDA: error copy DTOH textblocks"); return -1; }

    cudaFree(d_expandedkeys);
    cudaFree(d_textblocks);

    

};