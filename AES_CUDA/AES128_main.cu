#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cmath>
#include <thread>
#include <mutex>
#include <nvml.h>
#include "AES128_main.h"
#include "key_expansion.h"

nvmlReturn_t nvmlResult;
nvmlDevice_t device;
nvmlReturn_t result;
bool powerstate = 0;
std::thread power_usage_thread;
std::mutex startMutex;
std::condition_variable startCondition;


cudaEvent_t start, stop;



void power_usage_gpu() {
    unsigned int powerUsage = 0;
    startCondition.notify_one();

    // Get the power usage of the GPU
    while (powerstate) {

         result = nvmlDeviceGetPowerUsage(device, &powerUsage);
        if (result != NVML_SUCCESS) {
            printf("Failed to get power usage: %s\n", nvmlErrorString(result));
            nvmlShutdown();
            exit(0);
        }

        printf("Power Usage: %u mW\n", powerUsage);
    }


}

void nvml_start() {
    // Initialize NVML library
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        exit(0);
    }

    // since only one GPU is present, we'll use index 0
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        exit(0);
    }
    powerstate = 1;
    power_usage_thread = std::thread(power_usage_gpu);
    std::unique_lock<std::mutex> lock(startMutex);
    startCondition.wait(lock);

    
}

void nvml_stop() {
    //powerstate = 0;
    // Shutdown NVML library
    if (power_usage_thread.joinable()) {
        powerstate = 0;
        power_usage_thread.join();
    }
    result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
        exit(0);
    }
}




int main()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //std::thread power(power_usage_gpu);
    //power_usage_gpu();
    cudaError_t ret;
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
    if (pKeyFile == NULL) { printf("error opening input file"); return -1; }

    // Find plain text file size
    fseek(pInFile, 0, SEEK_END);
    float file_len = ftell(pInFile);
    rewind(pInFile);

    // Find 16-byte blocks in plain text files
    int NumofBlocks = ceil(file_len / BLOCKSIZE);
    int NumofThrds = 512;

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

    // send data: host to device
    cudaMemcpy(d_textblocks, textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_textblocks"); return -1; }
    cudaMemcpy(d_expandedkeys, expandedkeys, (sizeof(block_t) * NUMOFKEYS), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) { printf("CUDA: error copy HTOD d_expandedkeys"); return -1; }

    //cpu_cipher(textblocks, (block_t*)expandedkeys);
    // GPU cipher kernal 
    nvml_start();
    cudaEventRecord(start);
    gpu_cipher <<<(NumofBlocks/NumofThrds), NumofThrds >>> (d_textblocks, d_expandedkeys); // A round key for single block --  later used for cuda
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    nvml_stop();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time elapsed: %f", milliseconds);

    // send data: device to host
    cudaMemcpy(textblocks, d_textblocks, (sizeof(block_t) * NumofBlocks), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) { printf("CUDA: error copy DTOH textblocks"); return -1; }

    cudaFree(d_expandedkeys);
    cudaFree(d_textblocks);

    

};