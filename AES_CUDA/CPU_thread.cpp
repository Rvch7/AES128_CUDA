#include "CPU_thread.h"
#include "AES128_main.h"
#include <thread>


const int CPUnumThreads = 16;
std::thread threads[CPUnumThreads];


void spawn_threads(block_t* textblocks, block_t* expandedkeys, int NumofBlocks) {
    int CPUthreadblocks = ((NumofBlocks + CPUnumThreads - 1) / CPUnumThreads);

    for (int i = 0; i < CPUnumThreads; ++i)
    {
        threads[i] = std::thread(cpu_cipher_text, (textblocks + (i * CPUthreadblocks)), (block_t*)expandedkeys, CPUthreadblocks);
    }

}


void finish_all_threads() {
    for (int i = 0; i < CPUnumThreads; ++i) {
        threads[i].join();
    }
}