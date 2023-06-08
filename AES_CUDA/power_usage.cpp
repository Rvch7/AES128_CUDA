#include "power_usage.h"


nvmlReturn_t nvmlResult;
nvmlDevice_t device;
nvmlReturn_t result;
bool powerstate = 0;
std::thread power_usage_thread;
std::mutex startMutex;
std::condition_variable startCondition;



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


