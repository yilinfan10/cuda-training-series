#include <stdio.h>

__global__ void hello(){

  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

int main(){
  constexpr int kBlockSize = 8;
  constexpr int kNumBlocks = 2;

  hello<<<kNumBlocks, kBlockSize>>>();
  cudaDeviceSynchronize();
}

