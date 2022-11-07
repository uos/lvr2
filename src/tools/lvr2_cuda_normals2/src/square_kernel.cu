// Type your code here, or load an example.
extern "C" __global__ void square(int* array, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        array[tid] = array[tid] * array[tid];
}