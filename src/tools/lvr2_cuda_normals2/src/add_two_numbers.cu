
extern "C" __global__ 
void add_two_numbers_kernel(const int* a, const int* b, int* c, int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < N)
    {
        c[tid] = a[tid] + b[tid];
    }
}