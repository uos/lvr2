#ifndef KERNELS_HOST
#define KERNELS_HOST

// morton codes
void morton_codes_host(unsigned long long int* mortonCodes, 
                       float* points, 
                       int num_points);

// sorting
void radix_sort(unsigned long long int* keys, 
                int* values, 
                size_t num_points);


#endif // KERNELS_HOST