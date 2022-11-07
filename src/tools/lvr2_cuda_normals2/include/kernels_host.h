#ifndef KERNELS_HOST
#define KERNELS_HOST

// morton codes
// void morton_codes_host(unsigned long long int* mortonCodes, 
//                        float* points, 
//                        int num_points);

// sorting
void radix_sort(unsigned long long int* keys, 
                int* values, 
                unsigned long num_points);


// building tree
void build_lbvh(float* points, 
                unsigned long num_points,
                float* queries,
                size_t num_queries,
                float* args,
                const char* kernel);

#endif // KERNELS_HOST